import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision as tv

RHO_1 = 0.95
TAU_INV = 1e-4
MU_W_MEAN = 0
MU_W_STD = 0.1
VAR_W_CONST = -0.3
VAR_Z_CONST = -2

from torchsummary import summary

class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class VARIANCE_block(nn.Module):
    def __init__(self, feat_num, mixture_num, tau_inv):
        super(VARIANCE_block, self).__init__()
        self.h2var = nn.Linear(feat_num, mixture_num)
        self.var = nn.Sequential(
            self.h2var,
            Lambda(lambda x: torch.exp(x))
        )
        self.tau_inv = tau_inv

    def forward(self, feature, rho):
        var0 = self.var(feature) # B x mixture_num
        return (1-torch.exp(rho)) * var0 + self.tau_inv # B x mixture_num

class MDCN(nn.Module):
    def __init__(self, feat_num, mixture_num, tau_inv:float=TAU_INV, device='cuda:0'):
        super(MDCN, self).__init__()
        self.feat_num = feat_num
        self.mixture_num = mixture_num
    
        self.h2rho = nn.Linear(feat_num, mixture_num)
        self.h2pi = nn.Linear(feat_num, mixture_num)

        self.rho = nn.Sequential(
            self.h2rho,
            nn.Tanh(),
            Lambda(lambda x: torch.cat([x[:,:1]*0. + RHO_1, x[:,1:]], axis=1))
        )

        self.pi = nn.Sequential(
            self.h2pi,
            nn.Softmax()
        )

        self.var = VARIANCE_block(feat_num, mixture_num, tau_inv)

        self.device = device
    
    def forward(self, feature):
        W, Z, muW, logvarW, logvarZ = self._sample(self.feat_num, 
        MU_W_MEAN, MU_W_STD, VAR_W_CONST, VAR_Z_CONST, device=self.device)

        rho = self.rho(feature) # [B x mixtures]
        pi = self.pi(feature) # [B x mixtures]
        var = self.var(feature, rho) # [B x mixtures]

        W_ = self._cholesky(W, Z, rho, muW, logvarW, logvarZ, device=self.device)
        W_ = W_.permute(1,0,2) # B x M x F
        mu = torch.matmul(W_, feature.unsqueeze(-1)).squeeze(-1)

        return pi, mu, var
    
    def _sample(self, feat_num, mu_w_mean, mu_w_std, var_w_const, var_z_const, device):
        muW = D.Normal(mu_w_mean, mu_w_std).sample([feat_num]).to(device)
        logvarW = nn.init.constant_(torch.empty(feat_num), var_w_const).to(device)

        W = muW + torch.sqrt(torch.exp(logvarW)) * torch.randn(feat_num, dtype=torch.float).to(device)
        
        # muZ = torch.zeros(feat_num).to(device)
        logvarZ = nn.init.constant_(torch.empty(feat_num), var_z_const).to(device)

        Z = torch.sqrt(torch.exp(logvarZ)) * torch.randn(feat_num, dtype=torch.float).to(device)

        return W, Z, muW, logvarW, logvarZ
        
    def _cholesky(self, W, Z, rho, mu_, var_, zvar_, device):
        temp = []
        var_ = torch.sqrt(torch.exp(var_))
        zvar_ = torch.sqrt(torch.exp(zvar_))
        for idx in range(rho.shape[-1]): # K
            rho_i = rho[:, idx:idx+1]
            a1 = rho_i * mu_.unsqueeze(0).repeat(rho.shape[0], 1)
            a2 = torch.sqrt(1-torch.pow(rho_i,2))
            a3 = rho_i * zvar_/var_ * (W-mu_)
            a4 = Z * a2

            a = a1 + a2*(a3+a4)
            temp.append(a)
        return torch.stack(temp)

class TestM(nn.Module):
    def __init__(self):
        super(TestM, self).__init__()
        self.b = nn.Sequential(*list(tv.models.resnet18(False).children())[:-1])
        self.mcdn = MDCN(512, 10)
    
    def forward(self, x):
        f = self.b(x)
        f = f.view(f.size(0), -1)
        return self.mcdn(f)

summary(TestM().to("cuda:0"), (3, 256,256))