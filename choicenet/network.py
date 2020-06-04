import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision as tv

TAU_INV = 1e-4
MU_W_MEAN = 0
MU_W_STD = 0.1
VAR_W_CONST = -2.0
VAR_Z_CONST = 0

VAR_EPS = 1e-4
KL_REG_COEF = 1e-5

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
        self.var = nn.Sequential(
            nn.Linear(feat_num, mixture_num),
            nn.BatchNorm1d(mixture_num),
            nn.ReLU(),
            Lambda(lambda x: torch.exp(x))
        )
        self.tau_inv = tau_inv

    def forward(self, feature, rho):
        var0 = self.var(feature) # B x mixture_num
        return (1-torch.pow(rho, 2)) * var0 + self.tau_inv # B x mixture_num

class Rho_block(nn.Module):
    def __init__(self, feat_num, mixture_num):
        super(Rho_block, self).__init__()
        self.rho = nn.Sequential(
            nn.Linear(feat_num, mixture_num),
            nn.Tanh()
        )
        self.rho_end = lambda x, rho: torch.cat([x[:,:1]*0. + rho, x[:,1:]], axis=1)

    def forward(self, x, rho_1):
        return self.rho_end(self.rho(x), rho_1)

class MDCN(nn.Module):
    def __init__(self, feat_num, mixture_num, tau_inv:float=TAU_INV, device='cuda:0', PI1_BIAS=0.5):
        super(MDCN, self).__init__()
        self.feat_num = feat_num
        self.mixture_num = mixture_num

        self.start = [nn.Linear(feat_num, 16),nn.ReLU(),nn.BatchNorm1d(16)]
        self.modules = [nn.Linear(16, 16),nn.ReLU(),nn.BatchNorm1d(16)] * 16
        self.end = [nn.Linear(16, feat_num),nn.ReLU(),nn.BatchNorm1d(feat_num)]

        self.start += self.modules
        self.start += self.end

        self.feature_hdim = nn.Sequential(
            *self.start
        )

        self.rho = Rho_block(feat_num, mixture_num)

        self.pi = nn.Sequential(
            nn.Linear(feat_num, mixture_num),
            nn.BatchNorm1d(mixture_num),
            nn.ReLU(),
            nn.Softmax()
        )

        self.var = VARIANCE_block(feat_num, mixture_num, tau_inv)

        self.muW, self.logvarW, self.logvarZ = self._sample_init(self.feat_num, 
        MU_W_MEAN, MU_W_STD, VAR_W_CONST, VAR_Z_CONST)

        self.device = device

        self.mu_f = nn.Linear(feat_num, mixture_num)
    
    def forward(self, feature, rho_1=0.95):
        feature = self.feature_hdim(feature)

        rho = self.rho(feature, rho_1) # [B x mixtures]
        pi = self.pi(feature) # [B x mixtures]
        var = self.var(feature, rho) # [B x mixtures]

        W_ = self._cholesky(rho, self.muW, self.logvarW, self.logvarZ)
        W_ = W_.permute(1,0,2) # B x M x F
        mu = torch.matmul(W_, feature.unsqueeze(-1)).squeeze(-1)
        #mu = self.mu_f(feature)
        return rho, pi, mu, var
    
    def _sample_init(self, feat_num, mu_w_mean, mu_w_std, var_w_const, var_z_const):
        muW = nn.Parameter(torch.Tensor(feat_num))
        torch.nn.init.normal_(muW, mu_w_mean, mu_w_std)
        logvarW = nn.Parameter(torch.Tensor(feat_num))
        nn.init.constant_(logvarW, var_w_const)

        logvarZ = nn.Parameter(torch.Tensor(feat_num), requires_grad=False)
        nn.init.constant_(logvarZ, var_z_const)

        return muW, logvarW, logvarZ
    
    def _sample(self, batch_size, feat_num, muW, logvarW, logvarZ, device):
        W = muW.unsqueeze(0).repeat(batch_size, 1) + torch.sqrt(torch.exp(logvarW.unsqueeze(0).repeat(batch_size, 1))) * torch.randn((batch_size, feat_num), dtype=torch.float).to(device)
        Z = torch.sqrt(torch.exp(logvarZ.unsqueeze(0).repeat(batch_size, 1))) * torch.randn((batch_size, feat_num), dtype=torch.float).to(device)
        return W, Z

    def _cholesky(self, rho, mu_, var_, zvar_):
        temp = []
        var_ = torch.sqrt(torch.exp(var_))
        zvar_ = torch.sqrt(torch.exp(zvar_))
        batch_size = rho.shape[0]
        mu_ = mu_.unsqueeze(0).repeat(rho.shape[0], 1)
        zvar_ = zvar_.unsqueeze(0).repeat(batch_size, 1)
        var_ = var_.unsqueeze(0).repeat(batch_size, 1)
        for idx in range(rho.shape[-1]): # K
            W, Z = self._sample(batch_size, self.feat_num, self.muW, self.logvarW, self.logvarZ, self.device)
            rho_i = rho[:, idx:idx+1]
            a1 = rho_i * mu_
            a2 = 1-torch.pow(rho_i,2)
            a2 = a2.repeat(1, self.feat_num)
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
    
    def forward(self, x, rho=0.95):
        f = self.b(x)
        f = f.view(f.size(0), -1)
        return self.mcdn(f,rho)

def loss(pred, target):
    def mdnloss(pi, mu, var, target):
        quad = torch.pow(target.expand_as(mu) - mu, 2) * torch.reciprocal(var+VAR_EPS) * -0.5
        logdet = torch.log(var+VAR_EPS) * -0.5
        logconstant = torch.log(2*np.pi) * -0.5
        logpi = torch.log(pi)
        exponents = quad + logdet + logpi
        
        logprobs = torch.log(exponents)
        gmm_prob = torch.exp(logprobs)
        gmm_nll = -logprobs
        
        return gmm_nll

    def KLdiv(rho, pi):
        rho_pos = rho+1.
        kl_reg = KL_REG_COEF * (-rho_pos * torch.log(pi+1e-2) - torch.log(rho_pos+1e-2))
        return torch.mean(kl_reg)

    def MSE(pi, mu, target):
        fit_mse = 1e-2 * torch.pow(mu-target.expand_as(mu), 2)
        
    rho, pi, mu, var = pred
    
    l2 = MSE(pi, mu, target)
    mdn = mdnloss(pi, mu, var, target)
    kldiv = KLdiv(rho, pi)
    
    return l2 + mdn + kldiv

summary(TestM().to("cuda:0"), (3, 256,256))
