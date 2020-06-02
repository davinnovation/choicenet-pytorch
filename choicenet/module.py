import torch
import torch.nn as nn
from torchsummary import summary

MU_W_MEAN_INIT = 0
MU_W_STD_INIT = 0.1
VAR_W_INIT = -0.3
VAR_Z_INIT = -2
RHO_1 = 1.

class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class ResBlock(nn.Module):
  def __init__(self):
    super(ResBlock, self).__init__()

    self.conv1 = nn.Conv2d(1,64,3,padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu1 = nn.ReLU()
    self.maxpool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(64,64,3,padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.relu2 = nn.ReLU()
    self.maxpool2 = nn.MaxPool2d(2)
  
  def forward(self,x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out += identity
    out = self.relu1(out)
    out = self.maxpool1(out)

    identity = out

    out = self.conv2(out)
    out = self.bn2(out)
    out += identity
    out = self.relu2(out)
    out = self.maxpool2(out)


class MIXTURE_block(nn.Module):
    def __init__(self, feat_num, mixtures):
        self.f = nn.Sequential([
            nn.Linear(feat_num, mixtures),
            nn.Softmax()
        ])
    
    def forward(self, x):
        return self.f(x)

class VARIANCE_block(nn.Module):
    def __init__(self, feat_num, mixtures, tau_inv):
        self.f = nn.Sequential([
            nn.Linear(feat_num, mixtures),
            Lambda(lambda x: torch.exp(x))
        ])

        self.tau_inv = tau_inv
    
    def forward(self, x, rho):
        return self.f(x) * ( 1 - torch.exp(rho) ) + self.tau_inv

class MU_block(nn.Module):
    def __init__(self):
        self.f = self._cholesky
    
    def _cholesky(self, rho, mu_w:float, var_w:float, var_z:float):
        W = torch.distributions.Normal(mu_w, var_w).sample(rho.shape)
        Z = torch.distributions.Normal(0., var_z).sample(rho.shape)
        return rho * mu_w + torch.sqrt(1 - torch.pow(rho, 2)) * (rho * var_z/var_w * (W-mu_w) + Z * torch.sqrt(1 - torch.pow(rho, 2)))

    def forward(self, rho, mu_w:float, var_w:float, var_z:float, x):
        return self.f(rho, mu_w, var_w, var_z) * x

class MCDN(nn.Module):
    mode_dict = {
        'linear' : Lambda(lambda x: x),
        'regression' : nn.Tanh(),
        'classification' : nn.Sigmoid()
    }

    def _init_sample(self, feat_num, output, data_cnt, mu_w_mean=MU_W_MEAN_INIT, mu_w_std=MU_W_STD_INIT, var_w_cnt=VAR_W_INIT, var_z_cnt=VAR_Z_INIT, device='cpu'):
        mu_w = torch.nn.init.normal_(torch.empty(feat_num, output), mu_w_mean, mu_w_std).to(device)
        var_w = torch.nn.init.constant_(torch.empty(feat_num, output), var_w_cnt).to(device)
        mu_z = torch.zeros((feat_num, output)).to(device)
        var_z = torch.nn.init.constant_(torch.empty(feat_num, output), var_z_cnt).to(device)

        batch_f = lambda mt: mt.unsqueeze(0).repeat(data_cnt, 1, 1)

        return batch_f(mu_w), batch_f(var_w), batch_f(mu_z), batch_f(var_z)

    def __init__(self, feat_num:int, output:int, mixtures:int, tau_inv:int, mode='regression', device='cpu'):
        self.rho = nn.Sequential([
            nn.Linear(feat_num, mixtures),
            MCDN.mode_dict[mode],
            Lambda(lambda x: torch.cat([x[:,:1] * RHO_1, x[:,1:]], axis=1))
        ])

        self.mixture = MIXTURE_block(feat_num, mixtures)
        # self.mean = MU_block()
        self.variance = VARIANCE_block(feat_num, mixtures, tau_inv)

        self.feat_num = feat_num
        self.output = output
        self.device = device

    def forward(self, x):
        muW, varW, muZ, varZ = self._init_sample(self.feat_num, self.output, x.shape[0], self.device)

        rho = self.rho(x)
        mixture = self.mixture(x)
        variance = self.variance(x, rho)
        # mu = self.mean(rho, self.mu_w, self.var_w, self.var_z, x)
        W_ = self._cholesky(muW, varW, muZ, varZ, rho, x.shape[0], self.device)
        mu = W_ * x

    def _cholesky(self, muW, varW, muZ, varZ, rho, data_cnt, device):
        temp = []
        for idx in range(self.mixture):
            rho_t = rho[:,idx:idx]

            W = muW + torch.sqrt(varW)*torch.randn((data_cnt, self.feat_num, self.output), dtype=torch.float).to(device)
            Z = muZ + torch.sqrt(varZ)*torch.randn((data_cnt, self.feat_num, self.output), dtype=torch.float).to(device)

            chole = torch.mul(W, rho_t.unsqueeze(-1)) + torch.sqrt(1-torch.pow(rho_t, 2)) * (
                rho_t * torch.sqrt(varZ) / torch.sqrt(varW) * (W - muW) + Z * torch.sqrt(1-torch.pow(rho_t, 2))
            )
            temp.append(chole)
        return torch.stack(temp).transpose(0, 1)