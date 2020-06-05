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
            nn.Linear(feat_num, 1),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        self.mixture_num = mixture_num
        self.tau_inv = tau_inv

    def forward(self, feature, rho):
        var0 = self.var(feature) # B x 1
        var0.repeat(1, self.mixture_num)
        return (1-torch.pow(rho, 2)) * var0 + self.tau_inv # B x mixture_num

class Rho_block(nn.Module):
    def __init__(self, feat_num, mixture_num):
        super(Rho_block, self).__init__()
        self.rho = nn.Sequential(
            nn.Linear(feat_num, mixture_num),
            nn.BatchNorm1d(mixture_num),
            nn.Sigmoid()
        )
        self.rho_end = lambda x, rho: torch.cat([x[:,:1]*0. + rho, x[:,1:]], axis=1)

    def forward(self, x, rho_1):
        return self.rho_end(self.rho(x), rho_1)

class MDCN(nn.Module):
    def __init__(self, feat_num, mixture_num, tau_inv:float=TAU_INV, device='cuda:0', PI1_BIAS=0.5):
        super(MDCN, self).__init__()
        self.feat_num = feat_num
        self.mixture_num = mixture_num

        self.rho = Rho_block(feat_num, mixture_num)

        self.muW, self.logvarW, self.logvarZ = self._sample_init(self.feat_num, 
        MU_W_MEAN, MU_W_STD, VAR_W_CONST, VAR_Z_CONST)
        
        self.pi = nn.Sequential(
            nn.Linear(feat_num, mixture_num),
            nn.BatchNorm1d(mixture_num),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

        self.var = VARIANCE_block(feat_num, mixture_num, tau_inv)

        self.device = device
    
    def forward(self, feature, rho_1=0.95):
        rho = self.rho(feature, rho_1) # [B x mixtures]
        W_ = self._cholesky(rho, self.muW, self.logvarW, self.logvarZ)
        W_ = W_.permute(1,0,2) # B x M x F
        mu = torch.matmul(W_, feature.unsqueeze(-1)).squeeze(-1)
        
        pi = self.pi(feature) # [B x mixtures]
        var = self.var(feature, rho) # [B x mixtures]

        return rho, pi, mu, var
    
    def _sample_init(self, feat_num, mu_w_mean, mu_w_std, var_w_const, var_z_const):
        muW = nn.Parameter(torch.Tensor(feat_num))
        torch.nn.init.normal_(muW, mu_w_mean, mu_w_std)
        logvarW = nn.Parameter(torch.Tensor(feat_num))
        nn.init.constant_(logvarW, var_w_const)

        logvarZ = nn.Parameter(torch.Tensor(feat_num), requires_grad=False)
        nn.init.constant_(logvarZ, var_z_const)

        return muW, logvarW, logvarZ
    
    def _sample(self, batch_size, feat_num, muW, varW, varZ, device):
        W = muW + varW * torch.randn((batch_size, feat_num), dtype=torch.float).to(device)
        Z = varZ * torch.randn((batch_size, feat_num), dtype=torch.float).to(device)
        return W, Z

    def _cholesky(self, rho, mu_, var_, zvar_):
        temp = []
        var_ = torch.sqrt(torch.exp(var_))
        zvar_ = torch.sqrt(torch.exp(zvar_))
        batch_size = rho.shape[0]
        mu_ = mu_.unsqueeze(0).repeat(batch_size, 1)
        var_ = var_.unsqueeze(0).repeat(batch_size, 1)
        zvar_ = zvar_.unsqueeze(0).repeat(batch_size, 1)
        
        for idx in range(rho.shape[-1]):
            W, Z = self._sample(batch_size, self.feat_num, mu_, var_, zvar_, self.device)
            rho_i = rho[:, idx:idx+1]
            a1 = rho_i * mu_
            a2 = (1-torch.pow(rho_i,2)).repeat(1, self.feat_num)
            a3 = rho_i * zvar_/var_ * (W-mu_)
            a4 = Z * a2

            a = a1 + a2*(a3+a4)
            temp.append(a)
        return torch.stack(temp) # M x B x F

def loss(pred, target):
    def mdnloss(pi, mu, var, target):
        quad = torch.pow(target.expand_as(mu) - mu, 2) * torch.reciprocal(var+VAR_EPS) * -0.5
        logdet = torch.log(var+VAR_EPS) * -0.5
        # logconstant = torch.log(2*np.pi) * -0.5
        logpi = torch.log(pi)
        exponents = quad + logdet + logpi
        
        logprobs = torch.logsumexp(exponents, 1)
        gmm_prob = torch.exp(logprobs)
        gmm_nll = -torch.mean(logprobs)
        
        return gmm_nll

    def KLdiv(rho, pi):
        rho_pos = rho+1.
        kl_reg = KL_REG_COEF * (-rho_pos * torch.log(pi+1e-2) - torch.log(rho_pos+1e-2))
        return torch.mean(kl_reg)

    def MSE(mu, target):
        fit_mse = 1e-2 * torch.pow(mu[:,0:1]-target, 2)
        return torch.mean(fit_mse)
        
    rho, pi, mu, var = pred
    
    l2 = MSE(mu, target)
    mdn = mdnloss(pi, mu, var, target)
    kldiv = KLdiv(rho, pi)
    
    return l2 + mdn + kldiv