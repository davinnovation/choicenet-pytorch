import torch
import torch.nn as nn

class PI_block(nn.Module):
    def __init__(self, feat_num, mixtures):
        self.f = nn.Sequential([
            nn.Linear(feat_num, mixtures),
            nn.Softmax()
        ])
    
    def forward(self, x):
        return self.f(x)

class COV_block(nn.Module):
    def __init__(self, feat_num, mixtures):
        self.f = nn.Sequential([
            nn.Linear(feat_num, mixtures),
            nn.Softmax()
        ])
    
    def forward(self, x):
        return self.f(x)

class MU_block(nn.Module):
    def __init__(self, feat_num, mixtures):
        self.f = nn.Sequential([
            nn.Linear(feat_num, mixtures),
            nn.Softmax()
        ])
    
    def forward(self, x):
        return self.f(x)

class MCDN(nn.Module):
    mode_dict = {
        'regression' : nn.Tanh(),
        'classification' : nn.Sigmoid()
    }

    def __init__(self, basenet:nn.Module, mixtures:int=128, mode='regression'):
        self.basenet = basenet

        self.rho = nn.Sequential([
            nn.Linear(512, mixtures),
            MCDN.mode_dict[mode]
        ])

    def forward(self, x):
        x = self.basenet(x) # resnet50, 512 channel - after global average pool
