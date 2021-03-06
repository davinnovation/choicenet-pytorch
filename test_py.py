import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import argparse
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import math
from torchsummary import summary
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import torch.nn.init as init

from data

from choicenet import network
num_mixture = 10

class TestM(nn.Module):
    def __init__(self, hid=32, dep=16):
        def init_weights(m):
          if type(m) == nn.Linear:
              torch.nn.init.normal_(m.weight, 0, 0.01)
              m.bias.data.fill_(0.0)
          elif type(m) == nn.BatchNorm1d:
              torch.nn.init.constant_(m.weight, 0.)
              m.bias.data.fill_(0.0)
        super(TestM, self).__init__()
        li = [nn.Linear(1, hid), nn.ReLU(), nn.BatchNorm1d(hid)]
        for _ in range(dep):
          li.append(nn.Linear(hid, hid))
          li.append(nn.ReLU())
          li.append(nn.BatchNorm1d(hid))
        for m in li:
          init_weights(m)
        self.b = nn.Sequential(*li)
        self.mcdn = network.MDCN(hid, num_mixture, device="cuda:0")
    
    def forward(self, x, rho=0.95):
        f = self.b(x)
        f = f.view(f.size(0), -1)
        return self.mcdn(f, rho)
    """
    def __init__(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        super(TestM, self).__init__()
        self.b = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 10))
        self.b.apply(init_weights)
    
    def forward(self, x, rho=0.95):
        a= self.b(x)
        return a,a,a,a
    """
VAR_EPS = 1e-4
KL_REG_COEF = 1e-5
def loss_(pred, target):
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

def sampler(model, _x, num_mixture, n_samples=1, _deterministic=False, _y=None):
  model.train(False)
  rho, pi, mu, var = model(_x, 1.)
  n_points = _x.shape[0]
  _y_sampled = torch.zeros([n_points, n_samples])

  #print('mu',mu)
  #print('pi',pi)
  for i in range(n_points):
    for j in range(n_samples):
      if _deterministic: k=0
      else: k=np.random.choice(num_mixture,size=1, p=pi[i,:].view(-1).detach().numpy())
      #print(pi[i,:].view(-1).detach().numpy(), k)
      _y_sampled[i,j] = mu[i,k]
      
  return _y_sampled

x,y,t = data4reg(_type="linear",_n=1000,_oRange=[-1.5,+1.5],_oRate=0.0,measVar=1e-8)
model = TestM().cuda()
_x = x
_y = y
_yref = t
optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-1)
batchSize = 256
maxEpoch = 1000
lr = 1e-3

model.train()
_x_train, _y_train = torch.from_numpy(_x),torch.from_numpy(_y)
x_train, y_train = Variable(_x_train).float().cuda(),Variable(_y_train).float().cuda()

bestLossVal = np.inf
for epoch in range((int)(maxEpoch)+1):
  optimizer.zero_grad()
  output = model(x_train)
  loss = loss_(output, y_train)
  loss.backward()
  print(loss)
  optimizer.step()
  xtest = torch.Tensor(np.linspace(start=-3,stop=3,num=100).reshape((-1,1))).cuda()
  ytest = sampler(model, _x=xtest, num_mixture=10, n_samples=1, _deterministic=True, _y=y_train)
  #print(ytest)
  ytest = ytest.detach().numpy()
  x_plot, y_plot = _x[:,0], _y[:,0]
  plt.figure(figsize=(8,4))
  plt.axis([np.min(x_plot), np.max(x_plot), np.min(y_plot)-0.1, np.max(y_plot)+0.1])
  if _yref != '':
    plt.plot(x_plot,_yref[:,0],'r.')
  plt.plot(x_plot, y_plot, 'k.')
   
  for i in range(1):
    plt.plot(xtest[:,i].cpu().detach().numpy(),ytest[:,i],'g.')
  plt.title("[%d%d] name:[%s] lossVal:[%.3e]"%(epoch,maxEpoch,'gg',loss.item()))
  plt.show()