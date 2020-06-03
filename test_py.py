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

# from Choice's
def f_cosexp(x): return np.cos(np.pi/2*x)*np.exp(-(x/2)**2)
def f_linear(x): return 0.5*x
def f_step(x):
    n = x.shape[0]
    t = np.zeros(shape=(n,1))
    for i in range(n):
        if x[i] < -1: t[i] = -1.0
        elif x[i] < +1: t[i] = +1.0
        else: t[i] = -1.0
    return t
def data4reg(_type='',_n=1000,_oRange=[-1.5,+1.5],_oRate=0.1,measVar=0.01):
    np.random.seed(seed=0) # Fix random seed
    _xmin,_xmax = -3,+3
    x = np.float32(np.random.uniform(_xmin,_xmax,((int)(_n),1)))
    x.sort(axis=0)
    if _type == 'cosexp': t = f_cosexp(x)
    elif _type == 'linear': t = f_linear(x)
    elif _type == 'step': t = f_step(x)
    else: print ("Unknown function type [%s]."%(_type))
    # Add measurement nosie
    y = t + np.sqrt(measVar)*np.random.randn(_n,1)
    # Switch to outliers 
    nOutlier = (int)(_n*_oRate) # Number of outliers
    y[np.random.permutation((int)(_n))[:nOutlier],:] \
        = _oRange[0]+np.random.rand(nOutlier,1)*(_oRange[1]-_oRange[0])
    return x,y,t
def plot_1dRegData(_x,_y,_t,_type='',_figSize=(6,3)):
    plt.figure(figsize=_figSize) # Plot
    # ht,=plt.plot(_x,_t,'ro')
    hd,=plt.plot(_x,_y,'k.')
    # plt.legend([ht,hd],['Target function','Training data'],fontsize=15)
    plt.title('%s'%(_type),fontsize=18)
    plt.show()

from choicenet import network
num_mixture = 10

class TestM(nn.Module):
    def __init__(self):
        super(TestM, self).__init__()
        self.b = nn.Linear(1, 64)
        self.mcdn = network.MDCN(64, num_mixture, device="cuda:0")
    
    def forward(self, x):
        f = self.b(x)
        f = f.view(f.size(0), -1)
        return self.mcdn(f)

def gaussian_distribution(y, mu, sigma):
    result = -0.5 * torch.pow((y.expand_as(mu) - mu), 2) * torch.reciprocal(sigma+1e-2)
    return torch.exp(result) * torch.reciprocal(torch.sqrt(sigma)) / np.sqrt(2.0*math.pi)


def loss_(pred, target):
    # pred : pi, mu, var / target : int
    pi = pred[0]
    mu = pred[1]
    var = pred[2]
    
    l_ = gaussian_distribution(target, mu, var) * pi
    l_ = torch.sum(l_, dim=1)
    l_ = torch.mean(-torch.log(l_))
    
    
    a = nn.MSELoss()(mu[:,0], target)
    b = l_
    
    return a + l_

def sampler(model, _x, num_mixture, n_samples=1, _deterministic=False):
  model.train(False)
  pi, mu, var = model(_x)
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

x,y,t = data4reg(_type="linear",_n=1000,_oRange=[-1.5,+1.5],_oRate=0.1,measVar=1e-2)
model = TestM()
_x = x
_y = y
_yref = t
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
batchSize = 32
maxEpoch = 10
lr = 1e-3

model.train()
_x_train, _y_train = torch.from_numpy(_x),torch.from_numpy(_y)
maxIter = max(_x_train.shape[0]//batchSize,1)
bestLossVal = np.inf
for epoch in range((int)(maxEpoch)+1):
  train_rate = (float)(epoch/maxEpoch)
  _x_train,_y_train = shuffle(_x_train,_y_train)
  x_train, y_train = Variable(_x_train).float().cuda(),Variable(_y_train).float().cuda()
  for iter in range(maxIter):
    start, end = iter*batchSize,(iter+1)*batchSize
    lr_use = lr
    optimizer.zero_grad()
    model = model.cuda()
    #print('rho : ',rho,'\nmu : ', mu,'\nvar :' ,var,'\npi : ', pi)
    output = model(x_train)
    loss = loss_(output, y_train)
    loss.backward()
    print(loss)
    #print(acc)
    optimizer.step()
    nSample = 1
    ytest = sampler(model, _x=x_train, num_mixture=10, n_samples=1, _deterministic=True)
    #print(ytest)
    ytest = ytest.detach().numpy()
    x_plot, y_plot = _x[:,0], _y[:,0]
    plt.figure(figsize=(8,4))
    plt.axis([np.min(x_plot), np.max(x_plot), np.min(y_plot)-0.1, np.max(y_plot)+0.1])
    if _yref != '':
      plt.plot(x_plot,_yref[:,0],'r')
    plt.plot(x_plot, y_plot, 'k.')
   
    for i in range(nSample):
      plt.plot(_x,ytest[:,i],'b.')
    plt.title("[%d%d] name:[%s] lossVal:[%.3e]"%(epoch,maxEpoch,'gg',loss.item()))
    plt.show()
    # if batch_idx%args['log_interval'] == 0:
    #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #       epoch, batch_idx * len(data), len(train_loader.dataset),
    #       100. *batch_idx/len(train_loader), loss.data
    #   ))