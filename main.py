import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import datagen
from choicenet import network

class RegTest(nn.Module):
    def __init__(self, in_size, feat=32, depth=16, mixtures=5):
        super(RegTest, self).__init__()
        def init_weights(m):
          if type(m) == nn.Linear:
               torch.nn.init.normal_(m.weight, 0, 0.01)
               m.bias.data.fill_(1e-4)
          elif type(m) == nn.BatchNorm1d:
               torch.nn.init.constant_(m.weight, 1e-4)
               m.bias.data.fill_(1e-4)
              
        li = [nn.Linear(in_size, feat), nn.BatchNorm1d(feat), nn.ReLU()]
        
        for _ in range(depth):
            li.append(nn.Linear(feat, feat))
            li.append(nn.BatchNorm1d(feat))
            li.append(nn.ReLU())

        #for m in li:
        #   init_weights(m)
        
        self.feat = nn.Sequential(*li)
    
        self.mcdn = network.MDCN(feat, mixtures)
    
    def forward(self, x, rho=0.96):
        feat = self.feat(x)
        return self.mcdn(feat, rho)
class RegTest2(nn.Module):
    def __init__(self, in_size, feat=16, depth=16, mixtures=5):
        super(RegTest2, self).__init__()

        li = [nn.Linear(in_size, 1)]

        #for m in li:
        #   init_weights(m)
        
        self.feat = nn.Sequential(*li)
    
    def forward(self, x, rho=0.96):
        return self.feat(x)

DG_TYPE = "linear"
oRANGE=[-1.5, 2.5]
oRATE=0.01

x, y, t = datagen.data4reg(DG_TYPE, _n=1000, _oRange=oRANGE, _oRate=oRATE, measVar=1e-8)
xtest = np.linspace(-3, 3, 500).reshape((-1, 1))

model = RegTest2(1).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-4)

for epoch in range(100):
    
    model.train()
    x_train, y_train = torch.Tensor(x).float().cuda(), torch.Tensor(y).float().cuda()
    
    optimizer.zero_grad()
    
    output = model(x_train)
    #loss = network.loss(output, y_train)
    loss = nn.MSELoss()(output, y_train)
    loss.backward()
    print(loss)
    optimizer.step()
    
    xtest = torch.Tensor(np.linspace(start=-3,stop=3,num=100).reshape((-1,1))).cuda()
    ytest = datagen.sampler(model, _x=xtest, num_mixture=10, n_samples=1, _deterministic=True, _y=y_train)
    """
    x_plot, y_plot = x[:,0], y[:,0]
    plt.figure(figsize=(8,4))
    plt.axis([np.min(x_plot), np.max(x_plot), np.min(y_plot)-0.1, np.max(y_plot)+0.1])
    if t != '':
        plt.plot(x_plot,t[:,0],'r.')
    plt.plot(x_plot, y_plot, 'k.')
     
    for i in range(1):
        plt.plot(xtest[:,i].cpu().detach().numpy(),ytest[:,i],'b.')
    plt.title("[%d%d] name:[%s] lossVal:[%.3e]"%(epoch,100,'gg',loss.item()))
    plt.show()
    """
xtest = torch.Tensor(np.linspace(start=-3,stop=3,num=100).reshape((-1,1))).cuda()
ytest = datagen.sampler(model, _x=xtest, num_mixture=10, n_samples=1, _deterministic=True, _y=y_train)

x_plot, y_plot = x[:,0], y[:,0]
plt.figure(figsize=(8,4))
plt.axis([np.min(x_plot), np.max(x_plot), np.min(y_plot)-0.1, np.max(y_plot)+0.1])
if t != '':
    plt.plot(x_plot,t[:,0],'r.')
plt.plot(x_plot, y_plot, 'k.')
 
for i in range(1):
    plt.plot(xtest[:,i].cpu().detach().numpy(),ytest[:,i],'b.')
plt.title("ddd")
plt.show()
