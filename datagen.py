import numpy as np

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

def sampler(model, _x, num_mixture, n_samples=1, _deterministic=False, _y=None):
    model.train(False)
    # rho, pi, mu, var = model(_x, 1.)
    mu = model(_x, 1.)
    n_points = _x.shape[0]
    _y_sampled = np.zeros([n_points, n_samples])

    #print('mu',mu)
    #print('pi',pi)
    for i in range(n_points):
        for j in range(n_samples):
            if _deterministic: k=0
            #else: k=np.random.choice(num_mixture,size=1, p=pi[i,:].view(-1).detach().numpy())
            #print(pi[i,:].view(-1).detach().numpy(), k)
            #_y_sampled[i,j] = np.array(mu[i,k].cpu().detach())
            _y_sampled[i,j] = np.array(mu[i].cpu().detach())

    return _y_sampled