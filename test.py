import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
import os
import itertools
def create_gradient_clipping(loss,optm,vars,clipVal=1.0):
    grads, vars = zip(*optm.compute_gradients(loss, var_list=vars))
    grads = [None if grad is None else tf.clip_by_value(grad,-clipVal,clipVal) for grad in grads]
    op = optm.apply_gradients(zip(grads, vars))
    train_op = tf.tuple([loss], control_inputs=[op])
    return train_op[0]

def print_n_txt(_f,_chars,_addNewLine=True,_DO_PRINT=True):
    if _addNewLine: _f.write(_chars+'\n')
    else: _f.write(_chars)
    _f.flush();os.fsync(_f.fileno()) # Write to txt
    if _DO_PRINT:
        print (_chars)
class grid_maker(object): # For multi-GPU testing
    def __init__(self,*_arg):
        self.arg = _arg
        self.nArg = len(self.arg) # Number of total lists
        _product = itertools.product(*self.arg); _nIter = 0
        for x in _product: _nIter += 1
        self.nIter = _nIter
        self.paramList = ['']*self.nIter
        self.idxList = ['']*self.nIter
        _product = itertools.product(*self.arg);
        for idx,x in enumerate(_product):
            self.paramList[idx] = x
def get_properIdx(_processID,_maxProcessID,_nTask): # For multi-GPU testing
    ret = []
    if _processID > _nTask: return ret
    if _processID > _maxProcessID: return ret
    m = (_nTask-_processID-1) // _maxProcessID
    for i in range(m+1):
        ret.append(i*_maxProcessID+_processID)
    return ret
def f_cosexp(x):
  return np.cos(np.pi/2*x)*np.exp(-(x/2)**2)

def f_linear(x):
  return 0.5*x

def f_step(x):

  n = x.shape[0]
  t = np.zeros(shape=(n,1))
  for i in range(n):
    if x[i] < -1:
      t[i] = -1.0
    elif x[i] < 1:
      t[i] = 1.0
    else: t[i] = -1.0
  return t
def gpusession(): 
    config = tf.ConfigProto(); 
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)
    return sess
def data4reg(_type='', _n=1000, _oRange=[-1.5,1.5], _oRate=0.1, measVar=0.01):
  np.random.seed(seed=0)
  _xmin,_xmax = -3,3

  x = np.double(np.random.uniform(_xmin,_xmax,((int)(_n),1)))
  x.sort(axis=0)

  if _type == 'cosexp': t = f_cosexp(x)
  elif _type == 'linear': t = f_linear(x)
  elif _type == 'step': t = f_step(x)
  else: print ("Unknown function type [%s]."%(_type))
    # Add measurement nosie
  y = t + np.sqrt(measVar)*np.random.randn(_n,1)
    # Switch to outliers 
  nOutlier = (int)(_n*_oRate) # Number of outliers
  y[np.random.permutation((int)(_n))[:nOutlier],:] = _oRange[0]+np.random.rand(nOutlier,1)*(_oRange[1]-_oRange[0])
  return x,y,t

def plot_1dRegData(_x,_y,_t,_type='',_figSize=(6,3)):
    plt.figure(figsize=_figSize) # Plot
    # ht,=plt.plot(_x,_t,'ro')
    hd,=plt.plot(_x,_y,'k.')
    # plt.legend([ht,hd],['Target function','Training data'],fontsize=15)
    plt.title('%s'%(_type),fontsize=18)
    plt.show()
class choiceNet_reg_class(object):
    def __init__(self,_name='ChoiceNet',_scope='choicenet',_xdim=1,_ydim=1,_hdims=[64,64]
                 ,_kmix=5,_actv=tf.nn.relu,_bn=slim.batch_norm
                 ,_rho_ref_train=0.95,_tau_inv=1e-2,_var_eps=1e-2
                 ,_pi1_bias=0.0,_logSigmaZval=0
                 ,_kl_reg_coef=1e-5,_l2_reg_coef=1e-5
                 ,_SCHEDULE_MDN_REG=False
                 ,_GPU_ID=0,_VERBOSE=True):
        self.name = _name
        self.scope = _scope
        self.xdim = _xdim
        self.ydim = _ydim
        self.hdims = _hdims
        self.kmix = _kmix
        self.actv = _actv 
        self.bn   = _bn # slim.batch_norm / None
        self.rho_ref_train = _rho_ref_train # Rho for training 
        self.tau_inv = _tau_inv
        self.var_eps = _var_eps # This will be used for the loss function (var+var_eps)
        self.pi1_bias = _pi1_bias
        self.logSigmaZval = _logSigmaZval
        self.kl_reg_coef = _kl_reg_coef
        self.l2_reg_coef = _l2_reg_coef # L2 regularizer 
        self.SCHEDULE_MDN_REG = _SCHEDULE_MDN_REG
        self.GPU_ID = _GPU_ID
        self.VERBOSE = _VERBOSE
        with tf.device('/device:CPU:0'):
            # Build model
            self.build_model()
            # Build graph
            self.build_graph()
            # Check parameters
            self.check_params()
    # Build model
    def build_model(self):
        # Placeholders 
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,self.xdim],name='x') # Input [None x xdim]
        self.t = tf.placeholder(dtype=tf.float32,shape=[None,self.ydim],name='t') # Output [None x ydim]
        self.kp = tf.placeholder(dtype=tf.float32,shape=[],name='kp') # Keep probability 
        self.lr = tf.placeholder(dtype=tf.float32,shape=[],name='lr') # Learning rate
        self.is_training = tf.placeholder(dtype=tf.bool,shape=[]) # Training flag
        self.rho_ref = tf.placeholder(dtype=tf.float32,shape=[],name='rho_ref') # Training flag
        self.train_rate = tf.placeholder(dtype=tf.float32,shape=[],name='train_rate') # from 0.0~1.0
        # Initializers
        trni = tf.random_normal_initializer
        tci = tf.constant_initializer
        self.fully_init = trni(stddev=0.01)
        self.bias_init = tci(0.)
        self.bn_init = {'beta':tci(0.),'gamma':trni(1.,0.01)}
        self.bn_params = {'is_training':self.is_training,'decay':0.9,'epsilon':1e-5,
                           'param_initializers':self.bn_init,'updates_collections':None}
        # Build graph
        with tf.variable_scope(self.scope,self.name,reuse=False) as scope:
            with slim.arg_scope([slim.fully_connected],activation_fn=self.actv,
                                weights_initializer=self.fully_init,biases_initializer=self.bias_init,
                                normalizer_fn=self.bn,normalizer_params=self.bn_params,
                                weights_regularizer=None):
                _net = self.x # Now we have an input
                self.N = tf.shape(self.x)[0] # Input dimension
                for h_idx in range(len(self.hdims)): # Loop over hidden layers
                    _hdim = self.hdims[h_idx]
                    _net = slim.fully_connected(_net,_hdim,scope='lin'+str(h_idx))
                    _net = slim.dropout(_net,keep_prob=self.kp,is_training=self.is_training
                                        ,scope='dr'+str(h_idx))
                self.feat = _net # Feature [N x Q]
                self.Q = self.feat.get_shape().as_list()[1] # Feature dimension
                # Feature to K rhos (NO ACTIVATION !!!)
                _rho_raw = slim.fully_connected(self.feat,self.kmix,activation_fn=None
                                                ,scope='rho_raw')
                
                # self.rho_temp = tf.nn.tanh(_rho_raw) # [N x K] between -1.0~1.0 for regression
                self.rho_temp = tf.nn.sigmoid(_rho_raw) # [N x K] between 0.0~1.0 for classification
                
                # Maker sure the first mixture to have 'self.rho_ref' correlation
                self.rho = tf.concat([self.rho_temp[:,0:1]*0.0+self.rho_ref,self.rho_temp[:,1:]]
                                     ,axis=1) # [N x K] 
                # Variabels for the sampler 
                self.muW = tf.get_variable(name='muW',shape=[self.Q,self.ydim],
                                           initializer=tf.random_normal_initializer(stddev=0.1),
                                           dtype=tf.float32) # [Q x D]
                self.logSigmaW = tf.get_variable(name='logSigmaW'
                                        ,shape=[self.Q,self.ydim]
                                        ,initializer=tf.constant_initializer(-2.0)
                                        ,dtype=tf.float32) # [Q x D]
                self.muZ = tf.constant(np.zeros((self.Q,self.ydim))
                                        ,name='muZ',dtype=tf.float32) # [Q x D]
                self.logSigmaZ = tf.constant(self.logSigmaZval*np.ones((self.Q,self.ydim)) 
                                        ,name='logSigmaZ',dtype=tf.float32) # [Q x D]
                # Reparametrization track (THIS PART IS COMPLICATED, I KNOW)
                _muW_tile = tf.tile(self.muW[tf.newaxis,:,:]
                                    ,multiples=[self.N,1,1]) # [N x Q x D]
                _sigmaW_tile = tf.exp(tf.tile(self.logSigmaW[tf.newaxis,:,:]
                                              ,multiples=[self.N,1,1])) # [N x Q x D]
                _muZ_tile = tf.tile(self.muZ[tf.newaxis,:,:]
                                    ,multiples=[self.N,1,1]) # [N x Q x D]
                _sigmaZ_tile = tf.exp(tf.tile(self.logSigmaZ[tf.newaxis,:,:]
                                              ,multiples=[self.N,1,1])) # [N x Q x D]
                _samplerList = []
                for jIdx in range(self.kmix): # For all K mixtures
                    _rho_j = self.rho[:,jIdx:jIdx+1] # [N x 1] 
                    _rho_tile = tf.tile(_rho_j[:,:,tf.newaxis]
                                        ,multiples=[1,self.Q,self.ydim]) # [N x Q x D]
                    _epsW = tf.random_normal(shape=[self.N,self.Q,self.ydim],mean=0,stddev=1
                                             ,dtype=tf.float32) # [N x Q x D]
                    _W = _muW_tile + tf.sqrt(_sigmaW_tile)*_epsW # [N x Q x D]
                    self._W = _W
                    _epsZ = tf.random_normal(shape=[self.N,self.Q,self.ydim]
                                             ,mean=0,stddev=1,dtype=tf.float32) # [N x Q x D]
                    _Z = _muZ_tile + tf.sqrt(_sigmaZ_tile)*_epsZ # [N x Q x D]
                    self._Z = _Z
                    self.a1 = _rho_tile*_muW_tile
                    self.a3 = _rho_tile*tf.sqrt(_sigmaZ_tile)/tf.sqrt(_sigmaW_tile) *(_W-_muW_tile)
                    self.a4 = (_W-_muW_tile)+tf.sqrt(1-_rho_tile**2)*_Z
                    _Y = _rho_tile*_muW_tile + (1.0-_rho_tile**2) *(_rho_tile*tf.sqrt(_sigmaZ_tile)/tf.sqrt(_sigmaW_tile) *(_W-_muW_tile)+tf.sqrt(1-_rho_tile**2)*_Z)
                    _samplerList.append(_Y) # Append 
                WlistConcat = tf.convert_to_tensor(_samplerList) # K*[N x Q x D] => [K x N x Q x D]
                self.wSample = tf.transpose(WlistConcat,perm=[1,3,0,2]) # [N x D x K x Q]
                # K mean mixtures [N x D x K]
                _wTemp = tf.reshape(self.wSample
                                ,shape=[self.N,self.kmix*self.ydim,self.Q]) # [N x KD x Q]
                self._wTemp = _wTemp
                _featRsh = tf.reshape(self.feat,shape=[self.N,self.Q,1]) # [N x Q x 1]
                self._featRsh = _featRsh
                _mu = tf.matmul(_wTemp,_featRsh) # [N x KD x Q] x [N x Q x 1] => [N x KD x 1]
                self.mu = tf.reshape(_mu,shape=[self.N,self.ydim,self.kmix]) # [N x D x K]
                # K variance mixtures [N x D x K]
                _logvar_raw = slim.fully_connected(self.feat,self.ydim,scope='var_raw') # [N x D]
                _var_raw = tf.exp(_logvar_raw) # [N x D]
                _var_tile = tf.tile(_var_raw[:,:,tf.newaxis]
                                    ,multiples=[1,1,self.kmix]) # [N x D x K]
                _rho_tile = tf.tile(self.rho[:,tf.newaxis,:]
                                    ,multiples=[1,self.ydim,1]) # [N x D x K]
                _tau_inv = self.tau_inv
                self.var = (1.0-_rho_tile**2)*_var_tile + _tau_inv # [N x D x K]
                # Weight allocation probability pi [N x K]
                _pi_logits = slim.fully_connected(self.feat,self.kmix
                                                  ,scope='pi_logits') # [N x K]
                self.pi_temp = tf.nn.softmax(_pi_logits,dim=1) # [N x K]
                # Some heuristics to ensure that pi_1(x) is high enough
                if self.pi1_bias != 0:
                    self.pi_temp = tf.concat([self.pi_temp[:,0:1]+self.pi1_bias
                                              ,self.pi_temp[:,1:]],axis=1) # [N x K]
                    self.pi = tf.nn.softmax(self.pi_temp,dim=1) # [N x K]
                else: self.pi = self.pi_temp # [N x K]
    # Build graph
    def build_graph(self):
        # Parse
        _M = tf.shape(self.x)[0] # Current batch size
        t,pi,mu,var = self.t,self.pi,self.mu,self.var

        
        # Mixture density network loss 
        trepeat = tf.tile(t[:,:,tf.newaxis],[1,1,self.kmix]) # (N x D x K)
        self.quadratics = -0.5*tf.reduce_sum(((trepeat-mu)**2)/(var+self.var_eps),axis=1) # (N x K)
        self.logdet = -0.5*tf.reduce_sum(tf.log(var+self.var_eps),axis=1) # (N x K)
        self.logconstant = - 0.5*self.ydim*tf.log(2*np.pi) # (1)
        self.logpi = tf.log(pi) # (N x K)
        self.exponents = self.quadratics + self.logdet + self.logpi # + self.logconstant 
        self.logprobs = tf.reduce_logsumexp(self.exponents,axis=1) # (N)
        self.gmm_prob = tf.exp(self.logprobs) # (N)
        self.gmm_nll  = -tf.reduce_mean(self.logprobs) # (1)
        
        # Regression loss 
        maxIdx = tf.argmax(input=pi,axis=1, output_type=tf.int32) # Argmax Index [N]
        maxIdx = 0*tf.ones_like(maxIdx)
        coords = tf.stack([tf.transpose(gv) for gv in tf.meshgrid(tf.range(self.N),tf.range(self.ydim))] + 
                          [tf.reshape(tf.tile(maxIdx[:,tf.newaxis],[1,self.ydim]),shape=(self.N,self.ydim))]
                          ,axis=2) # [N x D x 3]
        self.mu_bar = tf.gather_nd(mu,coords) # [N x D]
        fit_mse_coef = 1e-2
        self.fit_mse = fit_mse_coef*tf.maximum((1.0-2.0*self.train_rate),0.0) \
            *tf.reduce_sum(tf.pow(self.mu_bar-self.t,2))/(tf.cast(self.N,tf.float32)) # (1)
        
        # KL-divergence
        _eps = 1e-2
        self.rho_pos = self.rho+1.0 # Make it positive
        self._kl_reg = self.kl_reg_coef*tf.reduce_sum(-self.rho_pos
                        *(tf.log(self.pi+_eps)-tf.log(self.rho_pos+_eps)),axis=1) # (N)
        self.kl_reg = tf.reduce_mean(self._kl_reg) # (1)

        # Weight decay
        _g_vars = tf.trainable_variables()
        self.c_vars = [var for var in _g_vars if '%s/'%(self.scope) in var.name]
        self.l2_reg = self.l2_reg_coef*tf.reduce_sum(tf.stack([tf.nn.l2_loss(v) for v in self.c_vars])) # [1]

        # Schedule MDN loss and regression loss 
        if self.SCHEDULE_MDN_REG:
            self.gmm_nll = tf.minimum((2.0*self.train_rate+0.1),1.0)*self.gmm_nll
            self.fit_mse = tf.maximum((1.0-2.0*self.train_rate),0.0)*self.fit_mse
            self.loss_total = self.gmm_nll+self.kl_reg+self.l2_reg+self.fit_mse # [1]
        else:
            self.gmm_nll = self.gmm_nll
            self.fit_mse = tf.constant(0.0)
            self.loss_total = self.gmm_nll+self.kl_reg+self.l2_reg
        
        # Optimizer
        USE_ADAM = False
        GRAD_CLIP = False
        if GRAD_CLIP: # Gradient clipping
            if USE_ADAM:
                _optm = tf.train.AdamOptimizer(learning_rate=self.lr
                                               ,beta1=0.9,beta2=0.999,epsilon=1e-1) # 1e-4
            else:
                _optm = tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=0.0)
            self.optm = create_gradient_clipping(self.loss_total
                                           ,_optm,tf.trainable_variables(),clipVal=1.0)
        else:
            if USE_ADAM:
                self.optm = tf.train.AdamOptimizer(learning_rate=self.lr
                            ,beta1=0.9,beta2=0.999,epsilon=1e-1).minimize(self.loss_total) 
            else:
                self.optm = tf.train.MomentumOptimizer(learning_rate=self.lr
                                                       ,momentum=0.0).minimize(self.loss_total)
                
    # Check parameters
    def check_params(self):
        _g_vars = tf.global_variables()
        self.g_vars = [var for var in _g_vars if '%s/'%(self.scope) in var.name]
        if self.VERBOSE:
            print ("==== Global Variables ====")
        for i in range(len(self.g_vars)):
            w_name  = self.g_vars[i].name
            w_shape = self.g_vars[i].get_shape().as_list()
            if self.VERBOSE:
                print (" [%02d] Name:[%s] Shape:[%s]" % (i,w_name,w_shape))
    
    # Sampler
    def sampler(self,_sess,_x,n_samples=1,_deterministic=True):
        pi, mu, var, _W, _featRsh, a1, a3, a4, feat = _sess.run([self.pi, self.mu, self.var, self._wTemp, self._featRsh, self.a1, self.a3, self.a4, self.feat],
                                feed_dict={self.x:_x,self.kp:1.0,self.is_training:False
                                          ,self.rho_ref:1.0}) #
        n_points = _x.shape[0]
        _y_sampled = np.zeros([n_points,self.ydim,n_samples])
        for i in range(n_points):
            for j in range(n_samples):
                if _deterministic: k = 0
                else: k = np.random.choice(self.kmix,p=pi[i,:])
                _y_sampled[i,:,j] = mu[i,:,k] # + np.random.randn(1,self.ydim)*np.sqrt(var[i,:,k])
        return _y_sampled 
    
    # Save 
    def save(self,_sess,_savename=None):
        """ Save name """
        if _savename==None:
            _savename='../net/net_%s.npz'%(self.scope)
        """ Get global variables """
        self.g_wnames,self.g_wvals,self.g_wshapes = [],[],[]
        for i in range(len(self.g_vars)):
            curr_wname = self.g_vars[i].name
            curr_wvar  = [v for v in tf.global_variables() if v.name==curr_wname][0]
            curr_wval  = _sess.run(curr_wvar)
            curr_wval_sqz  = curr_wval.squeeze()
            self.g_wnames.append(curr_wname)
            self.g_wvals.append(curr_wval_sqz)
            self.g_wshapes.append(curr_wval.shape)
        """ Save """
        np.savez(_savename,g_wnames=self.g_wnames,g_wvals=self.g_wvals,g_wshapes=self.g_wshapes)
        if self.VERBOSE:
            print ("[%s] Saved. Size is [%.4f]MB" % 
                   (_savename,os.path.getsize(_savename)/1000./1000.))
    
    # Save 
    def save_final(self,_sess,_savename=None):
        """ Save name """
        if _savename==None:
            _savename='../net/net_%s_final.npz'%(self.scope)
        """ Get global variables """
        self.g_wnames,self.g_wvals,self.g_wshapes = [],[],[]
        for i in range(len(self.g_vars)):
            curr_wname = self.g_vars[i].name
            curr_wvar  = [v for v in tf.global_variables() if v.name==curr_wname][0]
            curr_wval  = _sess.run(curr_wvar)
            curr_wval_sqz  = curr_wval.squeeze()
            self.g_wnames.append(curr_wname)
            self.g_wvals.append(curr_wval_sqz)
            self.g_wshapes.append(curr_wval.shape)
        """ Save """
        np.savez(_savename,g_wnames=self.g_wnames,g_wvals=self.g_wvals,g_wshapes=self.g_wshapes)
        print ("[%s] Saved. Size is [%.4f]MB" % 
               (_savename,os.path.getsize(_savename)/1000./1000.))
        
    # Restore
    def restore(self,_sess,_loadname=None):
        if _loadname==None:
            _loadname='../net/net_%s_final.npz'%(self.scope)
        l = np.load(_loadname)
        g_wnames = l['g_wnames']
        g_wvals  = l['g_wvals']
        g_wshapes = l['g_wshapes']
        for widx,wname in enumerate(g_wnames):
            curr_wvar  = [v for v in tf.global_variables() if v.name==wname][0]
            _sess.run(tf.assign(curr_wvar,g_wvals[widx].reshape(g_wshapes[widx])))
        if self.VERBOSE:
            print ("Weight restored from [%s] Size is [%.4f]MB" % 
                   (_loadname,os.path.getsize(_loadname)/1000./1000.))
    
    # Save to mat file
    def save2mat(self,_xdata='',_ydata='',_yref=''):
        # Save weights to mat file so that MATLAB can use it.
        npzPath = '../net/net_%s.npz'%(self.scope)
        l = np.load(npzPath)
        g_wnames = l['g_wnames']
        g_wvals  = l['g_wvals']
        g_wshapes = l['g_wshapes']
        D = {}
        for widx,wname in enumerate(g_wnames):
            cName = wname.replace(':0','')
            cName = cName.replace(self.scope+'/','')
            cName = cName.replace('/','_')
            cVal = g_wvals[widx].reshape(g_wshapes[widx])
            D[cName] = cVal
            # if self.VERBOSE: print ("name is [%s] shape is %s."%(cName,cVal.shape,))
        # Save data
        if _xdata!='': D['xdata']=_xdata
        if _ydata!='': D['ydata']=_ydata
        if _yref!='': D['yref']=_yref
        # Save dictionary D to the mat file
        matPath = '../data/net_%s.mat'%(self.scope)
        if self.VERBOSE: print ("[%s] saved."%(matPath))
        sio.savemat(matPath,D)
        
    # Save to mat file
    def save2mat_final(self,_xdata='',_ydata='',_yref=''):
        # Save weights to mat file so that MATLAB can use it.
        npzPath = '../net/net_%s_final.npz'%(self.scope)
        l = np.load(npzPath)
        g_wnames = l['g_wnames']
        g_wvals  = l['g_wvals']
        g_wshapes = l['g_wshapes']
        D = {}
        for widx,wname in enumerate(g_wnames):
            cName = wname.replace(':0','')
            cName = cName.replace(self.scope+'/','')
            cName = cName.replace('/','_')
            cVal = g_wvals[widx].reshape(g_wshapes[widx])
            D[cName] = cVal
            if self.VERBOSE:
                print ("name is [%s] shape is %s."%(cName,cVal.shape,))
        # Save data
        if _xdata!='': D['xdata']=_xdata
        if _ydata!='': D['ydata']=_ydata
        if _yref!='': D['yref']=_yref
        # Save dictionary D to the mat file
        matPath = '../data/net_%s_final.mat'%(self.scope)
        sio.savemat(matPath,D)
        print ("[%s] Saved. Size is [%.4f]MB" % 
               (matPath,os.path.getsize(matPath)/1000./1000.))
    
    # Train
    def train(self,_sess,_x,_y,_yref='',_lr=1e-3,_batchSize=512,_maxEpoch=1e4,_kp=1.0
              ,_LR_SCHEDULE=True
              ,_PRINT_EVERY=20,_PLOT_EVERY=20
              ,_SAVE_TXT=True,_SAVE_BEST_NET=True,_SAVE_FINAL=True):
        
        # Reference training data 
        _x_train,_y_train = _x,_y
        
        # Iterate
        if _PRINT_EVERY == 0: print_period = 0
        else: print_period = _maxEpoch//_PRINT_EVERY
        if _PLOT_EVERY == 0: plot_period = 0
        else: plot_period = _maxEpoch//_PLOT_EVERY
            
        maxIter = max(_x_train.shape[0]//_batchSize, 1)
        bestLossVal = np.inf

        for epoch in range((int)(_maxEpoch)+1): # For every epoch
            train_rate = (float)(epoch/_maxEpoch)
            _x_train,_y_train = shuffle(_x_train,_y_train)
            for iter in range(maxIter): # For every iteration
                start,end = iter*_batchSize,(iter+1)*_batchSize
                if _LR_SCHEDULE:
                    if epoch < 0.5*_maxEpoch:
                        lr_use = _lr
                    elif epoch < 0.75*_maxEpoch:
                        lr_use = _lr/10.
                    else:
                        lr_use = _lr/100.
                else:
                    lr_use = _lr
                feeds = {self.x:_x_train[start:end,:],self.t:_y_train[start:end,:]
                         ,self.kp:_kp,self.lr:lr_use,self.train_rate:(float)(epoch/_maxEpoch)
                         ,self.rho_ref:self.rho_ref_train,self.is_training:True}
                # Optimize 
                _sess.run(self.optm,feeds)

            # Track the Best result
            BEST_FLAG = False
            check_period = _maxEpoch//100
            if (epoch%check_period)==0:
                feeds = {self.x:_x,self.t:_y,self.kp:1.0,self.train_rate:train_rate
                         ,self.rho_ref:self.rho_ref_train,self.is_training:False}
                opers = [self.loss_total,self.gmm_nll,self.kl_reg,self.l2_reg,self.fit_mse]
                lossVal,gmm_nll,kl_reg,l2_reg,fit_mse = _sess.run(opers,feeds)
                if (lossVal < bestLossVal) & (train_rate >= 0.5):
                    bestLossVal = lossVal
                    BEST_FLAG = True
                    if _SAVE_BEST_NET:
                        self.save(_sess) # Save the current best model 
                        self.save2mat(_xdata=_x,_ydata=_y,_yref=_yref)
            
            # Print current result 
            if (print_period!=0) and ((epoch%print_period)==0 or (epoch==(_maxEpoch-1))): # Print 
                # Feed total dataset 
                feeds = {self.x:_x,self.t:_y,self.kp:1.0,self.train_rate:(float)(epoch/_maxEpoch)
                         ,self.rho_ref:self.rho_ref_train,self.is_training:False}
                opers = [self.loss_total,self.gmm_nll,self.kl_reg,self.l2_reg,self.fit_mse]
                lossVal,gmm_nll,kl_reg,l2_reg,fit_mse = _sess.run(opers,feeds)


                if self.VERBOSE:
                    print ("[%d/%d] loss:%.3f(gmm:%.3f+kl:%.3f+l2:%.3f+fit:%.3f) bestLoss:%.3f"
                               %(epoch,_maxEpoch,lossVal,gmm_nll,kl_reg,l2_reg,fit_mse,bestLossVal))

            # Plot current result 
            if (plot_period!=0) and ((epoch%plot_period)==0 or (epoch==(_maxEpoch-1))): # Plot
                # Get loss values
                feeds = {self.x:_x,self.t:_y,self.kp:1.0,self.train_rate:(float)(epoch/_maxEpoch)
                         ,self.rho_ref:self.rho_ref_train,self.is_training:False}
                opers = [self.loss_total,self.gmm_nll,self.kl_reg,self.l2_reg,self.fit_mse]
                lossVal,gmm_nll,kl_reg,l2_reg,fit_mse = _sess.run(opers,feeds)
                # Sampling
                nSample = 1
                ytest = self.sampler(_sess=_sess,_x=_x,n_samples=nSample)
                # Plot first dimensions of both input and output
                x_plot,y_plot = _x[:,0],_y[:,0] # Traning data 
                plt.figure(figsize=(8,4))
                plt.axis([np.min(x_plot),np.max(x_plot),np.min(y_plot)-0.1,np.max(y_plot)+0.1])
                if _yref != '': plt.plot(x_plot,_yref[:,0],'r.') # Plot reference
                plt.plot(x_plot,y_plot,'k.') # Plot training data
                for i in range(nSample): plt.plot(_x,ytest[:,0,i],'b.')
                plt.title("[%d/%d] name:[%s] lossVal:[%.3e]"%(epoch,_maxEpoch,self.name,lossVal)); 
                plt.show()
                
        # Save final weights 
        if _SAVE_FINAL:
            self.save_final(_sess)
            self.save2mat_final(_xdata=_x,_ydata=_y,_yref=_yref)
        
    # Test
    def test(self,_sess,_xdata,_ydata,_yref,_xtest,_titleStr
             ,_PLOT_TRAIN=True,_PLOT_RES=True,_SAVE_FIG=False):
        nSample = 1
        ytest = self.sampler(_sess=_sess,_x=_xtest,n_samples=nSample)
        # Compute error
        yPred = self.sampler(_sess=_sess,_x=_xdata,n_samples=nSample)
        rmseVal = np.sqrt(np.mean((yPred.flatten()-_yref.flatten())**2))
        
        # Plot 
        if _PLOT_TRAIN:
            plt.figure(figsize=(6,4))
            plt.axis([np.min(_xdata),np.max(_xdata),np.min(_ydata),np.max(_ydata)])
            plt.plot(_xdata,_ydata,'k.')
            plt.xlabel('Input',fontsize=13);plt.ylabel('Output',fontsize=13)
            plt.title('Training Data for a Regression Task',fontsize=16); 
            if _SAVE_FIG: 
                plt.savefig('../fig/fig_%s_data.png'%(self.scope)); plt.show()
            else: 
                plt.show()
        # Plot 
        if _PLOT_RES:
            fig = plt.figure(figsize=(6,4))
            plt.axis([np.min(_xdata),np.max(_xdata),np.min(_ydata),np.max(_ydata)])
            ht,=plt.plot(_xdata,_yref,'r-',linewidth=2) # GT
            hd,=plt.plot(_xdata,_ydata,'k.')
            for i in range(nSample): 
                hf,=plt.plot(_xtest,ytest[:,0,i],'b-',linewidth=3) # Prediction
            plt.xlabel('Input',fontsize=13);plt.ylabel('Output',fontsize=13)
            plt.title('%s rmse:%.4f'%(_titleStr,rmseVal),fontsize=16)
            plt.legend([ht,hd,hf],['Target function','Training data','Fitting result']
                       ,fontsize=15,loc='upper left')
            if _SAVE_FIG: 
                plt.savefig('../fig/fig_%s_res.png'%(self.scope)); plt.show()
            else: 
                plt.show()
if __name__ == "__main__":
    # Training data 
    dataType = 'cosexp' # ['cosexp','linear','step']
    oRate = 0.6
    measVar = 1e-8
    x,y,t=data4reg(_type=dataType,_n=1000,_oRange=[-1.5,+2.5],_oRate=oRate,measVar=measVar)
    xtest = np.linspace(start=-3,stop=3,num=500).reshape((-1,1))
    # plot_1dRegData(_x=x,_y=y,_t=t,_type='Training data [%s] function'%(dataType),_figSize=(8,4))
    
    # Make graph 
    tf.reset_default_graph(); sess = gpusession()
    tf.set_random_seed(0); np.random.seed(0)
    CN = choiceNet_reg_class(_name='CN_%s_oRate%02d_var%.1e'%(dataType,oRate*100,measVar)
                            ,_xdim=1,_ydim=1,_hdims=[32,32]    
                            ,_kmix=5,_actv=tf.nn.relu,_bn=slim.batch_norm
                            ,_rho_ref_train=0.99,_tau_inv=1e-2,_var_eps=1e-4
                            ,_pi1_bias=0.0,_logSigmaZval=0
                            ,_kl_reg_coef=1e-5,_l2_reg_coef=1e-5
                            ,_SCHEDULE_MDN_REG=False
                            ,_GPU_ID=0,_VERBOSE=False)
    sess.run(tf.global_variables_initializer()) # Initialize variables
    
    # Train 
    DO_TRAIN = True
    if DO_TRAIN:
        CN.train(_sess=sess,_x=x,_y=y,_yref=t
               ,_lr=1e-3,_batchSize=256,_maxEpoch=1e4,_kp=1.0
               ,_LR_SCHEDULE=False 
               ,_PRINT_EVERY=20,_PLOT_EVERY=20
               ,_SAVE_TXT=True,_SAVE_BEST_NET=True)
        print ("Train done.")
    else:
        CN.restore(sess)
        print ("Network restored.")
    # Test 
    CN.test(_sess=sess,_xdata=x,_ydata=y,_yref=t,_xtest=xtest
           ,_titleStr=CN.name
           ,_PLOT_TRAIN=True,_PLOT_RES=True,_SAVE_FIG=True)