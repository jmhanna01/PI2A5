#!/usr/bin/env python
# coding: utf-8

# In[282]:


try:
    get_ipython().run_line_magic('matplotlib', 'notebook')
except Exception:
    pass

# import and test
import tensorflow as tf2
print("TF version =", tf2.__version__)

# we want TF 2.x
assert tf2.__version__ >= "2.0"

# disable eager execution etc
tf = tf2.compat.v1
tf.disable_eager_execution()

# disable annoying warnings
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

# make sure we have GPU support
print("GPU support = ", tf.test.is_gpu_available())

# import other useful libs
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from tqdm import tqdm_notebook
from py_vollib_vectorized import vectorized_implied_volatility as implied_vol

# representation of real numbers in TF, change here for 32/64 bits
real_type = tf.float32
# real_type = tf.float64


# In[5]:


# 1. test


# In[6]:


def vanilla_net(
    input_dim,      # dimension of inputs, e.g. 10
    hidden_units,   # units in hidden layers, assumed constant, e.g. 20
    hidden_layers,  # number of hidden layers, e.g. 4
    seed):          # seed for initialization or None for random
    
    # set seed
    tf.set_random_seed(seed)
    
    # input layer
    xs = tf.placeholder(shape=[None, input_dim], dtype=real_type)
    
    # connection weights and biases of hidden layers
    ws = [None]
    bs = [None]
    # layer 0 (input) has no parameters
    
    # layer 0 = input layer
    zs = [xs] # eq.3, l=0
    
    # first hidden layer (index 1)
    # weight matrix
    ws.append(tf.get_variable("w1", [input_dim, hidden_units],         initializer = tf.variance_scaling_initializer(), dtype=real_type))
    # bias vector
    bs.append(tf.get_variable("b1", [hidden_units],         initializer = tf.zeros_initializer(), dtype=real_type))
    # graph
    zs.append(zs[0] @ ws[1] + bs[1]) # eq. 3, l=1
    
    # second hidden layer (index 2) to last (index hidden_layers)
    for l in range(1, hidden_layers): 
        ws.append(tf.get_variable("w%d"%(l+1), [hidden_units, hidden_units],             initializer = tf.variance_scaling_initializer(), dtype=real_type))
        bs.append(tf.get_variable("b%d"%(l+1), [hidden_units],             initializer = tf.zeros_initializer(), dtype=real_type))
        zs.append(tf.nn.softplus(zs[l]) @ ws[l+1] + bs[l+1]) # eq. 3, l=2..L-1

    # output layer (index hidden_layers+1)
    ws.append(tf.get_variable("w"+str(hidden_layers+1), [hidden_units, 1],             initializer = tf.variance_scaling_initializer(), dtype=real_type))
    bs.append(tf.get_variable("b"+str(hidden_layers+1), [1],         initializer = tf.zeros_initializer(), dtype=real_type))
    # eq. 3, l=L
    zs.append(tf.nn.softplus(zs[hidden_layers]) @ ws[hidden_layers+1] + bs[hidden_layers+1]) 
    
    # result = output layer
    ys = zs[hidden_layers+1]
    
    # return input layer, (parameters = weight matrices and bias vectors), 
    # [all layers] and output layer
    return xs, (ws, bs), zs, ys


# In[7]:


# compute d_output/d_inputs by (explicit) backprop in vanilla net
def backprop(
    weights_and_biases, # 2nd output from vanilla_net() 
    zs):                # 3rd output from vanilla_net()
    
    ws, bs = weights_and_biases
    L = len(zs) - 1
    
    # backpropagation, eq. 4, l=L..1
    zbar = tf.ones_like(zs[L]) # zbar_L = 1
    for l in range(L-1, 0, -1):
        zbar = (zbar @ tf.transpose(ws[l+1])) * tf.nn.sigmoid(zs[l]) # eq. 4
    # for l=0
    zbar = zbar @ tf.transpose(ws[1]) # eq. 4
    
    xbar = zbar # xbar = zbar_0
    
    # dz[L] / dx
    return xbar    

# combined graph for valuation and differentiation
def twin_net(input_dim, hidden_units, hidden_layers, seed):
    
    # first, build the feedforward net
    xs, (ws, bs), zs, ys = vanilla_net(input_dim, hidden_units, hidden_layers, seed)
    
    # then, build its differentiation by backprop
    xbar = backprop((ws, bs), zs)
    
    # return input x, output y and differentials d_y/d_z
    return xs, ys, xbar


# In[8]:


def vanilla_training_graph(input_dim, hidden_units, hidden_layers, seed):
    
    # net
    inputs, weights_and_biases, layers, predictions =         vanilla_net(input_dim, hidden_units, hidden_layers, seed)
    
    # backprop even though we are not USING differentials for training
    # we still need them to predict derivatives dy_dx 
    derivs_predictions = backprop(weights_and_biases, layers)
    
    # placeholder for labels
    labels = tf.placeholder(shape=[None, 1], dtype=real_type)
    
    # loss 
    loss = tf.losses.mean_squared_error(labels, predictions)
    
    # optimizer
    learning_rate = tf.placeholder(real_type)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    
    # return all necessary 
    return inputs, labels, predictions, derivs_predictions, learning_rate, loss, optimizer.minimize(loss)

# training loop for one epoch
def vanilla_train_one_epoch(# training graph from vanilla_training_graph()
                            inputs, labels, lr_placeholder, minimizer,   
                            # training set 
                            x_train, y_train,                           
                            # params, left to client code
                            learning_rate, batch_size, session):        
    
    m, n = x_train.shape
    
    # minimization loop over mini-batches
    first = 0
    last = min(batch_size, m)
    while first < m:
        session.run(minimizer, feed_dict = {
            inputs: x_train[first:last], 
            labels: y_train[first:last],
            lr_placeholder: learning_rate
        })
        first = last
        last = min(first + batch_size, m)


# In[9]:


def diff_training_graph(
    # same as vanilla
    input_dim, 
    hidden_units, 
    hidden_layers, 
    seed, 
    # balance relative weight of values and differentials 
    # loss = alpha * MSE(values) + beta * MSE(greeks, lambda_j) 
    # see online appendix
    alpha, 
    beta,
    lambda_j):
    
    # net, now a twin
    inputs, predictions, derivs_predictions = twin_net(input_dim, hidden_units, hidden_layers, seed)
    
    # placeholder for labels, now also derivs labels
    labels = tf.placeholder(shape=[None, 1], dtype=real_type)
    derivs_labels = tf.placeholder(shape=[None, derivs_predictions.shape[1]], dtype=real_type)
    
    # loss, now combined values + derivatives
    loss = alpha * tf.losses.mean_squared_error(labels, predictions)     + beta * tf. losses.mean_squared_error(derivs_labels * lambda_j, derivs_predictions * lambda_j)
    
    # optimizer, as vanilla
    learning_rate = tf.placeholder(real_type)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    
    # return all necessary tensors, including derivatives
    # predictions and labels
    return inputs, labels, derivs_labels, predictions, derivs_predictions,             learning_rate, loss, optimizer.minimize(loss)

def diff_train_one_epoch(inputs, labels, derivs_labels, 
                         # graph
                         lr_placeholder, minimizer,             
                         # training set, extended
                         x_train, y_train, dydx_train,          
                         # params
                         learning_rate, batch_size, session):   
    
    m, n = x_train.shape
    
    # minimization loop, now with Greeks
    first = 0
    last = min(batch_size, m)
    while first < m:
        session.run(minimizer, feed_dict = {
            inputs: x_train[first:last], 
            labels: y_train[first:last],
            derivs_labels: dydx_train[first:last],
            lr_placeholder: learning_rate
        })
        first = last
        last = min(first + batch_size, m)


# In[10]:


def train(description,
          # neural approximator
          approximator,              
          # training params
          reinit=True, 
          epochs=100, 
          # one-cycle learning rate schedule
          learning_rate_schedule=[    (0.0, 1.0e-8), \
                                      (0.2, 0.1),    \
                                      (0.6, 0.01),   \
                                      (0.9, 1.0e-6), \
                                      (1.0, 1.0e-8)  ], 
          batches_per_epoch=16,
          min_batch_size=256,
          # callback function and when to call it
          callback=None,           # arbitrary callable
          callback_epochs=[]):     # call after what epochs, e.g. [5, 20]
              
    # batching
    batch_size = max(min_batch_size, approximator.m // batches_per_epoch)
    
    # one-cycle learning rate sechedule
    lr_schedule_epochs, lr_schedule_rates = zip(*learning_rate_schedule)
            
    # reset
    if reinit:
        approximator.session.run(approximator.initializer)
    
    # callback on epoch 0, if requested
    if callback and 0 in callback_epochs:
        callback(approximator, 0)
        
    # loop on epochs, with progress bar (tqdm)
    for epoch in tqdm_notebook(range(epochs), desc=description):
        
        # interpolate learning rate in cycle
        learning_rate = np.interp(epoch / epochs, lr_schedule_epochs, lr_schedule_rates)
        
        # train one epoch
        
        if not approximator.differential:
        
            vanilla_train_one_epoch(
                approximator.inputs, 
                approximator.labels, 
                approximator.learning_rate, 
                approximator.minimizer, 
                approximator.x, 
                approximator.y, 
                learning_rate, 
                batch_size, 
                approximator.session)
        
        else:
        
            diff_train_one_epoch(
                approximator.inputs, 
                approximator.labels, 
                approximator.derivs_labels,
                approximator.learning_rate, 
                approximator.minimizer, 
                approximator.x, 
                approximator.y, 
                approximator.dy_dx,
                learning_rate, 
                batch_size, 
                approximator.session)
        
        # callback, if requested
        if callback and epoch in callback_epochs:
            callback(approximator, epoch)

    # final callback, if requested
    if callback and epochs in callback_epochs:
        callback(approximator, epochs)        


# In[11]:


# basic data preparation
epsilon = 1.0e-08
def normalize_data(x_raw, y_raw, dydx_raw=None, crop=None):
    
    # crop dataset
    m = crop if crop is not None else x_raw.shape[0]
    x_cropped = x_raw[:m]
    y_cropped = y_raw[:m]
    dycropped_dxcropped = dydx_raw[:m] if dydx_raw is not None else None
    
    # normalize dataset
    x_mean = x_cropped.mean(axis=0)
    x_std = x_cropped.std(axis=0) + epsilon
    x = (x_cropped- x_mean) / x_std
    y_mean = y_cropped.mean(axis=0)
    y_std = y_cropped.std(axis=0) + epsilon
    y = (y_cropped-y_mean) / y_std
    
    # normalize derivatives too
    if dycropped_dxcropped is not None:
        dy_dx = dycropped_dxcropped / y_std * x_std 
        # weights of derivatives in cost function = (quad) mean size
        lambda_j = 1.0 / np.sqrt((dy_dx ** 2).mean(axis=0)).reshape(1, -1)
    else:
        dy_dx = None
        lambda_j = None
    
    return x_mean, x_std, x, y_mean, y_std, y, dy_dx, lambda_j


# In[12]:


class Neural_Approximator():
    
    def __init__(self, x_raw, y_raw, 
                 dydx_raw=None):      # derivatives labels, 
       
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.dydx_raw = dydx_raw
        
        # tensorflow logic
        self.graph = None
        self.session = None
                        
    def __del__(self):
        if self.session is not None:
            self.session.close()
        
    def build_graph(self,
                differential,       # differential or not           
                lam,                # balance cost between values and derivs  
                hidden_units, 
                hidden_layers, 
                weight_seed):
        
        # first, deal with tensorflow logic
        if self.session is not None:
            self.session.close()

        self.graph = tf.Graph()
        
        with self.graph.as_default():
        
            # build the graph, either vanilla or differential
            self.differential = differential
            
            if not differential:
            # vanilla 
                
                self.inputs,                 self.labels,                 self.predictions,                 self.derivs_predictions,                 self.learning_rate,                 self.loss,                 self.minimizer                 = vanilla_training_graph(self.n, hidden_units, hidden_layers, weight_seed)
                    
            else:
            # differential
            
                if self.dy_dx is None:
                    raise Exception("No differential labels for differential training graph")
            
                self.alpha = 1.0 / (1.0 + lam * self.n)
                self.beta = 1.0 - self.alpha
                
                self.inputs,                 self.labels,                 self.derivs_labels,                 self.predictions,                 self.derivs_predictions,                 self.learning_rate,                 self.loss,                 self.minimizer = diff_training_graph(self.n, hidden_units,                                                      hidden_layers, weight_seed,                                                      self.alpha, self.beta, self.lambda_j)
        
            # global initializer
            self.initializer = tf.global_variables_initializer()
            
        # done
        self.graph.finalize()
        self.session = tf.Session(graph=self.graph)
                        
    # prepare for training with m examples, standard or differential
    def prepare(self, 
                m, 
                differential,
                lam=1,              # balance cost between values and derivs  
                # standard architecture
                hidden_units=20, 
                hidden_layers=4, 
                weight_seed=None):

        # prepare dataset
        self.x_mean, self.x_std, self.x, self.y_mean, self.y_std, self.y, self.dy_dx, self.lambda_j =             normalize_data(self.x_raw, self.y_raw, self.dydx_raw, m)
        
        # build graph        
        self.m, self.n = self.x.shape        
        self.build_graph(differential, lam, hidden_units, hidden_layers, weight_seed)
        
    def train(self,            
              description="training",
              # training params
              reinit=True, 
              epochs=100, 
              # one-cycle learning rate schedule
              learning_rate_schedule=[
                  (0.0, 1.0e-8), 
                  (0.2, 0.1), 
                  (0.6, 0.01), 
                  (0.9, 1.0e-6), 
                  (1.0, 1.0e-8)], 
              batches_per_epoch=16,
              min_batch_size=256,
              # callback and when to call it
              # we don't use callbacks, but this is very useful, e.g. for debugging
              callback=None,           # arbitrary callable
              callback_epochs=[]):     # call after what epochs, e.g. [5, 20]
              
        train(description, 
              self, 
              reinit, 
              epochs, 
              learning_rate_schedule, 
              batches_per_epoch, 
              min_batch_size,
              callback, 
              callback_epochs)
     
    def predict_values(self, x):
        # scale
        x_scaled = (x-self.x_mean) / self.x_std 
        # predict scaled
        y_scaled = self.session.run(self.predictions, feed_dict = {self.inputs: x_scaled})
        # unscale
        y = self.y_mean + self.y_std * y_scaled
        return y

    def predict_values_and_derivs(self, x):
        # scale
        x_scaled = (x-self.x_mean) / self.x_std
        # predict scaled
        y_scaled, dyscaled_dxscaled = self.session.run(
            [self.predictions, self.derivs_predictions], 
            feed_dict = {self.inputs: x_scaled})
        # unscale
        y = self.y_mean + self.y_std * y_scaled
        dydx = self.y_std / self.x_std * dyscaled_dxscaled
        return y, dydx


# In[30]:


# helper analytics    
def bsPrice(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + 0.5 * vol * vol * T) / vol / np.sqrt(T)
    d2 = d1 - vol * np.sqrt(T)
    return strike * norm.cdf(-d2) - spot * norm.cdf(-d1)

def bsDelta(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + 0.5 * vol * vol * T) / vol / np.sqrt(T)
    return norm.cdf(d1)

def bsVega(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + 0.5 * vol * vol * T) / vol / np.sqrt(T)
    return spot * np.sqrt(T) * norm.pdf(d1)
#
# helper analytics    
def bsPrice(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + 0.5 * vol * vol * T) / vol / np.sqrt(T)
    d2 = d1 - vol * np.sqrt(T)
    return strike * norm.cdf(-d2) - spot * norm.cdf(-d1)

def bsDelta(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + 0.5 * vol * vol * T) / vol / np.sqrt(T)
    return norm.cdf(d1)-1

def bsVega(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + 0.5 * vol * vol * T) / vol / np.sqrt(T)
    return spot * np.sqrt(T) * norm.pdf(d1)
#
    
# main class
class BlackScholes:
    
    def __init__(self, 
                 vol=0.2,
                 T1=1, 
                 T2=2, 
                 K=1.10,
                 volMult=1.5):
        
        self.spot = 1
        self.vol = vol
        self.T1 = T1
        self.T2 = T2
        self.K = K
        self.volMult = volMult
                        
    # training set: returns S1 (mx1), C2 (mx1) and dC2/dS1 (mx1)
    def trainingSet(self, m, anti=True, seed=None):
    
        np.random.seed(seed)
        
        # 2 sets of normal returns
        returns = np.random.normal(size=[m, 2])

        # SDE
        vol0 = self.vol * self.volMult
        R1 = np.exp(-0.5*vol0*vol0*self.T1 + vol0*np.sqrt(self.T1)*returns[:,0])
        R2 = np.exp(-0.5*self.vol*self.vol*(self.T2-self.T1)                     + self.vol*np.sqrt(self.T2-self.T1)*returns[:,1])
        S1 = self.spot * R1
        S2 = S1 * R2 

        # payoff
        pay = np.maximum(0,  self.K-S2)
        
        # two antithetic paths
        if anti:
            
            R2a = np.exp(-0.5*self.vol*self.vol*(self.T2-self.T1)                     - self.vol*np.sqrt(self.T2-self.T1)*returns[:,1])
            S2a = S1 * R2a             
            paya = np.maximum(0,  self.K-S2a)
            
            X = S1
            Y = 0.5 * (pay + paya)
    
            # differentials
            Z1 =  np.where(S2 < self.K, -R2, 0.0).reshape((-1,1)) 
            Z2 =  np.where(S2a < self.K, -R2a, 0.0).reshape((-1,1)) 
            Z = 0.5 * (Z1 + Z2)
                    
        # standard
        else:
        
            X = S1
            Y = pay
            
            # differentials
            Z =  np.where(S2 > self.K, -R2, 0.0).reshape((-1,1)) 
        
        return X.reshape([-1,1]), Y.reshape([-1,1]), Z.reshape([-1,1])
    
    # test set: returns a grid of uniform spots 
    # with corresponding ground true prices, deltas and vegas
    def testSet(self, lower=0.35, upper=1.65, num=100, seed=None):
        
        spots = np.linspace(lower, upper, num).reshape((-1, 1))
        # compute prices, deltas and vegas
        prices = bsPrice(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        deltas = bsDelta(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        vegas = bsVega(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        return spots, spots, prices, deltas, vegas   

# In[31]:


def test(generator, 
         sizes, 
         nTest, 
         simulSeed=None, 
         testSeed=None, 
         weightSeed=None, 
         deltidx=0):

    # simulation
    print("simulating training, valid and test sets")
    xTrain, yTrain, dydxTrain = generator.trainingSet(max(sizes), seed=simulSeed)
    xTest, xAxis, yTest, dydxTest, vegas = generator.testSet(num=nTest, seed=testSeed)
    print("done")

    # neural approximator
    print("initializing neural appropximator")
    regressor = Neural_Approximator(xTrain, yTrain, dydxTrain)
    print("done")
    
    predvalues = {}    
    preddeltas = {}
    for size in sizes:        
            
        print("\nsize %d" % size)
        regressor.prepare(size, False, weight_seed=weightSeed)
            
        t0 = time.time()
        regressor.train("standard training")
        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("standard", size)] = predictions
        preddeltas[("standard", size)] = deltas[:, deltidx]
        t1 = time.time()
        
        regressor.prepare(size, True, weight_seed=weightSeed)
            
        t0 = time.time()
        regressor.train("differential training")
        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("differential", size)] = predictions
        preddeltas[("differential", size)] = deltas[:, deltidx]
        t1 = time.time()
        
    return xAxis, yTest, dydxTest[:, deltidx], vegas, predvalues, preddeltas

def graph(title, predictions, xAxis, xAxisName, yAxisName, targets, sizes, computeRmse=False, weights=None):
    
    numRows = len(sizes)
    numCols = 2

    fig, ax = plt.subplots(numRows, numCols, squeeze=False)
    fig.set_size_inches(4 * numCols + 1.5, 4 * numRows)

    for i, size in enumerate(sizes):
        ax[i,0].annotate("size %d" % size, xy=(0, 0.5), 
          xytext=(-ax[i,0].yaxis.labelpad-5, 0),
          xycoords=ax[i,0].yaxis.label, textcoords='offset points',
          ha='right', va='center')
  
    ax[0,0].set_title("standard")
    ax[0,1].set_title("differential")
    
    for i, size in enumerate(sizes):        
        for j, regType, in enumerate(["standard", "differential"]):

            if computeRmse:
                errors = 100 * (predictions[(regType, size)] - targets)
                if weights is not None:
                    errors /= weights
                rmse = np.sqrt((errors ** 2).mean(axis=0))
                t = "rmse %.2f" % rmse
            else:
                t = xAxisName
                
            ax[i,j].set_xlabel(t)            
            ax[i,j].set_ylabel(yAxisName)

            ax[i,j].plot(xAxis*100, predictions[(regType, size)]*100, 'co',                          markersize=2, markerfacecolor='white', label="predicted")
            ax[i,j].plot(xAxis*100, targets*100, 'r.', markersize=0.5, label='targets')

            ax[i,j].legend(prop={'size': 8}, loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("% s -- %s" % (title, yAxisName), fontsize=16)
    plt.show()


# In[283]:


# simulation set sizes to perform
sizes = [1024, 8192]

# show delta?
showDeltas = True

# seed
# simulSeed = 1234
simulSeed = np.random.randint(0, 10000) 
print("using seed %d" % simulSeed)
weightSeed = None

# number of test scenarios
nTest = 100    

# go
generator = BlackScholes()
xAxis, yTest, dydxTest, vegas, values, deltas =     test(generator, sizes, nTest, simulSeed, None, weightSeed)


# In[284]:


# show predicitions
graph("Black & Scholes", values, xAxis, "", "values", yTest, sizes, True)

# show deltas
if showDeltas:
    graph("Black & Scholes", deltas, xAxis, "", "deltas", dydxTest, sizes, True)


# In[285]:


# Test the function
S0 = 100.0
V0 = 0.05
r = 0.05
kappa = 2.0
theta = 0.05
sigma = 0.3
rho = -0.7
T = 1.0
N = 1000


# In[354]:


class Heston:
    
    def __init__(self,numberTimsteps=100,vol=0.05, spot=100, K=110, r=0.05, T=1.0, sigma=0.03, kappa=2.0, theta=0.05, rho=-0.7, xi=0.1, isCall=True):
        
        self.numberTimsteps=numberTimsteps
        self.vol=vol
        self.spot = spot
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.kappa = kappa
        self.theta = theta
        self.rho = rho
        self.xi = xi
        self.isCall = isCall
    
    '''def __init__(self, 
                 vol=0.2,
                 S0=110.0, 
                 V0=0.05, 
                 K=100.0,
                 volMult=1.5):
        
        self.spot = 1
        self.vol = vol
        self.S0 = S0
        self.V0 = V0
        self.K = K
        self.volMult = volMult'''
    
        
        
    def test_set_heston(self, num=100):
    
        numMC=10_000
        S = self.spot
        vol = np.full((num, 1), self.vol)
        #changement 2*self.numberTimsteps
        returns1 = np.random.normal(size=[num,self.numberTimsteps,numMC])
        returns2 = np.random.normal(size=[num,self.numberTimsteps,numMC])#mettre les dimensions , 2e dim : nb d'échantillon : 1000 ligne, 1er dim: nb de timestep
        returns2 = self.rho * returns1 + (1 - self.rho**2)**0.5 * returns2
        
        #self.T1
        timestep = self.T / self.numberTimsteps
        
        for k in range(self.numberTimsteps):
            print(k)
            S *= np.exp((self.r - np.max(vol, 0)**2) * 0.5 * timestep + timestep**0.5 * returns1[:,k,0].reshape((-1, 1)))
            vol = vol + self.kappa * (self.theta - np.max(vol, 0)) * timestep + self.sigma * (np.max(vol, 0)**0.5) * returns2[:,k,0].reshape((-1,1))
        
        Spots=np.concatenate((np.copy(S),np.copy(vol)),axis=1)
        spots = np.repeat(S, repeats=numMC, axis=1)
        vols = np.repeat(vol, repeats=numMC, axis=1)
        
        Y=self.payoffCallHeston(S, vols, (returns1, returns2))
        Ya=self.payoffCallHeston(spots, vols, (-returns1, -returns2))
        #antitetic path ,réduire la variance
        Z= self.diffmatCallHeston(spots, vols, (returns1, returns2)) 
        Za=self.diffmatCallHeston(spots, vols, (-returns1, -returns2)) #reduire l"ecart

        Y= 0.5 * (Y + Ya)
        Z=0.5 * (Z + Za)
        prices= Y.reshape(-1, 1)
        deltas=Z
        vegas=0
    
        figPrice=plt.figure(figsize=(12,12))
        ax=figPrice.add_subplot(projection='3d')
        ax.scatter(100*S,100*vol,100*prices)
        plt.show()
        
        figDeltaplt.figure(figsize=(12,12))
        ax=figDelta.add_subplot(projection='3d')
        ax.scatter(100*S,100*vol,100*deltas[:,0])
        plt.show()
        
        return Spots,Spots,prices,deltas,vegas
                        
    # training set: returns S1 (mx1), C2 (mx1) and dC2/dS1 (mx1)
    def trainingSetHeston(self, m, anti=True, seed=None):
        numberTimsteps=self.numberTimsteps
        np.random.seed(seed)
        
        # 2 sets of normal returns
        numMC = 1_0000
        returns1=np.random.normal(size=[m,2*self.numberTimsteps, numMC])
        returns2=np.random.normal(size=[m,2*self.numberTimsteps, numMC])
        returns2=self.rho*returns1+(1-self.rho**2)**0.5*returns2
        
        S, vol = self.spot, self.vol
        time_step1= self.T/self.numberTimsteps

        for k in range(self.numberTimsteps):
            S*=np.exp((self.r - (np.maximum(vol,0)**2) * 0.5) * time_step1 + time_step1**0.5* returns1[:,k,0]*np.maximum(vol,0))
            vol = vol + self.kappa*(self.theta - np.maximum(vol,0))*time_step1 + self.sigma * (np.maximum(vol, 0)**0.5)*returns2[:,k,0]
        spots = np.copy(S).reshape((-1, 1))
        spots = np.repeat(spots, repeats=numMC, axis=1)
        vol = vol.reshape((-1, 1))
        vols = np.repeat(vol, repeats=numMC, axis=1)

        Y,Ya = self.payoffCallHeston(spots, vols, (returns1, returns2)), self.payoffCallHeston(spots, vols, (-returns1, -returns2))
        Z,Za = self.diffmatCallHeston(spots, vols, (returns1, returns2)), self.diffmatCallHeston(spots, vols, (-returns1,-returns2))
        Y,Z= 0.5* (Y+Ya), 0.5* (Z+Za)
        #Y= 0.5* (Y+Ya)
                                                                                                               
                                                                                                               
        figPrice = plt. figure(figsize=(12, 12))
        ax = figPrice.add_subplot(projection='3d')
        ax.set(title="Payoff Call Heston", xlabel="Spot", ylabel="Vol",zlabel="Payoff Option")
        ax.scatter (100*spots[:,0], vol, Y)
        plt.show()
        
                                                                                                              
        figDelta = plt.figure(figsize=(12, 12))
        ax = figDelta.add_subplot(projection='3d')
        ax.scatter(100*spots[:,0], 100*vol, 100*Z[:, 0])
        ax.set(title="Payoff Diff Call Heston", xlabel="Spot", ylabel="Vol",zlabel="Payoff Diff Option")
        plt.show()
        
        
        '''                          
        spots_sorted = np.sort(spots[:,0], axis=0)
        length = len(spots_sorted)
        percent5Beg,percent5End = spots_sorted[int(beTail * length)],spots_sorted[int(endTail*length)]
        X = spots[:0].reshape((-1,1)) 
        X=np.concatenate((spots[:0].reshape((-1,1)), vol[:,0].reshape((-1,1))), axis=1)                                                                                                    
                                                                                                               
        
        prices, deltas, vegas = Y.reshape((-1,1)),Z,0
        # Set up the arrays to store the asset and volatility processes
        S1 = np.zeros(N+1)
        V1 = np.zeros(N+1)
    
        # Set the initial values
        S[0] = S0
        V[0] = V0

        # SDE
        # Generate the asset and volatility processes
        for i in range(1, N+1):
            z1 = np.random.normal(V0,0.1)
            z2 = np.random.normal(V0,0.1)
            S[i] = S[i-1] * np.exp((r - 0.5*V[i-1])*dt + np.sqrt(V[i-1]*dt)*z1)
            V[i] = V[i-1] + kappa*(theta - V[i-1])*dt + sigma*np.sqrt(V[i-1]*dt)*(rho*z1 + np.sqrt(1 - rho**2)*z2)

        # payoff
        pay = np.zeros(len(S))
        for i in range(1, N+1):
            pay[i] = np.maximum(0,  S[i]-self.K)
        
        # two antithetic paths
        if anti:
            
            R2a = np.exp(-0.5*self.vol*self.vol*(self.T2-self.T1) \
                    - self.vol*np.sqrt(self.T2-self.T1)*returns[:,1])
            S2a = S1 * R2a             
            paya = np.maximum(0,  self.K-S2a)
            
            X = S1
            Y = 0.5 * (pay + paya)
    
            # differentials
            Z1 =  np.where(S2 < self.K, -R2, 0.0).reshape((-1,1)) 
            Z2 =  np.where(S2a < self.K, -R2a, 0.0).reshape((-1,1)) 
            Z = 0.5 * (Z1 + Z2)
                    
        # standard
        else:
        
            X = S1
            Y = pay
            
            # differentials
            Z =  np.where(S2 > self.K, -R2, 0.0).reshape((-1,1)) 
        '''
        return X,Y.reshape([-1,1]),Z
    
    '''# test set: returns a grid of uniform spots 
    # with corresponding ground true prices, deltas and vegas
    def testSetHeston(self, lower=0.35, upper=1.65, num=100, seed=None):
        spots = np.linspace(lower, upper, num).reshape((-1, 1))
        
        
        
        
        # compute prices, deltas and vegas
        prices = bsPrice(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        deltas = bsDelta(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        vegas = bsVega(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        return spots, spots, prices, deltas, vegas'''
    
    def diffmatCallHeston(self, S1, vol1, returns):
        diff = np.zeros((np.shape(S1)[0],2))
        h = 0.1
        Column1 = (self.payoffCallHeston(S1+h, vol1, returns)-self.payoffCallHeston(S1, vol1, returns))/(2*h)
        Column2 = (self.payoffCallHeston(S1, vol1 + h, returns) - self.payoffCallHeston(S1, vol1 - h, returns)) / (2*h)
        diff[:,0]=Column1.reshape((-1,))
        diff[:,1] = Column2.reshape ((-1,))
        return diff[:,0].reshape((-1,1))


    def payoffCallHeston(self, S1, vol1, returns):
        S, vol = np.copy (S1), np.copy(vol1)
        #time_step2 = self.T / self.numberTimsteps
        #for i in range(self.numberTimsteps,2*numberTimsteps):
        #    S*=np.exp((self.r - (np.maximum(vol,0)**2)*0.5)*time_step2+time_step2**0.5*returns[0][:,i,:]*np.maximum(vol,0))
        #    vol = vol + self.kappa * (self.theta - np.maximum(vol, 0))*time_step2+self.sigma*(np.maximum(vol,0)**0.5)*returns[1][:,i,:]
        # payoff
            #print ("shape S = ", S)
        #if self.boolCall:
        Y= np.mean(np.maximum(0, S - self.K), axis=1).reshape((-1, 1))
        #else:
            #Y= np.mean (np.maximum(0, self.K - S), axis=1).reshape((-1, 1))
        return Y


# # Test Payoff heston

# In[355]:


# Test the function
import numpy as np
S = 100.0
vol = 0.05
r = 0.05
kappa = 2.0
theta = 0.05
sigma = 0.3
rho = -0.7
T = 1.0


numMC = 1_000
N = 1000
m=100
numberTimsteps=100
time_step1= T/numberTimsteps


# In[356]:


#generation
returns1=np.random.normal(size=[m,2*numberTimsteps, numMC])
returns2=np.random.normal(size=[m,2*numberTimsteps, numMC])
returns2=rho*returns1+(1-rho**2)**0.5*returns2

for k in range(numberTimsteps):
    S*=np.exp((r - (np.maximum(vol,0)**2) * 0.5) * time_step1 + time_step1**0.5* returns1[:,k,0]*np.maximum(vol,0))
    vol = vol + kappa*(theta - np.maximum(vol,0))*time_step1 + sigma * (np.maximum(vol, 0)**0.5)*returns2[:,k,0]
spots = np.copy(S).reshape((-1, 1))
spots = np.repeat(spots, repeats=numMC, axis=1)
vol = vol.reshape((-1, 1))
vols = np.repeat(vol, repeats=numMC, axis=1)

#Z,Za = diffmatCallHeston(spots, vols, (returns1, returns2)), diffmatCallHeston(spots, vols, (-returns1,-returns2))


# In[357]:


test = Heston()
test.payoffCallHeston(spots,vols,(returns1,returns2))
#test.diffmatCallHeston(spots,vols,(returns1,returns2))


# In[275]:


diff = np.zeros((np.shape(spots)[0],2))


# In[274]:


diff=np.zeros(np.shape(spots)[0])
diff[:0]


# In[181]:


test = Heston()
test
a=np.maximum(2,3)
a


# In[152]:


numMC = 20
m=10
numberTimsteps=10
returns1=np.random.normal(size=[m,2*numberTimsteps, numMC])
returns2=np.random.normal(size=[m,2*numberTimsteps, numMC])
returns=(returns1,returns2)


# In[156]:


len(returns[0][:,11,:])


# In[147]:


time_step2 =T / numberTimsteps
time_step2
numberTimsteps


# In[290]:


test=Heston()
test.trainingSetHeston(100, True, None)


# In[292]:


test2=Heston()
test2.trainingSetHeston(100, True, None)


# In[296]:


test3=Heston()
test3.trainingSetHeston(100, True, None)


# In[308]:


test4=Heston()
test4.diffmatCallHeston(spots,vols,(returns1,returns2))


# In[358]:


def test_heston(generator, 
         sizes, 
         nTest, 
         simulSeed=None, 
         testSeed=None, 
         weightSeed=None, 
         deltidx=0):

    # simulation
    print("simulating training, valid and test sets")
    xTrain, yTrain, dydxTrain = generator.trainingSetHeston(max(sizes), seed=simulSeed)
    xTest, xAxis, yTest, dydxTest, vegas = generator.test_set_heston(num=nTest, seed=testSeed)
    print("done")

    # neural approximator
    print("initializing neural appropximator")
    regressor = Neural_Approximator(xTrain, yTrain, dydxTrain)
    print("done")
    
    predvalues = {}    
    preddeltas = {}
    for size in sizes:        
            
        print("\nsize %d" % size)
        regressor.prepare(size, False, weight_seed=weightSeed)
            
        t0 = time.time()
        regressor.train("standard training")
        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("standard", size)] = predictions
        preddeltas[("standard", size)] = deltas[:, deltidx]
        t1 = time.time()
        
        regressor.prepare(size, True, weight_seed=weightSeed)
            
        t0 = time.time()
        regressor.train("differential training")
        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("differential", size)] = predictions
        preddeltas[("differential", size)] = deltas[:, deltidx]
        t1 = time.time()
        
    return xAxis, yTest, dydxTest[:, deltidx], vegas, predvalues, preddeltas

def graph(title, predictions, xAxis, xAxisName, yAxisName, targets, sizes, computeRmse=False, weights=None):
    
    numRows = len(sizes)
    numCols = 2

    fig, ax = plt.subplots(numRows, numCols, squeeze=False)
    fig.set_size_inches(4 * numCols + 1.5, 4 * numRows)

    for i, size in enumerate(sizes):
        ax[i,0].annotate("size %d" % size, xy=(0, 0.5), 
          xytext=(-ax[i,0].yaxis.labelpad-5, 0),
          xycoords=ax[i,0].yaxis.label, textcoords='offset points',
          ha='right', va='center')
  
    ax[0,0].set_title("standard")
    ax[0,1].set_title("differential")
    
    for i, size in enumerate(sizes):        
        for j, regType, in enumerate(["standard", "differential"]):

            if computeRmse:
                errors = 100 * (predictions[(regType, size)] - targets)
                if weights is not None:
                    errors /= weights
                rmse = np.sqrt((errors ** 2).mean(axis=0))
                t = "rmse %.2f" % rmse
            else:
                t = xAxisName
                
            ax[i,j].set_xlabel(t)            
            ax[i,j].set_ylabel(yAxisName)

            ax[i,j].plot(xAxis*100, predictions[(regType, size)]*100, 'co',                          markersize=2, markerfacecolor='white', label="predicted")
            ax[i,j].plot(xAxis*100, targets*100, 'r.', markersize=0.5, label='targets')

            ax[i,j].legend(prop={'size': 8}, loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("% s -- %s" % (title, yAxisName), fontsize=16)
    plt.show()


# In[361]:


# simulation set sizes to perform
sizes = [1024, 8192]

# show delta?
showDeltas = True

# seed
# simulSeed = 1234
simulSeed = np.random.randint(0, 10000) 
print("using seed %d" % simulSeed)
weightSeed = None

# number of test scenarios
nTest = 100    

# go
generator = Heston()
xAxis, yTest, dydxTest, vegas, values, deltas =     test_heston(generator, sizes, nTest, simulSeed, None, weightSeed)


# # Test Heston generation

# In[21]:


def simulate_heston(X0, num_paths, num_steps, model_params2):

  sigma, kappa, theta, rho, r, q = model_params2
  # Initialize matrix to store simulated paths
  X = np.zeros((num_paths, num_steps+1, 4))
  X[:, 0, :] = X0
  # Set time step size
  dt = 1/num_steps
  # Loop over time steps
  for t in range(1, num_steps+1):
    # Generate random noise terms
    Z1 = np.random.normal(size=num_paths)
    Z2 = np.random.normal(size=num_paths)
    # Simulate asset price, variance, and correlation
    X[:, t, 0] = X[:, t-1, 0]*np.exp((r - q - 0.5*X[:, t-1, 1])*dt + np.sqrt(X[:, t-1, 1])*np.sqrt(dt)*Z1)
    X[:, t, 1] = X[:, t-1, 1] + kappa*(theta - np.maximum(X[:, t-1, 1], 0))*dt + sigma*np.sqrt(np.maximum(X[:, t-1, 1], 0))*np.sqrt(dt)*(rho*Z1 + np.sqrt(1-rho**2)*Z2)
    X[:, t, 2] = X[:, t-1, 2] + kappa*(theta - np.maximum(X[:, t-1, 2], 0))*dt + sigma*np.sqrt(np.maximum(X[:, t-1, 2], 0))*np.sqrt(dt)*(rho*Z1 + np.sqrt(1-rho**2)*Z2)
    X[:, t, 3] = X[:, t-1, 3] + kappa*(theta - np.maximum(X[:, t-1, 3], 0))*dt + sigma*np.sqrt(np.maximum(X[:, t-1, 3], 0))*np.sqrt(dt)*(rho*Z1 + np.sqrt(1-rho**2)*Z2)

  return X # simulate path





import numpy as np


def HestonDiff(seed, num_samples, num_paths, num_steps, model_params2, option_params):
  # Set seed for reproducibility
  np.random.seed(seed)
  # Generate initial asset values
  S0 = np.linspace(10, 200, num_samples)
  # Initialize lists to store payoffs and differential payoffs
  payoffs = []
  d_payoffs = []
  # Loop over initial asset values
  for s in S0:
    # Set initial state
    X0 = np.array([s, 0.0, 0.0, 0.0])
    # Simulate stock price paths
    S = simulate_heston(X0, num_paths, num_steps, model_params2)
    # Calculate payoffs
    payoff = np.mean(np.maximum(S[:, -1] - option_params[0], 0))
    # Calculate differential payoffs
    d_payoff = np.mean(np.where(S[:, -1] > option_params[0], 1, 0))
    # Store payoffs and differential payoffs
    payoffs.append(payoff)
    d_payoffs.append(d_payoff)
      # Return initial states, payoffs, and differential payoffs
  return S0, payoffs, d_payoffs


# In[20]:


#Generate simulation

option_param_list=(100,90)
model_params2 = (0.1, 0.1, 0.5, -0.9, 0.02, 0)
HestonLSM(123, 100, 2, 252, model_params2, option_param_list)

