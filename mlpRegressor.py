# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:30:59 2015

@author: bordingj
"""

import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from chainerLayers import layers
from chainerLayers.misc.regularization import L2_reg
import pickle
import copy
import time

class TwoLayerMLP(FunctionSet):
    def __init__(self, no_feats, no_units, act_func):
        super(TwoLayerMLP, self).__init__(
            x_to_h1 = layers.SimpleLayer(in_size=no_feats, 
                                                 out_size=no_units,
                                                 act_func=act_func),
            h1_to_h2    = layers.SimpleLayer(in_size=no_units, 
                                              out_size=no_units,
                                              act_func=act_func),
            h2_to_loss     = layers.MSELayer(no_units),
        )
    
    
    def forward(self, x_data, y_data, compute_loss, dropout_ratio=0.5, train=True):
        
            
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)
        if dropout_ratio==0:
            train=False
            
        h1      = self.x_to_h1( F.dropout(x, ratio=dropout_ratio, train=train) )
        h2      = self.h1_to_h2( F.dropout(h1, ratio=dropout_ratio, train=train) )
        self.h2_to_loss.return_y = False
        self.h2_to_loss.compute_loss = compute_loss
        loss = self.h2_to_loss(h2, t)
        
        return loss
    
    def predict(self, x_data):
        x = Variable(x_data, volatile=True)
        t = Variable(np.array([[np.nan]]), volatile=True)
        
        h1      = self.x_to_h1( x )
        h2      = self.h1_to_h2( h1 )
        self.h2_to_loss.return_y = True
        y       = self.h2_to_loss(h2, t)
        
        return y
    
    def save(self, fn):
        pickle.dump(copy.deepcopy(self).to_cpu(), open(fn, 'wb'))

class MLPRegressor(object):
    
    def __init__(self, no_feats, no_units, act_func='leakyrelu'):
        self.no_units = no_units
        self.no_feats = no_feats
        self.model = TwoLayerMLP(no_feats, no_units=self.no_units,
                                 act_func=act_func)
        
    def fit(self, X, y, on_gpu=False, 
            batchsize=100, lr=0.01, momentum=0.9,  alpha=1e-4, dropout_ratio=0.25,
            n_iterations=1000, clip_threshold=8, print_interval=100,  copy=True):
                
        N, no_feats = X.shape
        assert no_feats == self.no_feats
        
        n_iterations = int(n_iterations)
        
        if copy:
            X = X.copy()
            y = y.copy()
            
        X = X.astype(np.float32)
        y = y.astype(np.float32).reshape((N,1))
        
        X_minibatch = np.empty((batchsize,no_feats),dtype=np.float32)
        y_minibatch = np.empty((batchsize,1),dtype=np.float32)
        
        if on_gpu:
            cuda.init()
            self.model.to_gpu()        
            
        optimizer = optimizers.MomentumSGD(lr=lr,momentum=momentum)
        optimizer.setup(self.model.collect_parameters())
        
        print('training .... ')
        start = time.time()
        for i in range(n_iterations):
            indices = np.random.randint(0, N, batchsize).astype(np.int32)
            X_minibatch = X[indices]
            y_minibatch = y[indices]
            assert X_minibatch.flags.c_contiguous
            assert y_minibatch.flags.c_contiguous
            
            if on_gpu:      
                X_minibatch = cuda.to_gpu(X_minibatch)
                y_minibatch = cuda.to_gpu(y_minibatch)
                
            if i%print_interval==0:
                compute_loss=True
            else:
                compute_loss=False
                
            loss = self.model.forward(X_minibatch, y_minibatch, 
                               dropout_ratio=dropout_ratio, train=True, 
                               compute_loss=compute_loss)
            
            loss += alpha*L2_reg(self.model, on_gpu=on_gpu)
                               
            optimizer.zero_grads()
            loss.backward()
            optimizer.clip_grads(clip_threshold)
            optimizer.update()
            if i%print_interval==0:
                print(loss.data)
        end = time.time()
        print('training took {} seconds'.format((int(end-start))) )
        return self
    
    def predict(self, X, on_gpu=False, copy=True):
        
        N, no_feats = X.shape
        assert no_feats == self.no_feats
            
        if copy:
            X = X.copy()
            
        X = X.astype(np.float32)

        if on_gpu:
            cuda.init()
            self.model.to_gpu()
            X = cuda.to_gpu(X)
        else:
            self.model.to_cpu()
        
        y = self.model.predict(X)
        
        return cuda.to_cpu(y.data)
        
#%%
if __name__ == "__main__":
	import pandas as pd
	import matplotlib.pyplot as plt
	from datetime import datetime
	import matplotlib
	matplotlib.style.use('ggplot')
	pd.options.display.mpl_style = 'default'

	data = pd.read_stata('http://www.stata-press.com/data/r12/air2.dta')
	times = np.floor(data['time']).astype(np.int)
	time_index = []
	for i, year in enumerate(times):
	    month = (i+1)%12
	    if month==0:
		month=12
	    time_index.append(datetime(year,month,1))
	data.index = time_index
	data.drop(['t','time'],axis=1, inplace=True)
	data.columns =['y']

	from statsmodels.tsa.tsatools import lagmat
	from sklearn.preprocessing import StandardScaler
	MyScaler = StandardScaler()
	y_transformed = MyScaler.fit_transform(np.log(data['y']))
	D=13
	X = lagmat(y_transformed ,maxlag=D)
	y = y_transformed

	split = int(0.8*X.shape[0])
	X_train = X[:split]
	y_train = y[:split]
	X_test = X[split:]
	y_test = y[split:]

	MLPRegr = MLPRegressor(no_feats=D, no_units=50).fit(X_train, y_train, n_iterations=2000,
		                                                on_gpu=True, batchsize=100,
		                                                alpha=1e-4, dropout_ratio=0)

	fitted_values = np.exp(MyScaler.inverse_transform(MLPRegr.predict(X_train)))
	predictions = np.exp(MyScaler.inverse_transform(MLPRegr.predict(X_test)))
	data['fitted_values'] = np.nan
	data['predictions'] = np.nan
	data.ix[:split, 'fitted_values'] = fitted_values
	data.ix[split:, 'predictions'] = predictions
	data.plot(figsize=(10,8))
