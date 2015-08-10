# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:30:59 2015

@author: bordingj
"""

import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from chainerLayers import layers
import pickle
import copy
import time

class TwoLayerLSTMRegressor(FunctionSet):
    def __init__(self, no_feats, no_units, bias=True):
        super(TwoLayerLSTMRegressor, self).__init__(
            x_to_h1 = layers.EfficientLSTMLayer(in_size=no_feats, 
                                                 out_size=no_units,
                                                 nobias=not bias),
            h1_to_h2    = layers.EfficientLSTMLayer(in_size=no_units, 
                                              out_size=no_units,
                                              nobias=not bias),
            h2_to_loss     = layers.MSELayer(no_units,
                                             nobias=not bias),
        )
    
    def make_initial_states(self, batchsize, on_gpu=False, train=True):
        if on_gpu:
            module = cuda
        else:
            module = np
        states = {
            'c1': Variable(module.zeros((batchsize, self.x_to_h1.out_size), 
                                        dtype=np.float32),
                           volatile=not train),
            'h1': Variable(module.zeros((batchsize, self.x_to_h1.out_size), 
                                        dtype=np.float32),
                           volatile=not train),
            'c2': Variable(module.zeros((batchsize, self.h1_to_h2.out_size), 
                                        dtype=np.float32),
                           volatile=not train),
            'h2': Variable(module.zeros((batchsize, self.h1_to_h2.out_size), 
                                        dtype=np.float32),
                           volatile=not train)
        }
        return states
    
    def forward_one_step(self, x_data, y_data, states, dropout_ratio, train=True, compute_loss=False):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        if dropout_ratio==0:
            train=False
            
        h1, c1      = self.x_to_h1(F.dropout(x, ratio=dropout_ratio, train=train), 
                                       states['h1'], states['c1'])
        h2, c2      = self.h1_to_h2(F.dropout(h1, ratio=dropout_ratio, train=train),
                                    states['h2'], states['c2'])
        self.h2_to_loss.return_y = False
        self.h2_to_loss.compute_loss = compute_loss
        loss           = self.h2_to_loss(h2, t)
        states       = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
        
        return states, loss
    
    def predict_one_step(self, x_data, states):
        x = Variable(x_data, volatile=True)
        t = Variable(np.array([[np.nan]]), volatile=True)
        
        h1, c1      = self.x_to_h1(x, states['h1'], states['c1'])
        h2, c2      = self.h1_to_h2(h1, states['h2'], states['c2'])
        self.h2_to_loss.return_y = True
        y           = self.h2_to_loss(h2, t)
        states       = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
        
        return states, y
    
    def save(self, fn):
        pickle.dump(copy.deepcopy(self).to_cpu(), open(fn, 'wb'))

class LSTMRegressor(object):
    
    def __init__(self, no_feats, no_units, bias=True):
        self.no_units = no_units
        self.no_feats = no_feats
        self.model = TwoLayerLSTMRegressor(no_feats, no_units=self.no_units,
                                           bias=bias)

    def forward(self, X_batch, y_batch, on_gpu, optimizer, dropout_ratio,
                initial_states=None,
                 backprob_len=None,
                clip_threshold=8,
                compute_loss=False,
                return_states=False):
        
        T, batchsize, D = X_batch.shape
        
        if on_gpu:
            X_batch = cuda.to_gpu(X_batch)
            y_batch = cuda.to_gpu(y_batch)
        
        if backprob_len is None or backprob_len > T:
            backprob_len = T
            
        accum_loss = 0
        
        if initial_states is None:
            states = self.model.make_initial_states(batchsize=batchsize, 
                                                on_gpu=on_gpu, 
                                                train=True)
        else:
            states = initial_states
            if on_gpu:
                for key, value in states.items():
                    states[key].data = cuda.to_gpu(value.data)
            else:
                for key, value in states.items():
                    states[key].data = cuda.to_cpu(value.data)
                    
        loss_for_return = 0
        for t in range(T):
            x_data = X_batch[t].reshape(batchsize,D)
            y_data = y_batch[t].reshape(batchsize,1)
            
            states, loss = self.model.forward_one_step(x_data=x_data,
                                                       y_data=y_data,
                                                       states=states,
                                                       train=True,
                                                       compute_loss=compute_loss,
                                                       dropout_ratio=dropout_ratio)
            accum_loss += cuda.to_cpu(loss)
            loss_for_return += loss.data
            if (t+1) % backprob_len == 0:
                optimizer.zero_grads()
                accum_loss.backward()
                accum_loss.unchain_backward()  # truncate
                accum_loss = 0
                optimizer.clip_grads(clip_threshold)
                optimizer.update()
        
        if return_states:
            return states
        else:
            return loss_for_return

    def fit(self, X, y, on_gpu=False, 
            batchsize=100, seq_len=20, backprob_len=20,
            rho=0.95, eps=1e-6, dropout_ratio=0,
            n_iterations=1000, clip_threshold=8, print_interval=100,  copy=True):

        T, D = X.shape
        assert D == self.no_feats
        assert T>=seq_len
        
        n_iterations = int(n_iterations)
        
        if copy:
            X = X.copy()
            y = y.copy()
            
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        


        if on_gpu:
            cuda.init()
            self.model.to_gpu()
            
        optimizer = optimizers.AdaDelta(rho=rho, eps=eps)
        optimizer.setup(self.model.collect_parameters())
        
        
        print('training .... ')
        start = time.time()
        for i in range(n_iterations):
            starting_indices = np.random.randint(0, (T-seq_len), 
                                                     batchsize)
            X_batch = np.transpose(np.array([X[idx0:(idx0+seq_len)] \
                            for idx0 in starting_indices]),
                            axes=(1,0,2) )
            
            y_batch = np.array([y[idx0:(idx0+seq_len)] \
                            for idx0 in starting_indices]).T

            if i%print_interval==0:
                compute_loss=True
            else:
                compute_loss=False
                
            loss = self.forward(X_batch=np.ascontiguousarray(X_batch), 
                                y_batch=np.ascontiguousarray(y_batch),
                                on_gpu=on_gpu,
                                optimizer=optimizer, 
                                dropout_ratio=dropout_ratio,
                                initial_states=None,
                                backprob_len=backprob_len, 
                                clip_threshold=clip_threshold,
                                compute_loss=compute_loss)

            if i%print_interval==0:
                print(loss)
                
        
        end = time.time()
        print('training took {} seconds'.format((int(end-start))) )
            
        return self
    
    def one_step_forecasts(self, X, train_test_split, copy=True,
                           on_gpu=False):
        
        assert len(X.shape) == 2
        
        if copy:
            X = X.copy()
        X = X.astype(np.float32)
        
        T, D = X.shape
        assert D==self.no_feats
        assert T>train_test_split

        if on_gpu:
            cuda.init()
            self.model.to_gpu()
            X = cuda.to_gpu(X)
        else:
            self.model.to_cpu()
            
        states = self.model.make_initial_states(batchsize=1, 
                                                on_gpu=on_gpu, 
                                                train=False)

        predictions = np.empty((T-train_test_split,1), dtype=np.float32)
                                   
        for t in range(T):
            X_t = X[t].reshape(1,D)
            states, y = self.model.predict_one_step(x_data=X_t, 
                                                        states=states)
            
                
            if t >= train_test_split:
                if on_gpu:
                    y_hat = cuda.to_cpu(y.data)
                else:
                    y_hat = y.data
                    
                predictions[(t-train_test_split),0] = y_hat
                
        return predictions
    
    def online_fit_and_predict(self, X, y, initial_states=None, on_gpu=False, 
                               lr=0.01, clip_threshold=8, copy=True):

        T, D = X.shape
        assert D == self.no_feats
        
        if copy:
            X = X.copy()
            y = y.copy()
            
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        X_batch = X.reshape(T,1,D)
        y_batch = y.reshape(T,1)
        
        if on_gpu:
            cuda.init()
            self.model.to_gpu()
        else:
            self.model.to_cpu()
            
        optimizer = optimizers.SGD(lr)
        optimizer.setup(self.model.collect_parameters())
        

                
        states = self.forward(X_batch=np.ascontiguousarray(X_batch), 
                            y_batch=np.ascontiguousarray(y_batch),
                            on_gpu=on_gpu,
                            optimizer=optimizer, 
                            dropout_ratio=0,
                            initial_states=initial_states,
                            backprob_len=T, 
                            clip_threshold=clip_threshold,
                            compute_loss=False,
                            return_states=True)
        
        for key, value in states.items():
            value.volatile = True
            
        X_t = np.ascontiguousarray(X_batch[-1]).reshape(1,D)
        
        if on_gpu:
            X_t = cuda.to_gpu(X_t)
            
        states, y = self.model.predict_one_step(x_data=X_t, 
                                                states=states)
            
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
    from sklearn.preprocessing import StandardScaler
    MyScaler = StandardScaler()
    y_transformed = MyScaler.fit_transform(np.log(data['y']))
    X = y_transformed[:-1].reshape((data.shape[0]-1,1))
    y = y_transformed[1:]

    split = int(0.8*X.shape[0])
    X_train = X[:split]
    y_train = y[:split] 
    LSTMRegr = LSTMRegressor(no_feats=1, no_units=50).fit(X_train, y_train, seq_len=13, backprob_len=13,
		                                                n_iterations=2000,
		                                                on_gpu=True, batchsize=100,
		                                                dropout_ratio=0)

    predictions = LSTMRegr.one_step_forecasts(X,split)
    data['predictions'] = np.nan
    data.ix[(split+1):, 'predictions'] = np.exp(MyScaler.inverse_transform(predictions))
    data.plot(figsize=(10,8))
