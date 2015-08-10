# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:29:31 2015

@author: bordingj
"""


import numpy as np
from chainer import cuda
from chainer import function
import chainerLayers.cukernels as cuk
from chainerLayers.layers import utils
    
class SimpleLayer(function.Function):
    """
    This function is used to compute
    y = f(x.dot(W.T)+b)
    where x is an input matrices and W and b are parameters
    and h has the same dimensions as y.
    f is a non-linear elementwise activation function
    
    In:
        int in_size: number of columns in the input matrix;
        int out_size: number of columns in the output matrix (e.g. number of hidden states)
        str act_fun: activation function for the layer (default='tanh')
                    available activation functions: ('tanh', 'sigmoid', 'relu', 'leakyrelu')
        float Wscale: scale of the initialized weight matrix W (default=1.0)
        bool bias: if true the layer will have a bias parameter vector b (default=True)
        bool hot: if true we assumes that the input matrix x is K-hot encoded. Hence, the function should
                    be feed hot indices instead of a full matrix (default=False).
        Note that the function is non-differentiable with respect to its input if hot is True
    """
    def __init__(self, in_size, out_size, act_func='tanh', 
                 Wscale=1.0,
                 nobias=False,
                 bias=0.0,
                 hot=False):
   
        self.bias = bias
        self.nobias = nobias
        self.in_size = in_size
        self.out_size = out_size
        self.hot = hot
        self.act_func_str = act_func.lower()
        
        #initialize W weight matrix
        self.W = utils.weight_initialization(in_size, out_size, Wscale)
        self.gW = np.empty_like(self.W)
                      
        if not self.nobias:
            self.b = np.empty((1, out_size), dtype=np.float32)
            self.b.fill(self.bias)
            self.gb = np.empty_like(self.b)

        available_act_funcs = {
                    'sigmoid': (utils.Sigmoid, utils.dSigmoid),
                    'tanh': (utils.Tanh, utils.dTanh),
                    'relu': (utils.ReLU, utils.dReLU),
                    'leakyrelu': (utils.LeakyReLU, utils.dLeakyReLU)
                    }

        available_cu_act_funcs = {
            'sigmoid': (cuk.sigmoid, cuk.dsigmoid),
            'tanh': (cuk.tanh, cuk.dtanh),
            'relu': (cuk.reLU, cuk.dreLU),
            'leakyrelu': (cuk.leakyReLU, cuk.dleakyReLU)
        }
        
        self.act_func = available_act_funcs[self.act_func_str][0]
        self.dact_func = available_act_funcs[self.act_func_str][1]

        self.cu_act_func = available_cu_act_funcs[self.act_func_str][0]
        self.cu_dact_func = available_cu_act_funcs[self.act_func_str][1]
        
    @property
    def parameter_names(self):
        if not self.nobias:
            return 'W', 'b'
        else:
            return 'W',

    @property
    def gradient_names(self):
        if not self.nobias:
            return 'gW', 'gb'
        else:
            return 'gW', 
    
    def compile_cukernels(self):
        for kernel in self._kernels:
            kernel.prepare()

    def forward_cpu(self, inputs):
        x = inputs[0]
        N = x.shape[0]
        z = np.empty((N,self.out_size), dtype=np.float32)
        
        #Linear function
        if self.hot: # if x is hot indices (k-hot-encoding)
            utils.HotDot(self.W, x, out=z, dont_add=True)
        else:
            z = np.dot(x, self.W.T, out=z)
        if not self.nobias:
            z += self.b
        
        #apply non-linear activation
        if self.act_func_str in ('tanh', 'sigmoid'):
            h = self.act_func(x=z, out=z)
            self.h = h #save h for backpropagation
        elif self.act_func_str in ('leakyrelu', 'relu'):
            h = np.empty_like(z)
            h = self.act_func(x=z, out=h)
            self.z = z #save z for backpropagation
        else:
            raise NotImplementedError('the activation function is not available')
        return h,
        
    def forward_gpu(self, inputs):
        x = inputs[0]
        N = x.shape[0]
        z = cuda.empty((N,self.out_size), dtype=np.float32)
        
        #Linear function
        if self.hot:
            cuk.hotdot(self.W, x, out=z, dont_add=True)
        else:            
            cuk.dot(x, self.W, out=z, transb='t')
        if not self.nobias:
            cuk.addVec2Mat(z, self.b)

        #apply non-linear activation        
        if self.act_func_str in ('tanh', 'sigmoid'):
            h = self.cu_act_func(x=z, out=z)
            self.h = h #save h for backpropagation
        elif self.act_func_str in ('leakyrelu', 'relu'):
            h = cuda.empty_like(z)
            h = self.cu_act_func(x=z, out=h)            
            self.z = z #save z for backpropagation
        else:
            raise NotImplementedError('the activation function is not available')
        return h,

    def backward_cpu(self, inputs, grad_outputs):
        gh = grad_outputs[0]
        x = inputs[0]
        
        if self.act_func_str in ('tanh', 'sigmoid'):
            #backpropagate non-linearities
            gz = self.dact_func(gy=gh, y=self.h, out=self.h)
        elif self.act_func_str in ('leakyrelu', 'relu'):
            #backpropagate non-linearities
            gz = self.dact_func(x=self.z, gy=gh, out=self.z)
        else:
            raise NotImplementedError('the activation function is not available')
            
        #backpropagate linear function
        if self.hot:
            gx = None
            utils.DotHot(gz, x, out=self.gW)
        else:
            gx = np.dot(gz, self.W)
            self.gW += gz.T.dot(x)
        if not self.nobias:
            N = x.shape[0]
            gb_ones = np.ones((1,N),dtype=np.float32)
            self.gb += np.dot(gb_ones, gz)
        
        return gx,

    def backward_gpu(self, inputs, grad_outputs):
        gh = grad_outputs[0]
        x = inputs[0]
        N = x.shape[0]

        if self.act_func_str in ('tanh', 'sigmoid'):
            #backpropagate non-linearities
            gz = self.cu_dact_func(gy=gh, y=self.h, out=self.h)
        elif self.act_func_str in ('leakyrelu', 'relu'):
            #backpropagate non-linearities
            gz = self.cu_dact_func(x=self.z, gy=gh, out=self.z)
        else:
            raise NotImplementedError('the activation function is not available')
            
        #backpropagate linear function
        if self.hot:
            gx = None
            cuk.dothot(gz, x, in_size=self.in_size, out=self.gW)
        else:
            gx = cuda.empty_like(x)
            cuk.dot(gz, self.W, out=gx)
            cuk.dotAdd(gz, x, C=self.gW, transa='t')
        if not self.nobias:
            gb_ones = cuda.ones((1,N),dtype=np.float32)
            cuk.dotAdd(gb_ones, gz, C=self.gb)
        
        return gx,
