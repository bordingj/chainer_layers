# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:29:31 2015

@author: bordingj
"""


import numpy as np
from chainer import cuda
from chainer import function

import chainerLayers.cukernels as cuk
from chainerLayers.layers.utils import weight_initialization

class MSELayer(function.Function):
    """
    This function is used to compute
    y = (x.dot(W.T)+b)
    loss = mean((targets-y)**2)
    where x and W and b are parameters.
    
    In:
        int in_size: number of columns in the input matrix;
        float Wscale: scale of the initialized weight matrix W (default=1.0)
        bool nobias: if false the layer will have a bias parameter vector b (default=True)
        bias float: filler for the bias
    """
    def __init__(self, in_size,
                 Wscale=1.0, 
                 nobias=False,
                 bias=0.0,
                 compute_loss=True,
                 return_y=False,
                 use_cudnn=True):
   
        self.bias = np.float32(bias)
        self.nobias = nobias
        self.in_size = in_size
        self.compute_loss = compute_loss
        self.use_cudnn = use_cudnn
        self.return_y = return_y
        
        #initialize W weight matrix
        self.W = weight_initialization(in_size, 1, Wscale)
        self.gW = np.empty_like(self.W)
                      
        if not self.nobias:
            self.b = np.empty((1, 1), dtype=np.float32)
            self.b.fill(self.bias)
            self.gb = np.empty_like(self.b)
        
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

    def forward_cpu(self, inputs):
        x, targets = inputs
        N = x.shape[0]
        
        #Linear function
        y = np.empty((N,1), dtype=np.float32)
        y = np.dot(x, self.W.T, out=y)
        if not self.nobias:
            y += self.b
        
        if self.return_y:
            return y,
            
        #Compute mean squared error loss
        targets = targets.reshape((N,1))
        self.diff = y
        self.diff -= targets
        if self.compute_loss:
            loss = (self.diff**2).sum()*1.0/(2*N)
            loss = np.atleast_2d(np.asarray(loss,dtype=np.float32))
        else:
            loss = np.atleast_2d(np.asarray(np.nan,dtype=np.float32))
        return loss,
        
    def forward_gpu(self, inputs):
        x, targets = inputs
        N = x.shape[0]
        
        #Linear function
        y = cuda.empty((N,1), dtype=np.float32)
        cuk.dot(x, self.W, out=y, transb='t')
        if not self.nobias:
            cuk.addVec2Mat(y, self.b)
        
        if self.return_y:
            return y,

        self.diff = cuk.vecAdd(y, -targets)
        if self.compute_loss:
            loss = cuda.culinalg.norm(self.diff)**2
            loss = np.atleast_2d(np.array(cuda.to_cpu(loss)))*1.0/(2*N)
        else:
            loss = np.atleast_2d(np.array(np.nan,dtype=np.float32))
        
        return loss,

    def backward_cpu(self, inputs, grad_outputs):
        x, targets = inputs
        gloss = grad_outputs[0]
        N = x.shape[0]
        coeff = gloss*1.0/N
        gy = self.diff
        gy = coeff*gy
        gtargets = None # the function is non-differentiable with respect to the targets
        
        #backpropagate linear function
        gx = np.dot(gy, self.W)
        self.gW += gy.T.dot(x)
        if not self.nobias:
            gb_ones = np.ones((1,N),dtype=np.float32)
            self.gb += np.dot(gb_ones, gy)
        
        return gx, gtargets

    def backward_gpu(self, inputs, grad_outputs):
        
        x, targets = inputs
        gloss = grad_outputs[0]
        N = x.shape[0]
        coeff = gloss*1.0/N
        cuda.culinalg.scale(coeff, self.diff, alpha_real=True)
        gy = self.diff
        gtargets = None
            
        #backpropagate linear function
        gx = cuda.empty_like(x)
        cuk.dot(gy, self.W, out=gx)
        cuk.dotAdd(gy, x, C=self.gW, transa='t')
        if not self.nobias:
            gb_ones = cuda.ones((1,N),dtype=np.float32)
            cuk.dotAdd(gb_ones, gy, C=self.gb)
        
        return gx, gtargets
