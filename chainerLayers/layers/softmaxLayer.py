# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:29:31 2015

@author: bordingj
"""


import numpy as np
from chainer import cuda
from chainer import cudnn
from chainer import function

import chainerLayers.cukernels as cuk
from chainerLayers.layers import utils

if cudnn.available:
    from chainer.cudnn import libcudnn
    _algorithm = libcudnn.cudnnSoftmaxAlgorithm['CUDNN_SOFTMAX_ACCURATE']
    _mode = libcudnn.cudnnSoftmaxMode['CUDNN_SOFTMAX_MODE_INSTANCE']

class SoftmaxCrossEntropyLayer(function.Function):
    """
    Simple layer
    This function is used to compute
    y = S(x.dot(W.T)+b)
    where x and W and b are parameters.
    S is the softmax function
    
    In:
        int in_size: number of columns in the input matrix;
        int no_labels: number of columns in the output matrix (e.g. number of hidden states)
        float Wscale: scale of the initialized weight matrix W (default=1.0)
        bool nobias: if false the layer will have a bias parameter vector b (default=True)
        bias float: filler for the bias
    """
    def __init__(self, in_size, no_labels,
                 Wscale=1.0, 
                 nobias=False,
                 bias=0.0,
                 compute_loss=True,
                 return_probs=False,
                 use_cudnn=True):
   
        self.bias = np.float32(bias)
        self.nobias = nobias
        self.in_size = in_size
        self.no_labels = no_labels
        self.compute_loss = compute_loss
        self.use_cudnn = use_cudnn
        self.return_probs = return_probs
        
        #initialize W weight matrix
        self.W = utils.weight_initialization(in_size, no_labels, Wscale)
        self.gW = np.empty_like(self.W)
                      
        if not self.nobias:
            self.b = np.empty((1, no_labels), dtype=np.float32)
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
        z = np.empty((N,self.no_labels), dtype=np.float32)
        z = np.dot(x, self.W.T, out=z)
        if not self.nobias:
            z += self.b
            
        #Apply SoftMax
        self.probs = np.add(z, -np.amax(z, axis=1, keepdims=True))
        self.probs = np.exp(self.probs, out=self.probs)
        self.probs /= self.probs.sum(axis=1, keepdims=True)
        
        if self.return_probs:
            return self.probs,
            
        #Compute Cross entropy Loss
        if self.compute_loss:
            correct_probs = self.probs[np.arange(N,dtype=np.int32), targets]
            self.correct_probs = np.clip(correct_probs, 1e-8, 1.0, 
                                         out=correct_probs)
            loss = -np.log(correct_probs).sum(keepdims=True)/N
        else:
            loss = np.atleast_2d(np.asarray(np.nan,dtype=np.float32))
        return loss,
        
    def forward_gpu(self, inputs):
        x, targets = inputs
        N = x.shape[0]
        
        
        #Linear function
        z = cuda.empty((N,self.no_labels), dtype=np.float32)
        cuk.dot(x, self.W, out=z, transb='t')
        if not self.nobias:
            cuk.addVec2Mat(z, self.b)
        
        self.probs = z
        if cudnn.enabled and self.use_cudnn:
            handle = cudnn.get_default_handle()
            desc = cudnn.get_tensor_desc(z, 1, 1)
            libcudnn.cudnnSoftmaxForward(
                handle, _algorithm, _mode, 1, desc.value, cudnn.get_ptr(z),
                0, desc.value, cudnn.get_ptr(self.probs))
        else:
            cuk.softmax(z, self.probs)
        
        if self.return_probs:
            return self.probs,
            
        if self.compute_loss:
            correct_probs = cuda.empty((N,),dtype=np.float32)
            cuk.getByIndex_LogAndClip(
                                        self.probs, targets,
                                         out=correct_probs)
            loss = -cuda.cumisc.sum(correct_probs, keepdims=True)/N
        else:
            loss = np.atleast_2d(np.array(np.nan,dtype=np.float32))
        
        return loss,

    def backward_cpu(self, inputs, grad_outputs):
        x, targets = inputs
        gloss = grad_outputs[0]
        
        gtargets = None # the function is non-differentiable with respect to the targets
        
        #backpropagate Softmax Cross Entropy Error
        gz = self.probs
        targets = np.atleast_1d(targets)
        targets = targets.ravel()
        N = gz.shape[0]
        utils.dSoftMaxInner(gz, targets, N)
        gz *= gloss[0] / N

        #backpropagate linear function
        gx = np.dot(gz, self.W)
        self.gW += gz.T.dot(x)
        if not self.nobias:
            gb_ones = np.ones((1,N),dtype=np.float32)
            self.gb += np.dot(gb_ones, gz)
        
        return gx, gtargets

    def backward_gpu(self, inputs, grad_outputs):
        x, targets = inputs
        gloss = cuda.to_gpu(grad_outputs[0])
        N = x.shape[0]
        gtargets = None # the function is non-differentiable with respect to the targets
        
        #backpropagate Softmax Cross Entropy Error
        gz = self.probs
        gz = cuk.dSoftmaxCrossEntropy(gz, targets, gloss)
            
        #backpropagate linear function
        gx = cuda.empty_like(x)
        cuk.dot(gz, self.W, out=gx)
        cuk.dotAdd(gz, x, C=self.gW, transa='t')
        if not self.nobias:
            gb_ones = cuda.ones((1,N),dtype=np.float32)
            cuk.dotAdd(gb_ones, gz, C=self.gb)
        
        return gx, gtargets
