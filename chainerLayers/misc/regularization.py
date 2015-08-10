# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:18:28 2015

@author: bordingj
"""

import numpy as np
from chainer import function, Variable
from chainer import cuda
import chainerLayers.cukernels as cuk


class L2(function.Function):

    """Frobenious norm or 2-norm of a matrix or vector, respectively, divided by 2"""

    def __init__(self, W, gW):
        self.W = W
        self.gW = gW

    @property
    def parameter_names(self):
        return 'W',

    @property
    def gradient_names(self):
        return 'gW'
            
    def forward_cpu(self, inputs):
        out = 1.0/2*np.linalg.norm(self.W)**2
        return np.atleast_2d(np.asarray(out,dtype=np.float32)),
        

    def forward_gpu(self, inputs):
        
        out = 1.0/2*cuda.cublas.cublasSnrm2(cuda.get_cublas_handle(), 
                                                 self.W.size, 
                                                 self.W.gpudata, 1)**2
        return np.atleast_2d(np.array(out,dtype=np.float32)),
            
    def backward_cpu(self, inputs, gradient_outputs):
        self.gW += gradient_outputs[0]*self.W
        return np.atleast_2d(np.asarray(0.0,dtype=np.float32)),

    def backward_gpu(self, inputs, gradient_outputs):
        cuk.matAdd(self.gW, self.W, scalar=gradient_outputs[0][0,0])
        return np.atleast_2d(np.asarray(0.0,dtype=np.float32)),

def L2_reg(model, on_gpu, volatile=False):
    l2_reg = 0
    for func in model._get_sorted_funcs():
        for param_data, grad_data, param_name in zip(func[1].parameters, func[1].gradients, func[1].parameter_names):
            if param_name[0] != 'b':
                a_variable = Variable(np.array([[1.0]], dtype=np.float32), volatile=volatile)
                if on_gpu:
                    a_variable.data = cuda.to_gpu(a_variable.data)
                l2_reg += L2(param_data, grad_data)(a_variable)
    return l2_reg

