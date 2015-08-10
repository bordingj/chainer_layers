import numpy as np
from chainer import cuda, function
import chainerLayers.cukernels as cuk
from chainerLayers.layers import utils

            
class EfficientLSTMLayer(function.Function):
    """
    This is a no peepholes LSTM 
    """
    def __init__(self, in_size, out_size, act_func='tanh', 
                 Wscale=1.0, Vscale=1.0, 
                 nobias=False, bias=0.0):
   
        self.bias = np.float32(bias)
        self.nobias = nobias
        self.in_size = in_size
        self.out_size = out_size
        self.act_func_str = act_func.lower()

        #initialize weight matrices that operates on the input
        self.W = utils.weight_initialization(in_size, out_size*4, Wscale)
        self.gW = np.empty_like(self.W)
        
        self.V = utils.weight_initialization(out_size, out_size*4, Vscale)
        self.gV = np.empty_like(self.V)
        
        if not self.nobias:
            self.b = np.empty((1, out_size*4), dtype=np.float32)
            self.b.fill(self.bias)
            self.gb = np.empty_like(self.b)
    
    @property
    def parameter_names(self):
        if not self.nobias:
            return 'W', 'V', 'b'
        else:
            return 'W', 'V'

    @property
    def gradient_names(self):
        if not self.nobias:
            return 'gW', 'gV', 'gb'
        else:
            return 'gW', 'gV'

    def forward_cpu(self, inputs):
        x, h_tm1, c_tm1 = inputs
        N = x.shape[0]
        z = np.empty((N,self.out_size*4),dtype=np.float32)
        
        z = np.dot(x, self.W.T, out=z)
        z += np.dot(h_tm1, self.V.T)
        if not self.nobias:
            z+= self.b
       
        self.z = utils.lstm_apply_nonlinearity(z=z, out_size=self.out_size)
        
        
        #final memory cell
        self.c = np.empty_like(c_tm1)
        self.h = np.empty_like(h_tm1)
        utils.lstm_final_mem_cell(z=self.z, c_tm1=c_tm1, c=self.c, h=self.h)
        return self.h, self.c


    def forward_gpu(self, inputs):
        x, h_tm1, c_tm1 = inputs
        N = x.shape[0]
        z = cuda.empty((N,self.out_size*4),dtype=np.float32)
        
        z = cuk.dot(x, self.W, out=z, transb = 't')
        cuk.dotAdd(h_tm1, self.V, C=z, transb='t')
        if not self.nobias:
            cuk.addVec2Mat(z, self.b)
       
        self.z = z
        
        self.c = cuda.empty_like(c_tm1)
        self.h = cuda.empty_like(h_tm1)
        
        cuk.lstm_forward(z=z, c_tm1=c_tm1, c=self.c, 
                         h=self.h, out_size=self.out_size)
            
        return self.h, self.c

        

    def backward_cpu(self, inputs, grad_outputs):
        gh, gc = grad_outputs
        x, h_tm1, c_tm1 = inputs
        
        if gh is None:
            gh = np.array([[0]], dtype=np.float32)
            gh_is_none = True
        else:
            gh_is_none = False
        if gc is None:
            gc = np.array([[0]], dtype=np.float32)
            gc_is_none = True
        else:
            gc_is_none = False        
        
        gc_tm1 = self.c
        utils.lstm_backward_finalmem_and_nonlinearities(z=self.z, gh=gh, 
                                             c=self.c, c_tm1=c_tm1, 
                                             gc=gc, 
                                             gh_is_none=gh_is_none, 
                                             gc_is_none=gc_is_none)
        
        gz = self.z
        
        gh_tm1 = np.dot(gz, self.V, out=self.h)

        
        # compute gradient with respect to the input x
        gx = np.empty_like(x)
        gx = np.dot(gz, self.W, out=gx)

        
         # compute gradients of weight matrices
        self.gW += gz.T.dot(x)
        
        self.gV += gz.T.dot(h_tm1)

        
        if not self.nobias:
            N = x.shape[0]
            gb_ones = np.ones((1,N),dtype=np.float32)
            self.gb += np.dot(gb_ones, gz)

        
        return gx, gh_tm1, gc_tm1

    def backward_gpu(self, inputs, grad_outputs):
        gh, gc = grad_outputs
        x, h_tm1, c_tm1 = inputs
        
        if gh is None:
            gh = cuda.to_gpu(np.array([[0]], dtype=np.float32))
            gh_is_none = 1
        else:
            gh_is_none = 0
        if gc is None:
            gc = cuda.to_gpu(np.array([[0]], dtype=np.float32))
            gc_is_none = 1
        else:
            gc_is_none = 0  
        
        
        gc_tm1 = self.c
        cuk.lstm_backward(c=self.c, z=self.z, gh=gh, 
                          gc=gc, c_tm1=c_tm1,
                          gc_is_none=gc_is_none, gh_is_none=gh_is_none)

        gz = self.z
        gh_tm1 = cuk.dot(gz, self.V, out=self.h)

        
        # compute gradient with respect to the input x
        gx = cuda.empty_like(x)
        gx = cuk.dot(gz, self.W, out=gx)

        
         # compute gradients of weight matrices
        cuk.dotAdd(gz, x, C=self.gW, transa='t')
        cuk.dotAdd(gz, h_tm1, C=self.gV, transa='t')
        
       
        if not self.nobias:
            N = x.shape[0]
            gb_ones = cuda.ones((1,N),dtype=np.float32)
            cuk.dotAdd(gb_ones, gz, C=self.gb)

        
        return gx, gh_tm1, gc_tm1


        
        
