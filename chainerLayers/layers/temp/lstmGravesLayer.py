import numpy as np
from chainer import cuda, function
import chainerLayers.cukernels as cuk
from chainerLayers.layers.utils import weight_initialization
from numba import jit
from chainerLayers.layers.cpu_activations import *
import math

@jit('void(f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:])')
def final_mem_cell(z, c_tilde, c_tm1, c, h):
    N, M = c.shape
    for i in range(N):
        for j in range(M):
            i_t = z[i,j]
            f_t = z[i,(j+M)]
            o_t = z[i,(j+M*2)]
            c_tilde_t = c_tilde[i,j]
            
            c_k = f_t*c_tm1[i,j] + i_t*c_tilde_t
            
            c[i,j] = c_k
            h[i,j] = math.tanh(c_k)*o_t

@jit('void(f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], b1, b1)')   
def backward_finalmem_and_nonlinearities(z, c_tilde,
                                         gh, c, c_tm1, gc, gh_is_none, gc_is_none):
    N, M = c.shape
    for i in range(N):
        for j in range(M):
            i_t = z[i,j]
            f_t = z[i,(j+M)]
            o_t = z[i,(j+M*2)]
            c_tilde_t = c_tilde[i,j]
            
            tanh_c_k = math.tanh(c[i,j])
            
            if gh_is_none:
                gh_k = np.float32(0)
            else:
                gh_k = gh[i,j]
            if gc_is_none:
                gc_k = np.float32(0)
            else:
                gc_k = gc[i,j]
            
            q_k = gh_k * o_t * (1 - tanh_c_k**2) + gc_k
            c[i,j] = q_k*f_t #we use the memory for c as gc_tm1_1
            
            gi_k = q_k* c_tilde_t * i_t * (1-i_t);

            c_tilde[i,j] = q_k* i_t * (1-c_tilde_t*c_tilde_t);

            gf_k = q_k* c_tm1[i,j] * f_t * (1-f_t);

            z[i,(j+M*2)] = gh_k* tanh_c_k * o_t * (1-o_t);

            z[i,j] = gi_k;

            z[i,(j+M)] = gf_k;
            
class StandardLSTMLayer(function.Function):
    """

    """
    def __init__(self, in_size, out_size, act_func='tanh', 
                 Wscale=1.0, Uscale=1.0, Vscale=1.0,
                 nobias=False, bias=0.0):
   
        self.bias = np.float32(bias)
        self.nobias = nobias
        self.in_size = in_size
        self.out_size = out_size
        self.act_func_str = act_func.lower()

        #initialize weight matrices that operates on the input
        self.W = weight_initialization(in_size, out_size*3, Wscale)
        self.gW = np.empty_like(self.W)
        
        self.V = weight_initialization(out_size, out_size*3, Uscale)
        self.gV = np.empty_like(self.V)

        #initialize weight matrices that operates on the input
        self.Wc = weight_initialization(in_size, out_size, Wscale)
        self.gWc = np.empty_like(self.Wc)
        
        self.Vhc = weight_initialization(out_size, out_size, Uscale)
        self.gVhc = np.empty_like(self.Vhc)

        self.Vcc = weight_initialization(out_size, out_size, Vscale)
        self.gVcc = np.empty_like(self.Vcc)
        
        if not self.nobias:
            self.b = np.empty((1, out_size*3), dtype=np.float32)
            self.b.fill(self.bias)
            self.gb = np.empty_like(self.b)

            self.bc = np.empty((1, out_size), dtype=np.float32)
            self.bc.fill(self.bias)
            self.gbc = np.empty_like(self.bc)
    
    
    @property
    def parameter_names(self):
        if not self.nobias:
            return 'W', 'V', 'b','Wc', 'Vhc', 'bc', 'Vcc'
        else:
            return 'W', 'V', 'Wc', 'Vhc', 'Vcc'

    @property
    def gradient_names(self):
        if not self.nobias:
            return 'gW', 'gV', 'gb','gWc', 'gVhc', 'gbc', 'gVcc'
        else:
            return 'gW', 'gV', 'gWc', 'gVhc', 'gVcc'

    def forward_cpu(self, inputs):
        x, h_tm1, c_tm1 = inputs
        N = x.shape[0]
        z = np.empty((N,self.out_size*3),dtype=np.float32)
        
        z = np.dot(x, self.W.T, out=z)
        z += np.dot(h_tm1, self.V.T)
        
        c_tilde = np.empty((N, self.out_size), dtype=np.float32)
        c_tilde = np.dot(x, self.Wc.T, out=c_tilde)
        c_tilde += np.dot(h_tm1, self.Vhc.T)
        c_tilde += np.dot(c_tm1, self.Vcc.T)
        
        if not self.nobias:
            z += self.b
            c_tilde += self.bc
       
        self.z = _Sigmoid(x=z, out=z)
        self.c_tilde = _tanh(x=c_tilde, out=c_tilde)
        
        
        #final memory cell
        self.c = np.empty_like(c_tm1)
        self.h = np.empty_like(h_tm1)
        final_mem_cell(z=self.z, c_tilde=self.c_tilde, c_tm1=c_tm1, 
                       c=self.c, h=self.h)
        return self.h, self.c


    def forward_gpu(self, inputs):
        x, h_tm1, c_tm1 = inputs
        N = x.shape[0]
        z = cuda.empty((N,self.out_size*4),dtype=np.float32)
        
        z = cuk.dot(x, self.W, out=z, transb = 't')
        cuk.dotAdd(h_tm1, self.U, C=z, transb='t')
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
        
        
        backward_finalmem_and_nonlinearities(z=self.z, c_tilde=self.c_tilde,
                                             gh=gh, c=self.c, 
                                             c_tm1=c_tm1, gc=gc, 
                                             gh_is_none=gh_is_none, 
                                             gc_is_none=gc_is_none)
        
              
        
        gc_tilde = self.c_tilde
        gc_tm1 = self.c  
        gc_tm1 += np.dot(gc_tilde, self.Vcc)
        
        gz = self.z
        gh_tm1 = np.dot(gz, self.V, out=self.h)
        gh_tm1 += np.dot(gc_tilde, self.Vhc)

        
        # compute gradient with respect to the input x
        gx = np.empty_like(x)
        gx = np.dot(gz, self.W, out=gx)
        gx += np.dot(gc_tilde, self.Wc)
        
         # compute gradients of weight matrices
        self.gW += gz.T.dot(x)
        
        self.gV += gz.T.dot(h_tm1)

        self.gWc += gc_tilde.T.dot(x)
        
        self.gVhc += gc_tilde.T.dot(h_tm1)
        
        self.gVcc += gc_tilde.T.dot(c_tm1)
        
        if not self.nobias:
            N = x.shape[0]
            gb_ones = np.ones((1,N),dtype=np.float32)
            self.gb += np.dot(gb_ones, gz)
            self.gbc += np.dot(gb_ones, gc_tilde)
        
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
        gh_tm1 = cuk.dot(gz, self.U, out=self.h)

        
        # compute gradient with respect to the input x
        gx = cuda.empty_like(x)
        gx = cuk.dot(gz, self.W, out=gx)

        
         # compute gradients of weight matrices
        cuk.dotAdd(gz, x, C=self.gW, transa='t')
        cuk.dotAdd(gz, h_tm1, C=self.gU, transa='t')
        
       
        if not self.nobias:
            N = x.shape[0]
            gb_ones = cuda.ones((1,N),dtype=np.float32)
            cuk.dotAdd(gb_ones, gz, C=self.gb)

        
        return gx, gh_tm1, gc_tm1


        
        
