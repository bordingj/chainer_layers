import numpy as np
from chainer import cuda, function
import chainerLayers.cukernels as cuk
from chainerLayers.layers.utils import weight_initialization
from chainerLayers.layers.cpu_activations import *


@jit('float32[:,:](float32[:,:], float32[:,:], float32[:,:], float32[:,:])', 
     nopython=True, nogil=True)
def _Multiply_and_dSigmoid(a, b, y, out):
    N, M = out.shape
    for i in range(N):
        for j in range(M):
            out = a[i,j] * b[i,j] * y[i,j] * (1 - y[i,j])
    return out

def _Multiply_dtanh_and_add(o, gh, h, gc, out):
    N, M = out.shape
    for i in range(N):
        for j in range(M):
            out[i,j] = gh[i,j] * (1 - h[i,j] * h[i,j])*o[i,j]+gc[i,j]
    return out
    
class LSTMRecurrentLayer(function.Function):
    """

    """
    def __init__(self, in_size, out_size, act_func='tanh', 
                 Wscale=1.0, Uscale=1.0, 
                 nobias=False, bias=0.0):
   
        self.bias = np.float32(bias)
        self.nobias = nobias
        self.in_size = in_size
        self.out_size = out_size
        self.act_func_str = act_func.lower()

        #initialize weight matrices that operates on the input
        self.W = weight_initialization(in_size, out_size*4, Wscale)
        self.gW = np.empty_like(self.W)
        
        self.Wi = self.W[:,]
        self.gWi = np.empty_like(self.Wi)
        
        self.Wf = weight_initialization(in_size, out_size, Wscale)
        self.gWf = np.empty_like(self.Wf)
        
        self.Wo = weight_initialization(in_size, out_size, Wscale)
        self.gWo = np.empty_like(self.Wo)
        
        self.Wc = weight_initialization(in_size, out_size, Wscale)
        self.gWc = np.empty_like(self.Wc)
        
        #initialize weight matrices that operates on the previous hidden state
        self.Ui = weight_initialization(out_size, out_size, Uscale)
        self.gUi = np.empty_like(self.Ui)
        
        self.Uf = weight_initialization(out_size, out_size, Uscale)
        self.gUf = np.empty_like(self.Wf)
        
        self.Uo = weight_initialization(out_size, out_size, Uscale)
        self.gUo = np.empty_like(self.Uo)
        
        self.Uc = weight_initialization(out_size, out_size, Uscale)
        self.gUc = np.empty_like(self.Uc)
                      
        if not self.nobias:
            self.bi = np.empty((1, out_size), dtype=np.float32).fill(self.bias)
            self.gbi = np.empty_like(self.bi)

            self.bf = np.empty((1, out_size), dtype=np.float32).fill(self.bias)
            self.gbf = np.empty_like(self.bf)

            self.bo = np.empty((1, out_size), dtype=np.float32).fill(self.bias)
            self.gbo = np.empty_like(self.bo)

            self.bc = np.empty((1, out_size), dtype=np.float32).fill(self.bias)
            self.gbc = np.empty_like(self.bc)
      
        
    @property
    def parameter_names(self):
        if self.bias:
            return 'Wi','Wf','Wo','Wc','Ui','Uf','Uo','Uc','bi','bf','bo','bc',
        else:
            return 'Wi','Wf','Wo','Wc','Ui','Uf','Uo','Uc',

    @property
    def gradient_names(self):
        if self.bias:
            return 'gWi','gWf','gWo','gWc','gUi','gUf','gUo','gUc','gbi','gbf','gbo','gbc',
        else:
            return 'gWi','gWf','gWo','gWc','gUi','gUf','gUo','gUc',

    def forward_cpu(self, inputs):
        x, h_tm1, c_tm1 = inputs
        N = x.shape[0]
        i = np.empty((N,self.out_size*4),dtype=np.float32)
        
        #input gate
        i = np.empty_like(h_tm1)
        i = np.dot(x, self.Wi.T, out=i)
        i += np.dot(h_tm1, self.Ui.T)
        if self.bias:
            i += self.bi
        self.i = _Sigmoid(x=i, out=i)

        #forget gate
        f = np.empty_like(h_tm1)
        f = np.dot(x, self.Wf.T, out=f)
        f += np.dot(h_tm1, self.Uf.T)
        if self.bias:
            f += self.bf
        self.f = _Sigmoid(x=f, out=f)

        #output/exposure gate
        o = np.empty_like(h_tm1)
        o = np.dot(x, self.Wo.T, out=o)
        o += np.dot(h_tm1, self.Uo.T)
        if self.bias:
            o += self.bo
        self.o = _Sigmoid(x=o, out=o)

        #New memory cell
        c_tilde = np.empty_like(h_tm1)
        c_tilde = np.dot(x, self.Wc.T, out=c_tilde)
        c_tilde += np.dot(h_tm1, self.Uc.T)
        if self.bias:
            c_tilde += self.bc
        self.c_tilde = _tanh(x=c_tilde, out=c_tilde)
        
        #final memory cell
        self.c = f*c_tm1+i*c_tilde
        h = np.empty_like(h_tm1)
        self.h = _tanh(x=c, out=h)
        self.h *= o
        return self.h, self.c


    
    def backward_cpu(self, inputs, grad_outputs):
        gh, gc = grad_outputs
        x, h_tm1, c_tm1 = inputs
        
        go = _Multiply_and_dSigmoid(a=gh, b=self.tanh_c, 
                                    y=self.o, out=self.o)
        
        gc_tm1 = _Multiply_dtanh_and_add(o=o, gh=gh, 
                                         h=self.h, gc=gc, 
                                         out=self.c)

        gf = _Multiply_and_dSigmoid(a=gc_tm1, b=c_tm1, 
                                    y=self.f, out=self.f)
        
        gi = _Multiply_and_dSigmoid(a=gc_tm1, b=self.c_tilde, 
                                    y=self.i, out=self.i)
        
        
        gc_tilde = _Multiply_and_dSigmoid(a=gc_tm1, b=i, 
                                    y=self.c_tilde, out=self.c_tilde)
        
        gc_tm1 *= f
        
        # compute gradient with respect to the hidden input state
        gh_tm1 = np.dot(go, self.Uo, out=self.h)
        gh_tm1 += np.dot(gf, self.Uf)
        gh_tm1 += np.dot(gi, self.Ui)
        gh_tm1 += np.dot(gc_tilde, self.Uc)
        
        # compute gradient with respect to the input x
        gx = np.empty_like(x)
        gx += np.dot(go, self.Wo, out=gx)
        gx += np.dot(gf, self.Wf)
        gx += np.dot(gi, self.Wi)
        gx += np.dot(gc_tilde, self.Wc)
        
         # compute gradients of weight matrices
        self.gWo += go.T.dot(x)
        self.gWf += gf.T.dot(x)
        self.gWi += gi.T.dot(x)
        self.gWc += gc.T.dot(x)
        
        self.gUo += go.T.dot(x)
        self.gUf += gf.T.dot(x)
        self.gUi += gi.T.dot(x)
        self.gUc += gc.T.dot(x)
        
        if self.bias:
            gb_ones = np.ones((1,N),dtype=np.float32)
            self.gbo += np.dot(gb_ones, gbo)
            self.gbf += np.dot(gb_ones, gbf)
            self.gbi += np.dot(gb_ones, gbi)
            self.gbc += np.dot(gb_ones, gbc)
        
        return gx, gh_tm1
