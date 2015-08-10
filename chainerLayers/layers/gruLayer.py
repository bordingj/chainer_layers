import numpy as np
from chainer import cuda, function
import chainerLayers.cukernels as cuk
from chainerLayers.layers import utils
            
class GRULayer(function.Function):
    """
    Gated Recurrent Units layer
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
        self.Wu = utils.weight_initialization(in_size, out_size, Wscale)
        self.gWu = np.empty_like(self.Wu)
        
        self.Vu = utils.weight_initialization(out_size, out_size, Vscale)
        self.gVu = np.empty_like(self.Vu)

        #initialize weight matrices that operates on the input
        self.Wr = utils.weight_initialization(in_size, out_size, Wscale)
        self.gWr = np.empty_like(self.Wr)
        
        self.Vr = utils.weight_initialization(out_size, out_size, Vscale)
        self.gVr = np.empty_like(self.Vr)
        
        #initialize weight matrices that operates on the input
        self.Wh = utils.weight_initialization(in_size, out_size, Wscale)
        self.gWh = np.empty_like(self.Wh)
        
        self.Vh = utils.weight_initialization(out_size, out_size, Vscale)
        self.gVh = np.empty_like(self.Vh)
        
                      
        if not self.nobias:
            self.bu = np.empty((1, out_size), dtype=np.float32)
            self.bu.fill(self.bias)
            self.gbu = np.empty_like(self.bu)

            self.br = np.empty((1, out_size), dtype=np.float32)
            self.br.fill(self.bias)
            self.gbr = np.empty_like(self.br)
            
            self.bh = np.empty((1, out_size), dtype=np.float32)
            self.bh.fill(self.bias)
            self.gbh = np.empty_like(self.bh)
        
    @property
    def parameter_names(self):
        if not self.nobias:
            return 'Wu', 'Vu', 'bu', 'Wr', 'Vr', 'br', 'Wh', 'Vh', 'bh'
        else:
            return 'Wu', 'Vu', 'Wr', 'Vr', 'Wh', 'Vh'

    @property
    def gradient_names(self):
        if not self.nobias:
            return 'gWu', 'gVu', 'gbu', 'gWr', 'gVr', 'gbr', 'gWh', 'gVh', 'gbh'
        else:
            return 'gWu', 'gVu', 'gWr', 'gVr', 'gWh', 'gVh'

    def forward_cpu(self, inputs):
        x, h_tm1 = inputs
        
        #update gate
        u = x.dot(self.Wu.T) + h_tm1.dot(self.Vu.T)
        #reset gate
        r = np.dot(x, self.Wr.T) + np.dot(h_tm1, self.Vr.T)
        if not self.nobias:
            u += self.bu
            r += self.br
        self.u = utils.Sigmoid(x=u, out=u)
        self.r = utils.Sigmoid(x=r, out=r)
        
        #new memory
        self.HV = h_tm1.dot(self.Vh.T)
        h_tilde = r*self.HV + x.dot(self.Wh.T)
        if not self.nobias:
            h_tilde += self.bh
        self.h_tilde = utils.Tanh(x=h_tilde, out=h_tilde)

        #hidden state
        self.h = (1-u)*h_tilde + u*h_tm1
        
        return self.h,


    def forward_gpu(self, inputs):
        x, h_tm1 = inputs
        N = x.shape[0]
        
        #update gate
        u = cuda.empty((N,self.out_size),dtype=np.float32)
        cuk.dot(x, self.Wu, out=u, transb = 't')
        cuk.dotAdd(h_tm1, self.Vu, C=u, transb='t')
        #reset gate
        r = cuda.empty((N,self.out_size),dtype=np.float32)
        cuk.dot(x, self.Wr, out=r, transb = 't')
        cuk.dotAdd(h_tm1, self.Vr, C=r, transb='t')
        if not self.nobias:
            cuk.addVec2Mat(u, self.bu)
            cuk.addVec2Mat(r, self.br)
        self.u = cuk.sigmoid(x=u, out=u)
        self.r = cuk.sigmoid(x=r, out=r)
        
        #new memory
        HV = cuda.empty((N,self.out_size),dtype=np.float32)
        self.HV = cuk.dot(h_tm1, self.Vh, out=HV, transb='t')
        h_tilde = cuda.empty((N,self.out_size),dtype=np.float32)
        h_tilde = cuk.hadamard(r, self.HV, out=h_tilde)
        cuk.dotAdd(x, self.Wh, C=h_tilde, transb='t')
        if not self.nobias:
            cuk.addVec2Mat(h_tilde, self.bh)
        self.h_tilde = cuk.tanh(x=h_tilde, out=h_tilde)

        #hidden state
        h = cuda.empty((N,self.out_size),dtype=np.float32)
        self.h = cuk.gru_forward(u=u, h_tilde=h_tilde, h_tm1=h_tm1,
                                 out=h)
        
        return self.h,


    def backward_cpu(self, inputs, grad_outputs):
        x, h_tm1 = inputs
        gh = grad_outputs[0]
        
        gu = self.h
        gu = np.add(h_tm1, -self.h_tilde, out=gu)
        gu *= gh * self.u * (1 - self.u)
        gh_tilde = np.add(1, -self.h_tilde**2, out=self.h_tilde)
        gh_tilde *= gh * ( 1 - self.u )
        
        gh_tm1 = np.multiply(gh, self.u, out=self.u)
        gr = np.multiply((1-self.r), self.HV, out=self.HV)
        ghr = np.multiply(gh_tilde, self.r, out=self.r)
        gr *= ghr
        
        gx = gu.dot(self.Wu)
        gx += gr.dot(self.Wr)
        gx += gh_tilde.dot(self.Wh)
        
        gh_tm1 += ghr.dot(self.Vh)
        gh_tm1 += gr.dot(self.Vr)
        gh_tm1 += gu.dot(self.Vu)

        self.gWu += gu.T.dot(x)
        self.gVu += gu.T.dot(h_tm1)

        self.gWr += gr.T.dot(x)
        self.gVr += gr.T.dot(h_tm1)
        
        self.gWh += gh_tilde.T.dot(x)
        self.gVh += ghr.T.dot(h_tm1)
        
        if not self.nobias:
            N = x.shape[0]
            gb_ones = np.ones((1,N),dtype=np.float32)
            self.gbu += np.dot(gb_ones, gu)
            self.gbr += np.dot(gb_ones, gr)
            self.gbh += np.dot(gb_ones, gh_tilde)  
            
        return gx, gh_tm1

    def backward_gpu(self, inputs, grad_outputs):
        x, h_tm1 = inputs
        gh = grad_outputs[0]
        
        
        gu, gh_tilde, gh_tm1, gr, ghr = cuk.gru_backward(
            gu=self.h, h_tm1=h_tm1, h_tilde=self.h_tilde,
                  gh_tilde=self.h_tilde, gh=gh, u=self.u, 
                  gh_tm1=self.u, gr=self.HV, r=self.r, 
                  HV=self.HV, ghr=self.r)
        
        gx = cuda.empty_like(x)
        cuk.dot(gu, self.Wu, out=gx)
        cuk.dotAdd(gr, self.Wr, C=gx)
        cuk.dotAdd(gh_tilde, self.Wh, C=gx)

        cuk.dotAdd(ghr, self.Vh, C=gh_tm1)
        cuk.dotAdd(gr, self.Vr, C=gh_tm1)
        cuk.dotAdd(gu, self.Vu, C=gh_tm1)
        
        cuk.dotAdd(gu, x, C=self.gWu, transa='t')
        cuk.dotAdd(gu, h_tm1, C=self.gVu, transa='t')

        cuk.dotAdd(gr, x, C=self.gWr, transa='t')
        cuk.dotAdd(gr, h_tm1, C=self.gVr, transa='t')

        cuk.dotAdd(gh_tilde, x, C=self.gWh, transa='t')
        cuk.dotAdd(ghr, h_tm1, C=self.gVh, transa='t')
        
       
        if not self.nobias:
            N = x.shape[0]
            gb_ones = cuda.ones((1,N),dtype=np.float32)
            cuk.dotAdd(gb_ones, gu, C=self.gbu)
            cuk.dotAdd(gb_ones, gr, C=self.gbr)
            cuk.dotAdd(gb_ones, gh_tilde, C=self.gbh)
    
        return gx, gh_tm1


        
        
