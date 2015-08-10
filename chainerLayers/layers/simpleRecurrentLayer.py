import numpy as np
from chainer import cuda
from chainer import function

import chainerLayers.cukernels as cuk
from chainerLayers.layers import utils

    
class SimpleRecurrentLayer(function.Function):
    """
    Recurrent layer
    This function is used to compute
    y = f(x.dot(W.T)+h_tm1.dot(V.T)+b)
    where x and h are input matrices and W, V and b are parameters
    and h has the same dimensions as y.
    f is a non-linear elementwise activation function
    
    In:
        int in_size: number of columns in the input matrix;
        int out_size: number of columns in the output matrix (e.g. number of hidden states)
        str act_fun: activation function for the layer (default='tanh')
                    available activation functions: ('tanh', 'sigmoid', 'relu', 'leakyrelu')
        float Wscale: scale of the initialized weight matrix W (default=1.0)
        float Vscale: scale of the initialized weight matrix V (default=1.0)
        bool nobias: if true the layer will have a bias parameter vector b (default=True)
	float bias: filler for the bias
        bool initV2identity: if true the weight matrix V will be initialized as an identity matrix (default=False)
        bool hot: if true we assumes that the input matrix x is K-hot encoded. Hence, the function should
                    be feed hot indices instead of a full matrix (default=False).
        Note that the function is non-differentiable with respect to its input if hot is True
    """
    def __init__(self, in_size, out_size, act_func='tanh', 
                 Wscale=1.0, Vscale=1.0, 
                 nobias=False, bias=0.0,
                 initV2identity=False,
                 hot=False):
   
        self.bias = bias
        self.nobias=False
        self.in_size = in_size
        self.out_size = out_size
        self.initV2identity = initV2identity
        self.hot = hot
        self.act_func_str = act_func.lower()
        
        #initialize W weight matrix
        self.W = utils.weight_initialization(in_size, out_size, Wscale)
        self.gW = np.empty_like(self.W)
        
        #initialize V square weight matrix
        if initV2identity:
            self.V = np.eye(out_size, dtype=np.float32)*Vscale
        else:
            self.V = utils.weight_initialization(out_size, out_size, Vscale)
        self.gV = np.empty_like(self.V)
                      
        if not self.nobias:
            self.b = np.empty((1, out_size), dtype=np.float32)
            self.b.fill(self.bias)
            self.gb = np.empty_like(self.b)

        available_act_funcs = {
                    'sigmoid': (utils.Sigmoid, utils.dSigmoid),
                    'tanh': (utils.Tanh, utils.dTanh),
                    'relu': (utils.ReLU, utils.dReLU),
                    'leakyrelu': (utils.LeakyReLU, utils.dLeakyReLU),
                    }

        available_cu_act_funcs = {
            'sigmoid': (cuk.sigmoid, cuk.dsigmoid),
            'tanh': (cuk.tanh, cuk.dtanh),
            'relu': (cuk.reLU, cuk.dreLU),
            'leakyrelu': (cuk.leakyReLU, cuk.dleakyReLU),
        }
        
        self.act_func = available_act_funcs[self.act_func_str][0]
        self.dact_func = available_act_funcs[self.act_func_str][1]

        self.cu_act_func = available_cu_act_funcs[self.act_func_str][0]
        self.cu_dact_func = available_cu_act_funcs[self.act_func_str][1]
        
    @property
    def parameter_names(self):
        if self.bias:
            return 'W', 'V', 'b'
        else:
            return 'W', 'V',

    @property
    def gradient_names(self):
        if self.bias:
            return 'gW', 'gV', 'gb'
        else:
            return 'gW', 'gV',

    def forward_cpu(self, inputs):
        x, h_tm1 = inputs
        z = np.empty_like(h_tm1)
        
        #Linear function
        if self.hot: # if x is hot indices (k-hot-encoding)
            utils.HotDot(self.W, x, out=z, dont_add=True)
        else:
            z = np.dot(x, self.W.T, out=z)
        z += np.dot(h_tm1, self.V.T)
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
        x, h_tm1 = inputs
        z = cuda.empty_like(h_tm1)
        
        #Linear function
        if self.hot:
            cuk.hotdot(self.W, x, out=z, dont_add=True)
        else:            
            cuk.dot(x, self.W, out=z, transb='t')
        cuk.dotAdd(h_tm1, self.V, C=z, transb='t')
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
        x, h_tm1 = inputs
        
        gz = np.empty_like(gh)
        if self.act_func_str in ('tanh', 'sigmoid'):
            #backpropagate non-linearities
            gz = self.dact_func(gy=gh, y=self.h, out=gz)
            # compute gradient with respect to the hidden input state
            gh_tm1 = np.dot(gz, self.V, out=self.h) 
        elif self.act_func_str in ('leakyrelu', 'relu'):
            #backpropagate non-linearities
            gz = self.dact_func(x=self.z, gy=gh, out=gz)
            # compute gradient with respect to the hidden input state
            gh_tm1 = np.dot(gz, self.V, out=self.z)
        else:
            raise NotImplementedError('the activation function is not available')
            
        #backpropagate linear function
        if self.hot:
            gx = None
            utils.DotHot(gz, x, out=self.gW)
        else:
            gx = np.dot(gz, self.W)
            self.gW += gz.T.dot(x)
        self.gV += gz.T.dot(h_tm1)
        if not self.nobias:
            N = x.shape[0]
            gb_ones = np.ones((1,N),dtype=np.float32)
            self.gb += np.dot(gb_ones, gz)
        
        return gx, gh_tm1

    def backward_gpu(self, inputs, grad_outputs):
        gh = grad_outputs[0]
        x, h_tm1 = inputs
        N = x.shape[0]

        gz = cuda.empty_like(gh)
        if self.act_func_str in ('tanh', 'sigmoid'):
            #backpropagate non-linearities
            gz = self.cu_dact_func(gy=gh, y=self.h, out=gz)
            # compute gradient with respect to the hidden input state
            gh_tm1 = cuk.dot(gz, self.V, out=self.h) 
        elif self.act_func_str in ('leakyrelu', 'relu'):
            #backpropagate non-linearities
            gz = self.cu_dact_func(x=self.z, gy=gh, out=gz)
            # compute gradient with respect to the hidden input state
            gh_tm1 = cuk.dot(gz, self.V, out=self.z)
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
        cuk.dotAdd(gz, h_tm1, C=self.gV, transa='t')
        if not self.nobias:
            gb_ones = cuda.ones((1,N),dtype=np.float32)
            cuk.dotAdd(gb_ones, gz, C=self.gb)
        
        return gx, gh_tm1
