import numpy as np
cimport numpy as np
cimport cython
from libc.math import tanh, exp
from cython.parallel import prange
           
def ReLU(np.float32_t[:,:] x, np.float32_t[:,:] out):
    N, M = x.shape
    cdef np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = x[i,j]
            else:
                out[i,j] = np.float32(1e-6)
    return out

def dReLU(np.float32_t[:,:] x, np.float32_t[:,:] gy, np.float32_t[:,:] out):
    N, M = x.shape
    cdef np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = gy[i,j]
            else:
                out[i,j] = np.float32(1e-6)
    return out
    
def LeakyReLU(np.float32_t[:,:] x, np.float32_t[:,:] out, np.float32_t alpha=0.1):
    N, M = x.shape
    cdef np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = x[i,j]
            else:
                out[i,j] = x[i,j]*alpha
    return out

def dLeakyReLU(np.float32_t[:,:] x, np.float32_t[:,:] gy, np.float32_t[:,:] out, 
                np.float32_t alpha=0.1):
    N, M = x.shape
    cdef np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = gy[i,j]
            else:
                out[i,j] = gy[i,j]*alpha
    return out

def Tanh(np.float32_t[:,:] x, np.float32_t[:,:] out):
    N, M = x.shape
    cdef np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            out[i,j] = tanh(x[i,j])
    return out

def dTanh(np.float32_t[:,:] gy, np.float32_t[:,:] y, np.float32_t[:,:] out):
    N, M = y.shape
    cdef np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for i in range(M):
            out[i,j] = gy[i,j] * (1 - y[i,j] * y[i,j])
    return out

def Sigmoid(np.float32_t[:,:] x, np.float32_t[:,:] out):
    N, M = x.shape
    cdef np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            out[i,j] = 1/(1+exp(-x[i,j]))
    return out

    
def dSigmoid(np.float32_t[:,:] gy, np.float32_t[:,:] y, np.float32_t[:,:] out):
    N, M = y.shape
    cdef np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for i in range(M):
            out[i,j] = gy[i,j] * y[i,j] * (1 - y[i,j])
    return out

