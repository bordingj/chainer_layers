import numpy as np
from numba import jit 
import math

__all__ = ['_ReLU', '_dReLU', '_LeakyReLU', '_dLeakyReLU',
           '_Sigmoid', '_dSigmoid', '_tanh', '_dtanh']
           
           
@jit('float32[:,:](float32[:,:], float32[:,:])', nopython=True, nogil=True)
def _ReLU(x, out):
    N, M = x.shape
    for i in range(N):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = x[i,j]
            else:
                out[i,j] = np.float32(1e-6)
    return out

@jit('float32[:,:](float32[:,:], float32[:,:], float32[:,:])', nopython=True, nogil=True)
def _dReLU(x, gy, out):
    N, M = x.shape
    for i in range(N):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = gy[i,j]
            else:
                out[i,j] = np.float32(1e-6)
    return out
    
@jit('float32[:,:](float32[:,:], float32[:,:], float32)', nopython=True, nogil=True)
def _LeakyReLU(x, out, alpha=0.1):
    N, M = x.shape
    for i in range(N):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = x[i,j]
            else:
                out[i,j] = x[i,j]*alpha
    return out

@jit('float32[:,:](float32[:,:], float32[:,:], float32[:,:], float32)', nopython=True, nogil=True)
def _dLeakyReLU(x, gy, out, alpha=0.1):
    N, M = x.shape
    for i in range(N):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = gy[i,j]
            else:
                out[i,j] = gy[i,j]*alpha
    return out

@jit('float32[:,:](float32[:,:], float32[:,:])', nopython=True, nogil=True)
def _tanh(x, out):
    N, M = x.shape
    for i in range(N):
        for j in range(M):
            out[i,j] = math.tanh(x[i,j])
    return out

@jit('float32[:,:](float32[:,:], float32[:,:])', nopython=True, nogil=True)
def _Sigmoid(x, out):
    N, M = x.shape
    for i in range(N):
        for j in range(M):
            out[i,j] = 1/(1+math.exp(-x[i,j]))
    return out
    
    
@jit('float32[:,:](float32[:,:], float32[:,:], float32[:,:])', nopython=True, nogil=True)
def _dtanh(gy, y, out):
    N, M = y.shape
    for j in range(M):
        for i in range(N):
            out[i,j] = gy[i,j] * (1 - y[i,j] * y[i,j])
    return out
    
@jit('float32[:,:](float32[:,:], float32[:,:], float32[:,:])', nopython=True, nogil=True)
def _dSigmoid(gy, y, out):
    N, M = y.shape
    for j in range(M):
        for i in range(N):
            out[i,j] = gy[i,j] * y[i,j] * (1 - y[i,j])
    return out

