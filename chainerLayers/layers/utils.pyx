import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport tanh, exp
from cython.parallel import prange

def weight_initialization(in_size, out_size, scale):
    return np.random.normal(0, scale * np.sqrt(1. / in_size),
            (out_size, in_size)).astype(np.float32)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lstm_apply_nonlinearity(np.float32_t[:,:] z, int out_size):
    
    cdef:
        int N = z.shape[0]
        int M = z.shape[1]
        np.intp_t cut = out_size*3
        np.intp_t i, j
        
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            if j < cut:
                z[i,j] = 1/(1+exp(-z[i,j]))
            else:
                z[i,j] = tanh(z[i,j])
    return np.array(z, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def lstm_final_mem_cell(np.float32_t[:,:] z, np.float32_t[:,:] c_tm1, 
                        np.float32_t[:,:] c, np.float32_t[:,:] h):
    cdef:
        int N = c.shape[0]
        int M = c.shape[1]
        np.intp_t i, j
        np.float32_t i_t, f_t, o_t, c_tilde_t, c_k
        
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            i_t = z[i,j]
            f_t = z[i,(j+M)]
            o_t = z[i,(j+M*2)]
            c_tilde_t = z[i,(j+M*3)]
            
            c_k = f_t*c_tm1[i,j] + i_t*c_tilde_t
            
            c[i,j] = c_k

            h[i,j] = tanh(c_k)*o_t

@cython.boundscheck(False)
@cython.wraparound(False)
def lstm_backward_finalmem_and_nonlinearities(np.float32_t[:,:] z, 
                                              np.float32_t[:,:] gh, 
                                              np.float32_t[:,:] c, 
                                              np.float32_t[:,:] c_tm1, 
                                              np.float32_t[:,:] gc, 
                                              bint gh_is_none, bint gc_is_none):
    cdef:
        int N = c.shape[0]
        int M = c.shape[1]
        np.intp_t i, j
        np.float32_t i_t, f_t, o_t, c_tilde_t, tanh_c_k, gh_k, gc_k, gc_tm1_k, gi_k, gf_k
        
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            i_t = z[i,j]
            f_t = z[i,(j+M)]
            o_t = z[i,(j+M*2)]
            c_tilde_t = z[i,(j+M*3)]
            
            tanh_c_k = tanh(c[i,j])
            
            if gh_is_none:
                gh_k = 0.0
            else:
                gh_k = gh[i,j]
            if gc_is_none:
                gc_k = 0.0
            else:
                gc_k = gc[i,j]

            gc_tm1_k = gh_k * o_t * (1 - tanh_c_k**2)+gc_k
            c[i,j] = gc_tm1_k*f_t #we use the memory for c as gc_tm1
            
            gi_k = gc_tm1_k* c_tilde_t * i_t * (1-i_t);

            z[i,(j+M*3)] = gc_tm1_k* i_t * (1-c_tilde_t*c_tilde_t);

            gf_k = gc_tm1_k* c_tm1[i,j] * f_t * (1-f_t);

            z[i,(j+M*2)] = gh_k* tanh_c_k * o_t * (1-o_t);

            z[i,j] = gi_k;

            z[i,(j+M)] = gf_k;

@cython.boundscheck(False)
@cython.wraparound(False)
def ReLU(np.float32_t[:,:] x, np.float32_t[:,:] out):
    cdef: 
        int N = x.shape[0]
        int M = x.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = x[i,j]
            else:
                out[i,j] = 1e-6
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def dReLU(np.float32_t[:,:] x, np.float32_t[:,:] gy, np.float32_t[:,:] out):
    cdef:
        int N = x.shape[0]
        int M = x.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = gy[i,j]
            else:
                out[i,j] = 1e-6
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def LeakyReLU(np.float32_t[:,:] x, np.float32_t[:,:] out, np.float32_t alpha=0.1):
    cdef: 
        int N = x.shape[0]
        int M = x.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = x[i,j]
            else:
                out[i,j] = x[i,j]*alpha
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def dLeakyReLU(np.float32_t[:,:] x, np.float32_t[:,:] gy, np.float32_t[:,:] out, 
                np.float32_t alpha=0.1):
    cdef:
        int N = x.shape[0]
        int M = x.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            if x[i,j]>1e-6:
                out[i,j] = gy[i,j]
            else:
                out[i,j] = gy[i,j]*alpha
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def Tanh(np.float32_t[:,:] x, np.float32_t[:,:] out):
    cdef:
        int N = x.shape[0]
        int M = x.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            out[i,j] = tanh(x[i,j])
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def dTanh(np.float32_t[:,:] gy, np.float32_t[:,:] y, np.float32_t[:,:] out):
    cdef:
        int N = y.shape[0]
        int M = y.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            out[i,j] = gy[i,j] * (1 - y[i,j] * y[i,j])
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def Sigmoid(np.float32_t[:,:] x, np.float32_t[:,:] out):
    cdef:
        int N = x.shape[0]
        int M = x.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            out[i,j] = 1/(1+exp(-x[i,j]))
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def dSigmoid(np.float32_t[:,:] gy, np.float32_t[:,:] y, np.float32_t[:,:] out):
    cdef:
        int N = y.shape[0]
        int M = y.shape[1]
        np.intp_t i, j
    for i in prange(N, schedule='guided', nogil=True):
        for j in range(M):
            out[i,j] = gy[i,j] * y[i,j] * (1 - y[i,j])
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def dSoftMaxInner(np.float32_t[:,:] gx, np.int32_t[:] t, int N):
    cdef np.intp_t i, k
    for i in prange(N, schedule='guided', nogil=True):
        k = <np.intp_t>t[i]
        gx[i,k] -= 1


@cython.boundscheck(False)
@cython.wraparound(False)
def HotDot(np.float32_t[:,:] a, np.int32_t[:,:] indices, np.float32_t[:,:] out, 
           dont_add=False):
    """
    In:
        a: a numpy array
        indices: hot indices a K-hot encoded matrix
    out:
        out: x.dot(a.T), where x is a K-hot encoded matrix 
    
    """
    
    cdef:
        int H = a.shape[0]
        int D = a.shape[1]
        int N = indices.shape[0]
        int K = indices.shape[1]
        cdef np.intp_t i, j, k, idx
    
    if dont_add:
        for i in prange(N, schedule='guided', nogil=True):
            for j in range(H):
                out[i,j] = 0
        
    if K > 1:
        for i in prange(N, schedule='guided', nogil=True):
            for k in range(K):
                idx = <np.intp_t>indices[i,k]
                for j in range(H):
                    out[i,j] += a[j,idx]
    else:
        for i in prange(N, schedule='guided', nogil=True):
            idx = <np.intp_t>indices[i,1]
            for j in range(H):
                out[i,j] += a[j,idx]     
    return np.array(out, np.float32, copy=False, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def DotHot(np.float32_t[:,:] a, np.int32_t[:,:] indices, 
           np.float32_t[:,:] out, dont_add=False):
    """
    In:
        a: a numpy array
        indices: hot indices a K-hot encoded matrix
    out:
        out: a.T.dot(x), where x is a K-hot encoded matrix 
    
    """
    cdef:
        int N = a.shape[0]
        int H = a.shape[1]
        int _N = a.shape[0]
        int K = a.shape[1]
        np.intp_t i, j, k, idx

    if _N != N:
        raise ValueError( 'a.shape[0] != idx.shape[0]' )


        
    if dont_add:
        M = out.shape[1]
        for i in prange(H, schedule='guided', nogil=True):
            for j in range(M):
                out[i,j] = 0
        
    if K > 1:
        for j in prange(N, schedule='guided', nogil=True):
            for k in range(K):
                idx = <np.intp_t>indices[j,k]
                for i in range(H):
                    out[i,idx] += a[j,i]
    else:
        for j in prange(N, schedule='guided', nogil=True):
            idx = <np.intp_t>indices[j,1]
            for i in range(H):
                out[i,idx] += a[j,i]
    return np.array(out, np.float32, copy=False, order='C')

