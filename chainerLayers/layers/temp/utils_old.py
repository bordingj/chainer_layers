# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:44:50 2015

@author: bordingj
"""

import numpy as np
import numba as nb



@nb.jit([(nb.float32[:,:], nb.int32[:,:], nb.float32[:,:], nb.boolean)])
def HotDot(a, indices, out=None, dont_add=False):
    """
    In:
        a: a numpy array
        indices: hot indices a K-hot encoded matrix
    out:
        out: x.dot(a.T), where x is a K-hot encoded matrix 
    
    """
    H, D = a.shape
    N, K = indices.shape
    if out is None:
        out = np.zeros((N,H), dtype=np.float32)
    
    if dont_add:
        out[:] = 0
        
    if K > 1:
        for i in range(N):
            for k in range(K):
                idx = indices[i,k]
                for j in range(H):
                    out[i,j] += a[j,idx]
    else:
        indices = indices.ravel()
        for i in range(N):
            idx = indices[i]
            for j in range(H):
                out[i,j] += a[j,idx]     
    return out


@nb.jit([(nb.float32[:,:], nb.int32[:,:], nb.int64, nb.float32[:,:], nb.boolean)])
def DotHot(a, indices, in_size, out=None, dont_add=False):
    """
    In:
        a: a numpy array
        indices: hot indices a K-hot encoded matrix
    out:
        out: a.T.dot(x), where x is a K-hot encoded matrix 
    
    """
    N, H = a.shape
    _N, K = indices.shape
    if _N != N:
        raise ValueError( 'a.shape[0] != idx.shape[0]' )
        
    if out is None:
        out = np.zeros((H,in_size), dtype=np.float32)
        
    if dont_add:
        out[:] = 0
    if K > 1:
        for j in range(N):
            for k in range(K):
                idx = indices[j,k]
                for i in range(H):
                    out[i,idx] += a[j,i]
    else:
        indices = indices.ravel()
        for j in range(N):
            idx = indices[j]
            for i in range(H):
                out[i,idx] += a[j,i]
    return out