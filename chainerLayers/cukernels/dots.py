import numpy as np
from chainer import cuda

try:
    handle = cuda.get_cublas_handle()
except:
    pass

def dot(A, B, out, transa='n', transb='n', 
            alpha=np.float32(1.0)):
        """
        This is just the blas-routine Sgemm:
        out = alpha*A.dot(B)
        where default alpha is 1 and default beta is 0
        """
        beta=np.float(0.0)
        if transa == 't':
            l, n = A.shape
        else:
            n, l = A.shape
        if transb == 't':
            m, k = B.shape
        else:
            k, m = B.shape
        if l != k:
            raise ValueError('objects are not aligned')
        if out.shape != (n, m) or out.dtype != A.dtype:
            raise ValueError('invalid value for c_gpu')
        cuda.cublas.cublasSgemm(handle, transb, transa, m, n, k, alpha, B.gpudata,
                np.int32(B.shape[1]), A.gpudata, np.int32(A.shape[1]), beta, 
                out.gpudata, np.int32(out.shape[1]))
	
        return out

def dotAdd(A, B, C, transa='n', transb='n', 
            alpha=np.float32(1.0), beta=np.float(1.0)):
    """
    This is just the blas-routine Sgemm:
    C = alpha*A.dot(B)+beta*C,
    where default alpha is 1 and default beta is 0
    """
    if transa == 't':
        l, n = A.shape
    else:
        n, l = A.shape
    if transb == 't':
        m, k = B.shape
    else:
        k, m = B.shape
    if l != k:
        raise ValueError('objects are not aligned')
    if C.shape != (n, m) or C.dtype != A.dtype:
        raise ValueError('invalid value for c_gpu')
    cuda.cublas.cublasSgemm(handle, transb, transa, m, n, k, alpha, B.gpudata,
                np.int32(B.shape[1]), A.gpudata, np.int32(A.shape[1]), beta, 
                C.gpudata, np.int32(C.shape[1]))
    return C