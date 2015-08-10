import numpy as np
from chainerLayers.cukernels.utils import Get_bdim_and_gdim1D, Get_bdim_and_gdim2D
from chainerLayers.cukernels.utils import Get_bdim_and_gdimRowVec, Get_bdim_and_gdimSmallNBigM
from chainer import cuda

try:
    from pycuda.compiler import SourceModule

    Hadamard_code = SourceModule("""
        __global__
        void Hadamard(float *A, float *B, float *out, float scalar, const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
             
            if (i < N && j < M){
                int k = i+i*(M-1)+j;
                out[k] = scalar*A[k]*B[k];
            }
        }
        """)
    Hadamard_kernel = Hadamard_code.get_function("Hadamard")
    Hadamard_kernel.prepare("PPPfii")
    
    #Get_by_index2d and Clip
    IndexAndClipAndLog_code = SourceModule("""
    #include <math.h>
    __global__
    void IndexAndClipAndLog(float* probs, int *t, float *out, const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
             
            if (i < N){
            out[i] = log(fmax(1.0E-8, fmin( 1.0, probs[i+i*(M-1)+t[i]] ) ));
            }
        }
        """)
    IndexAndClipAndLog_kernel = IndexAndClipAndLog_code.get_function("IndexAndClipAndLog")
    IndexAndClipAndLog_kernel.prepare("PPPii")

    #Derivative of softmax
    dSoftmax_code = SourceModule("""
        __global__
        void dSoftmax(float* gx, int *t, float *gloss, const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
            
            if (i < N && j < M){
            float scale = gloss[0]/N;
            int k = i+i*(M-1)+j;
            gx[k] = (j == t[i]) ? (gx[k]-1)*scale : gx[k]*scale;
            }
        }
        """)
    dSoftmax_kernel = dSoftmax_code.get_function("dSoftmax")
    dSoftmax_kernel.prepare("PPPii")
    
    dropout_code = SourceModule("""
        __global__
        void dropout(float* out, float *y, float *mask, float scale, float dropout_ratio, 
                          const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
             
            if (i < N && j < M){
                int k = i+i*(M-1)+j;
                out[k] = mask[k] < dropout_ratio ? 0 : scale * x[k]
                mask[k] = scale * mask[k]
            }
        }
        """)
    dropout_kernel = dropout_code.get_function("dropout")
    dropout_kernel.prepare("PPPffii")
    
except:
    pass


def hadamard(A, B, out, scalar=np.float32(1.0)):
    """
    This kernel computes the elementwise product (hadamard product) of A and B scaled by scalar
    """
    N, M = A.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)   
    Hadamard_kernel.prepared_call(gdim, bdim,
                                  A.gpudata, B.gpudata, out.gpudata, 
                                  np.float32(scalar),
                                    np.int32(N), np.int32(M))
    return out

def getByIndex_LogAndClip(probs, t, out=None):
    """
    This kernel takes an element in each row of probs at indices t, 
            and clips the output from 1e-8 to 1 and takes the log
    """
    N, M = probs.shape
    bdim, gdim = Get_bdim_and_gdim1D(N)
    if out is None:
        out = cuda.empty((N,1),dtype=np.float32)
             
    IndexAndClipAndLog_kernel.prepared_call(gdim, bdim,
                                probs.gpudata, t.gpudata, out.gpudata,
                                np.int32(N), np.int32(M))
    return out

def dSoftmaxCrossEntropy(gx, t, gloss):

    N, M = gx.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)
    dSoftmax_kernel.prepared_call(gdim, bdim,
                                gx.gpudata, t.gpudata, gloss.gpudata,
                                np.int32(N), np.int32(M))
        
    return gx
        
def dropout(x, out, mask, scale, dropout_ratio):
    N, M = x.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)   
    dropout_kernel.prepared_call(gdim, bdim,
                                        x.gpudata, out.gpudata, mask.gpudata,
                                        scale, dropout_ratio,
                                    np.int32(N), np.int32(M))
    return out, mask
