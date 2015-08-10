import numpy as np
from chainerLayers.cukernels.utils import Get_bdim_and_gdim1D, Get_bdim_and_gdim2D
from chainerLayers.cukernels.utils import Get_bdim_and_gdimRowVec, Get_bdim_and_gdimSmallNBigM

try:
    from pycuda.compiler import SourceModule
    
    AddVec2Mat_code = SourceModule("""
    __global__
    void AddVec2Mat(float* h, float* b, const int N, const int M)
    {   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        
        if (i < N && j < M)
            {
            h[i+i*(M-1)+j] += b[j];
            }
    }
    """)
    AddVec2Mat_kernel = AddVec2Mat_code.get_function("AddVec2Mat")
    AddVec2Mat_kernel.prepare("PPii")
    
    MatAdd_code = SourceModule("""
        __global__
        void MatAdd(float *A, float *B, float scalar, const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
            
            if (i < N && j < M){
                int k = i+i*(M-1)+j;
                A[k] += scalar*B[k];
            }
        }
        """)
    MatAdd_kernel = MatAdd_code.get_function("MatAdd")
    MatAdd_kernel.prepare("PPfii")
    
    VecAdd_code = SourceModule("""
        __global__
        void VecAdd(float *a, float *b, float scalar, const int N)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            
            if (i < N){
                a[i] += scalar*b[i];
            }
        }
        """)
    VecAdd_kernel = VecAdd_code.get_function("VecAdd")
    VecAdd_kernel.prepare("PPfi")
    
except:
    pass



def addVec2Mat(h, b):
    """
    In:
        h: a gpuarray matrix as shape NxH
        b: a gpuarray vector of shape 1xH (or Hx1)
        
    This kernel adds vector b to every row of matrix a
    """
    N, M = h.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M) 

    AddVec2Mat_kernel.prepared_call(gdim, bdim,
                                h.gpudata, b.gpudata, np.int32(N), np.int32(M))
                                
    return h

def matAdd(A, B, scalar=1.0):
    """
    This kernel adds a 2d pycuda-gpuarray B (scaled by scalar) to a another array A
    """ 
    N, M = A.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)
    MatAdd_kernel.prepared_call(gdim, bdim,
                                        A.gpudata, B.gpudata, np.float32(scalar),
                                    np.int32(N), np.int32(M))
    return A

def vecAdd(a,b,scalar=1.0):
    """
    This kernel adds a 1d pycuda-gpuarray a (scaled by scalar) to a another array b
    """
    orig_shape = a.shape
    a = a.ravel()
    N = a.shape[0]
    bdim, gdim = Get_bdim_and_gdim1D(N)
    VecAdd_kernel.prepared_call(gdim, bdim,
                                        a.gpudata, b.gpudata, np.float32(scalar),
                                    np.int32(N))
    a = a.reshape(orig_shape)
    return a
