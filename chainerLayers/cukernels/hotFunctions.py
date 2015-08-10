import numpy as np
from chainerLayers.cukernels.utils import Get_bdim_and_gdim1D, Get_bdim_and_gdim2D
from chainerLayers.cukernels.utils import Get_bdim_and_gdimRowVec, Get_bdim_and_gdimSmallNBigM

try:
    from pycuda.compiler import SourceModule
    from chainer import cuda
    
    HotDot1_code = SourceModule("""
        __global__
        void HotDot1(float* a, float* out, int* indices, 
                     int K, int N, int H, int D, int B)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
            
            if (i < N && j < H){
                int n = i+i*(H-1)+j;
                if(B){
                    out[n] = 0;
                }
                for (int k=0;k<K;k++){
                    int idx = indices[i+i*(K-1)+k];
                    out[n] += a[j+j*(D-1)+idx];
                }
                
            }
        }
        """)
    HotDot1_kernel = HotDot1_code.get_function("HotDot1")
    HotDot1_kernel.prepare("PPPiiiii")
    
    HotDot2_code = SourceModule("""
        __global__
        void HotDot2(float* a, float* out, int* indices, 
                     int N, int H, int D, int B)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
            
            
            if (i < N && j < H){
                int n = i+i*(H-1)+j;
                int idx = indices[i];
                if (B){
                    out[n] = a[j+j*(D-1)+idx];
                }else{
                    out[n] += a[j+j*(D-1)+idx];
                }
            }
        }
        """)
    HotDot2_kernel = HotDot2_code.get_function("HotDot2")
    HotDot2_kernel.prepare("PPPiiii")
    
    DotHot1_code = SourceModule("""
        __global__
        void DotHot1(float* a, float* out, int* indices, 
                     int K, int N, int H, int D, int B)
        {   
            
            int j = threadIdx.x + blockIdx.x * blockDim.x;
            
            if (j < H){
                
                for (int i=0;i<N;i++){
                    for (int k=0;k<K;k++){
                        int idx = indices[i+i*(K-1)+k];
                        out[j+j*(D-1)+idx] += a[i+i*(H-1)+j];
                        }
                    }
            }
        }
        """)
        
    DotHot1_kernel = DotHot1_code.get_function("DotHot1")
    DotHot1_kernel.prepare("PPPiiiii")
        
    DotHot2_code = SourceModule("""
        __global__
        void DotHot2(float* a, float* out, int* indices, 
                     int N, int H, int D, int B)
        {   
            int j = threadIdx.x + blockIdx.x * blockDim.x;
            
            if (j < H){
                
                for (int i=0;i<N;i++){
                    int idx = indices[i];
                    out[j+j*(D-1)+idx] += a[i+i*(H-1)+j];
                    }
            }
        }
        """)

    DotHot2_kernel = DotHot2_code.get_function("DotHot2")
    DotHot2_kernel.prepare("PPPiiii")
        
except:
    pass


def hotdot(a, indices, out=None, dont_add=False):
    """
    In:
        a: a pycuda gpuarray
        indices: hot indices a K-hot encoded matrix
    out:
        out: x.dot(a.T), where x is a K-hot encoded matrix 
    
    """
    H, D = a.shape
    N, K = indices.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(H)
    elif H >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,H)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,H)
    if dont_add:
        B = np.int32(1)
    else:
        B = np.int32(0)
        
    if out is None:
        out = cuda.empty((N,H), dtype=np.float32)
        B = np.int32(1)
    
    if K > 1:
        HotDot1_kernel.prepared_call(gdim, bdim,
                                a.gpudata, out.gpudata, indices.gpudata,
                np.int32(K), np.int32(N), np.int32(H), np.int32(D), np.int32(B))
    else:
        HotDot2_kernel.prepared_call(gdim, bdim,
                                a.gpudata, out.gpudata, indices.gpudata,
                        np.int32(N), np.int32(H), np.int32(D), np.int32(B))
        return out

def dothot(a, indices, in_size, out=None, dont_add=False):
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
        
    bdim, gdim = Get_bdim_and_gdim1D(H)
    if dont_add:
        B = np.int32(1)
    else:
        B = np.int32(0)
    
    if out is None:
        out = cuda.zeros((H,in_size), dtype=np.float32)
    
    if K > 1:
        DotHot1_kernel.prepared_call(gdim, bdim,
                            a.gpudata, out.gpudata, indices.gpudata,
            np.int32(K), np.int32(N), np.int32(H), np.int32(in_size), np.int32(B))
    else:
        DotHot2_kernel.prepared_call(gdim, bdim,
                            a.gpudata, out.gpudata, indices.gpudata,
                    np.int32(N), np.int32(H), np.int32(in_size), np.int32(B))
    return out
