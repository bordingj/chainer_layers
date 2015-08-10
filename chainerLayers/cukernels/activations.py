import numpy as np
from chainerLayers.cukernels.utils import Get_bdim_and_gdim2D
from chainerLayers.cukernels.utils import Get_bdim_and_gdimRowVec, Get_bdim_and_gdimSmallNBigM

try:
    from pycuda.compiler import SourceModule
    
    # ReLU
    ReLU_code = SourceModule("""
        __global__
        void ReLU(float* x, float *y, const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
             
            if (i < N && j < M){
                int k = i+i*(M-1)+j;
                y[k] = fmaxf(0.000001f, x[k]);
            }
        }
        """)
    ReLU_kernel = ReLU_code.get_function("ReLU")
    ReLU_kernel.prepare("PPii")
    
    dReLU_code = SourceModule("""
        __global__
        void dReLU(float* x, float *gy, float *out, const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
             
            if (i < N && j < M){
                int k = i+i*(M-1)+j;
                out[k] = (x[k] >= 0.000001f) ? gy[k] : 0.000001f*gy[k];
            }
        }
        """)
    dReLU_kernel = dReLU_code.get_function("dReLU")
    dReLU_kernel.prepare("PPPii")
    
    # Leaky ReLU
    LeakyReLU_code = SourceModule("""
        __global__
        void LeakyReLU(float* x, float *y, float alpha, const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
             
            if (i < N && j < M){
                int k = i+i*(M-1)+j;
                y[k] = (x[k] >= 0.000001f) ? x[k] : alpha*x[k];
            }
        }
        """)
    LeakyReLU_kernel = LeakyReLU_code.get_function("LeakyReLU")
    LeakyReLU_kernel.prepare("PPfii")
    
    dLeakyReLU_code = SourceModule("""
        __global__
        void dLeakyReLU(float* x, float *gy, float *out, float alpha, const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
             
            if (i < N && j < M){
                int k = i+i*(M-1)+j;
                out[k] = (x[k] >= 0.000001f) ? gy[k] : alpha*gy[k];
            }
        }
        """)
    dLeakyReLU_kernel = dLeakyReLU_code.get_function("dLeakyReLU")
    dLeakyReLU_kernel.prepare("PPPfii")
    
    sigmoid_code = SourceModule("""
        __global__
        void Sigmoid(float *x, float *y, const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
             
            if (i < N && j < M){
                int k = i+i*(M-1)+j;
                y[k] = 1/(1+exp(-x[k]));
            }
        }
        """)
    sigmoid_kernel = sigmoid_code.get_function("Sigmoid")
    sigmoid_kernel.prepare("PPii")

    dsigmoid_code = SourceModule("""
        __global__
        void dsigmoid(float *gy, float *y, float *out, const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
             
            if (i < N && j < M){
                int k = i+i*(M-1)+j;
                out[k] = y[k]*(1-y[k])*gy[k];
            }
        }
        """)
    dsigmoid_kernel = dsigmoid_code.get_function("dsigmoid")
    dsigmoid_kernel.prepare("PPPii")
    
    tanh_code = SourceModule("""
        __global__
        void MyTanh(float *x, float *y, const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
             
            if (i < N && j < M){
                int k = i+i*(M-1)+j;
                y[k] = tanhf(x[k]);
            }
        }
        """)
    tanh_kernel = tanh_code.get_function("MyTanh")
    tanh_kernel.prepare("PPii")

    dtanh_code = SourceModule("""
        __global__
        void dtanh(float *gy, float *y, float *out, const int N, const int M)
        {   
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
             
            if (i < N && j < M){
                int k = i+i*(M-1)+j;
                out[k] = (1-y[k]*y[k])*gy[k];
            }
        }
        """)
    dtanh_kernel = dtanh_code.get_function("dtanh")
    dtanh_kernel.prepare("PPPii")
     
except:
    pass


def reLU(x, out):
    """
    This kernel is the rectifier unit max(x, 1e-6)
    """
    N, M = x.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)   
    ReLU_kernel.prepared_call(gdim, bdim,
                                        x.gpudata, out.gpudata,
                                    np.int32(N), np.int32(M))
    return out
     
     
def dreLU(x, gy, out):
    """
    In:
        gy: gradient of the output y
        x: input x
    This kernel is the hadamard product of gy and the derivative of the rectifier unit 
    with respect to its input x
    """
    N, M = gy.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)   
    dReLU_kernel.prepared_call(gdim, bdim,
                                        x.gpudata, gy.gpudata, out.gpudata,
                                    np.int32(N), np.int32(M))
    return out

def leakyReLU(x, out, alpha=0.1):
    """
    This kernel is the leaky rectifier unit
    """
    N, M = x.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)   
    LeakyReLU_kernel.prepared_call(gdim, bdim,
                                        x.gpudata, out.gpudata, np.float32(alpha),
                                    np.int32(N), np.int32(M))
    return out

def dleakyReLU(x, gy, out, alpha=0.1):
    """
    In:
        gy: gradient of the output y
        x: input x
    This kernel is the hadamard product of gy and the derivative of the leaky rectifier unit 
    with respect to its input x
    """
    N, M = gy.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)   
    dLeakyReLU_kernel.prepared_call(gdim, bdim,
                                x.gpudata, gy.gpudata, out.gpudata, np.float32(alpha),
                                    np.int32(N), np.int32(M))
    return out

def sigmoid(x, out):
    """
    This kernel is the logistic sigmoid function 1/(1+exp(-x))
    """
    N, M = x.shape
    bdim, gdim = Get_bdim_and_gdim2D(N,M)
    sigmoid_kernel.prepared_call(gdim, bdim,
                                        x.gpudata, out.gpudata,
                                    np.int32(N), np.int32(M))
    return out

def dsigmoid(gy, y, out):
    """
    In:
        gy: gradient of the output y
    This kernel is the hadamard product of gy and the derivative of the sigmoid function 
    with respect to its input. 
    """
    N, M = gy.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)
    dsigmoid_kernel.prepared_call(gdim, bdim,
                                        gy.gpudata, y.gpudata, out.gpudata,
                                    np.int32(N), np.int32(M))
    return out

def tanh(x, out):
    """
    This kernel is the tanh function
    """
    N, M = x.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)   
    tanh_kernel.prepared_call(gdim, bdim,
                                        x.gpudata, out.gpudata,
                                    np.int32(N), np.int32(M))
    return out

def dtanh(gy, y, out):
    """
    In:
        gy: gradient of the output y
    This kernel is the hadamard product of gy and the derivative of the tanh function 
    with respect to its input. 
    """
    N, M = gy.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)   
    dtanh_kernel.prepared_call(gdim, bdim,
                                        gy.gpudata, y.gpudata, out.gpudata,
                                    np.int32(N), np.int32(M))
    return out
