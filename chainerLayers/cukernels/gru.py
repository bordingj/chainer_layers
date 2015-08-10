
import numpy as np
from chainerLayers.cukernels.utils import Get_bdim_and_gdimRowVec, Get_bdim_and_gdim2D, Get_bdim_and_gdimSmallNBigM

try:
    from pycuda.compiler import SourceModule
    
    hidden_state_code = SourceModule("""
            #include <math.h>
            __global__
            void hidden_state(float *u, float *h_tilde, float *h_tm1, float *out, 
                              const int N, const int M)
            {   
                int idx_i = threadIdx.x + blockIdx.x * blockDim.x;
                int idx_j = threadIdx.y + blockIdx.y * blockDim.y;
                 
                if (idx_i < N && idx_j < M){
                    int k = idx_i + idx_i*(M-1)+idx_j;
                    
                    out[k] = (1-u[k])*h_tilde[k] + u[k]*h_tm1[k];
                    
                }
            }
            """)
    hidden_state_code_kernel = hidden_state_code.get_function("hidden_state")
    hidden_state_code_kernel.prepare("PPPPii")

    backward_gru_code = SourceModule("""
            #include <math.h>
            __global__
            void backward_gru(float *gu, float *h_tm1, float *h_tilde,
                              float *gh_tilde, float *gh, float *u, 
                              float *gh_tm1, float *gr, float *r, 
                              float *HV, float *ghr,
                              const int N, const int M)
            {   
                int idx_i = threadIdx.x + blockIdx.x * blockDim.x;
                int idx_j = threadIdx.y + blockIdx.y * blockDim.y;
                 
                if (idx_i < N && idx_j < M){
                    int k = idx_i + idx_i*(M-1)+idx_j;
                    
                    
                    gu[k]       = (h_tm1[k] - h_tilde[k]) * gh[k] * u[k] * (1 - u[k]);
                    gh_tilde[k] = (1 - h_tilde[k]*h_tilde[k]) * gh[k] * (1 - u[k]);
                    gh_tm1[k]   = gh[k] * u[k];
                    gr[k]       = (1 - r[k]) * HV[k];
                    ghr[k]      = gh_tilde[k]*r[k];
                    gr[k]      *= ghr[k];
                    
                }
            }
            """)
    backward_gru_kernel = backward_gru_code.get_function("backward_gru")
    backward_gru_kernel.prepare("PPPPPPPPPPPii")
        
except:
    pass

def gru_forward(u, h_tilde, h_tm1, out):
    
    
    N, M = u.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)   

    hidden_state_code_kernel.prepared_call(gdim, bdim,
                                 u.gpudata, h_tilde.gpudata, h_tm1.gpudata, out.gpudata,
                                np.int32(N), np.int32(M))
    return out
    
def gru_backward(gu, h_tm1, h_tilde,
                  gh_tilde, gh, u, 
                  gh_tm1, gr, r, 
                  HV, ghr):

    N, M = u.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)   

    backward_gru_kernel.prepared_call(gdim, bdim,
                            gu.gpudata, h_tm1.gpudata, h_tilde.gpudata,
                            gh_tilde.gpudata, gh.gpudata, u.gpudata, 
                            gh_tm1.gpudata, gr.gpudata, r.gpudata, 
                             HV.gpudata, ghr.gpudata,
                                np.int32(N), np.int32(M))
    return gu, gh_tilde, gh_tm1, gr, ghr