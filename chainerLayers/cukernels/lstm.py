
import numpy as np
from chainerLayers.cukernels.utils import Get_bdim_and_gdimRowVec, Get_bdim_and_gdim2D, Get_bdim_and_gdimSmallNBigM

try:
    from pycuda.compiler import SourceModule
    
    apply_nonlinearity_code = SourceModule("""
            #include <math.h>
            __global__
            void apply_nonlinearity(float *z, int out_size, const int N, const int M)
            {   
                int idx_i = threadIdx.x + blockIdx.x * blockDim.x;
                int idx_j = threadIdx.y + blockIdx.y * blockDim.y;
                 
                if (idx_i < N && idx_j < M){
                    int k = idx_i + idx_i*(M-1)+idx_j;
                    
                    z[k] = (idx_j < (out_size*3)) ? 1/(1+exp(-z[k])) : tanhf(z[k]);
                    
                }
            }
            """)
    apply_nonlinearity_kernel = apply_nonlinearity_code.get_function("apply_nonlinearity")
    apply_nonlinearity_kernel.prepare("Piii")


    final_mem_cell_code = SourceModule("""
                #include <math.h>
                __global__
                void final_mem_cell(float *z, float *c_tm1, float *c, float* h, 
                                    const int N, const int M)
                {   
                    int idx_i = threadIdx.x + blockIdx.x * blockDim.x;
                    int idx_j = threadIdx.y + blockIdx.y * blockDim.y;
        
        
                    if (idx_i < N && idx_j < M){
                        int k = idx_i + idx_i*(M-1)+idx_j;
                        int t = idx_i + idx_i*(M*4-1)+idx_j;
                        
                        float *i = &z[0];
                        float *f = &z[M];
                        float *o = &z[M*2];
                        float *c_tilde = &z[M*3];
                        

                        float c_k = f[t] * c_tm1[k] + i[t] * c_tilde[t];
                        c[k] = c_k;
                        h[k] = tanhf( c_k  ) * o[t];
                        
                    }
                }
                """)
    final_mem_cell_kernel = final_mem_cell_code.get_function("final_mem_cell")
    final_mem_cell_kernel.prepare("PPPPii")
                        
    
    backward_finalmem_and_nonlinearities_code = SourceModule("""
            __global__
            void backward_finalmem_and_nonlinearities(
                    float *z,
                    float *gh,
                    float *c,
                    float *c_tm1,
                    float *gc,
                    int gc_is_none,
                    int gh_is_none,
                    const int N, const int M)
            {   
                int idx_i = threadIdx.x + blockIdx.x * blockDim.x;
                int idx_j = threadIdx.y + blockIdx.y * blockDim.y;
                 
                if (idx_i < N && idx_j < M){
                    int k = idx_i + idx_i*(M-1)+idx_j;
                    int t = idx_i + idx_i*(M*4-1)+idx_j;
                    
                    float *i = &z[0];
                    float *f = &z[M];
                    float *o = &z[M*2];
                    float *c_tilde = &z[M*3];
                    
                    float gc_k = (gc_is_none) ? 0.0 : gc[k];
                    float gh_k = (gh_is_none) ? 0.0 : gh[k];
                    
                    float tanh_c_k = tanhf(c[k]);
                    float gc_tm1_k = gh_k * o[t] * (1 - tanh_c_k*tanh_c_k) + gc_k;
                    c[k]  = gc_tm1_k*f[t];
                    float gi_k = gc_tm1_k* c_tilde[t] * i[t] * (1-i[t]);
                    c_tilde[t] = gc_tm1_k* i[t] * (1-c_tilde[t]*c_tilde[t]);
                    float gf_k = gc_tm1_k* c_tm1[k] * f[t] * (1-f[t]);
                    o[t] = gh_k* tanh_c_k * o[t] * (1-o[t]);
                    i[t] = gi_k;
                    f[t] = gf_k;
                }
            }
            """)
    
    backward_finalmem_and_nonlinearities_kernel = backward_finalmem_and_nonlinearities_code.get_function(
                                                        "backward_finalmem_and_nonlinearities")
    backward_finalmem_and_nonlinearities_kernel.prepare("PPPPPiiii")
    
except:
    pass

def lstm_forward(z, c_tm1, c, h, out_size):
    
    N, M = z.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)   

    apply_nonlinearity_kernel.prepared_call(gdim, bdim,
                                 z.gpudata, np.int32(out_size),
                                np.int32(N), np.int32(M))

    N, M = h.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)
        
    final_mem_cell_kernel.prepared_call(gdim, bdim,
                    z.gpudata, c_tm1.gpudata, c.gpudata, h.gpudata, 
                                    np.int32(N), np.int32(M))
    
def lstm_backward(c, z, gh, gc, c_tm1, gc_is_none, gh_is_none):

    N, M = c.shape
    if N == 1:
        bdim, gdim = Get_bdim_and_gdimRowVec(M)
    elif M >= (N*4):
        bdim, gdim = Get_bdim_and_gdimSmallNBigM(N,M)
    else:
        bdim, gdim = Get_bdim_and_gdim2D(N,M)

    backward_finalmem_and_nonlinearities_kernel.prepared_call(gdim, bdim,
                 z.gpudata, gh.gpudata, c.gpudata, 
                 c_tm1.gpudata, gc.gpudata, np.int32(gc_is_none), np.int32(gh_is_none),
                 np.int32(N), np.int32(M))