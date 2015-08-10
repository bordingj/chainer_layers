import math
from numba import jit

@jit
def Get_bdim_and_gdim1D(N):
    """
    returns 1d block and 1d grid dimensions for pycuda sourcemodule kernel call 
    """
    k = math.ceil(N/32.0)
    blocksize = min(256,k*32)
    bdim = (blocksize, 1, 1)
    gdim = (math.ceil(N/blocksize),1)
    return bdim, gdim

@jit
def Get_bdim_and_gdimRowVec(M):
    """
    returns 1d block and 1d grid dimensions for pycuda sourcemodule kernel call 
    """
    k = math.ceil(M/32)
    blocksize = min(256,k*32)
    bdim = (1, blocksize, 1)
    gdim = (1,math.ceil(M/blocksize))
    return bdim, gdim


@jit
def Get_bdim_and_gdimSmallNBigM(N, M):
    """
    returns 2d block and 2d grid dimensions for pycuda sourcemodule kernel call 
    """
    bdim = (4, 64, 1)
    gdim = (math.ceil(N/4.0),math.ceil(M/64.0))
    return bdim, gdim

@jit
def Get_bdim_and_gdim2D(N, M):
    """
    returns 2d block and 2d grid dimensions for pycuda sourcemodule kernel call 
    """
    bdim = (16, 16, 1)
    gdim = (math.ceil(N/16.0),math.ceil(M/16.0))
    return bdim, gdim
