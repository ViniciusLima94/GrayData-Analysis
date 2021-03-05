import ctypes
import os
import numpy             as np
import multiprocessing
from   joblib            import Parallel, delayed

# Loading shared object file
#  _it = ctypes.CDLL('./src/libit.so')
_it = ctypes.CDLL('/home/vinicius/GrayData-Analysis/GDa/it/src/libit.so')

#####################################################################################
# Entropy from probabilities
#####################################################################################
_it.entropyfromprobabilities.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int)
_it.entropyfromprobabilities.restype  = ctypes.c_float

def entropyfromprobabilities(p):
    global _it
    n_bins     = len(p)
    array_type = ctypes.c_float * n_bins
    H          = _it.entropyfromprobabilities(array_type(*p), ctypes.c_int(n_bins))
    return float(H)

#####################################################################################
# Entropy for spike-train
#####################################################################################
_it.BinEntropy.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_int)
_it.BinEntropy.restype  = ctypes.c_float

def BinEntropy(x):
    global _it
    n_bins     = len(x)
    array_type = ctypes.c_int * n_bins
    H          = _it.BinEntropy(array_type(*x), ctypes.c_int(n_bins))
    return float(H)

#####################################################################################
# Joint entropy for spike-trains
#####################################################################################
_it.BinJointEntropy.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),ctypes.c_int)
_it.BinJointEntropy.restype  = ctypes.c_float

def BinJointEntropy(x,y):
    if len(x) != len(y):
        raise ValueError('Inputs x and y should have the same length')
    global _it
    n_bins     = len(x)
    array_type = ctypes.c_int * n_bins
    Hxy        = _it.BinJointEntropy(array_type(*x), array_type(*y),ctypes.c_int(n_bins))
    return float(Hxy)

#####################################################################################
# Mutual information for spike-trains
#####################################################################################
_it.BinMutualInformation.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),ctypes.c_int)
_it.BinMutualInformation.restype  = ctypes.c_float

def BinMutualInformation(x,y,use='MI'):
    if use not in ['MI', 'CC']:
        raise ValueError('use should be either "MI" or "CC"')
    if len(x) != len(y):
        raise ValueError('Inputs x and y should have the same length')
    global _it
    n_bins     = len(x)
    array_type = ctypes.c_int * n_bins
    if use == 'MI':
        MI     = _it.BinMutualInformation(array_type(*x), array_type(*y),ctypes.c_int(n_bins))
    elif use == 'CC':
        MI     = np.corrcoef(x, y)[0,1]
    return float(MI)

def BinLaggedMutualInformation(x,y,lag=0, use='MI'):
    if use not in ['MI', 'CC']:
        raise ValueError('use should be either "MI" or "CC"')
    if len(x) != len(y):
        raise ValueError('Inputs x and y should have the same length')
    global _it
    if lag == 0:
        return BinMutualInformation(x,y,use=use)
    else:
        if lag > 0:
            x_lagged = x[lag:]
            y_lagged = y[:-lag]
        elif lag < 0:
            x_lagged = x[:lag]
            y_lagged = y[-lag:]            
        return BinMutualInformation(x_lagged,y_lagged,use=use)

#####################################################################################
# High-order functions for high-dimensional data
#####################################################################################
def pairwiseMI(X, pairs, min_lag, max_lag, use, n_jobs):

    def MI(x, y, min_lag, max_lag, use):
        lags = np.arange(min_lag,max_lag+1, 1,dtype=int)
        aux  = [BinLaggedMutualInformation(x.astype(int),y.astype(int),lag=l,use=use) for l in lags]
        if use == 'MI':
            return np.max( aux ), lags[np.argmax(aux)]
        elif use == 'CC':
            idx = np.argmax( np.abs(aux) )
            return aux[idx], lags[idx]

    out = Parallel(n_jobs=n_jobs, backend='loky', timeout=1e6)(delayed(MI)(X[i,:],X[j,:],min_lag,max_lag,use) for i,j in pairs)
    out = np.squeeze( out ) 
    MI  = out[:,0]
    tau = out[:,1]

    return MI, tau
