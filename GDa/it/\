import ctypes
import numpy             as np

# Loading shared object file
_it = ctypes.CDLL('src/libit.so')

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

def BinMutualInformation(x,y):
    if len(x) != len(y):
        raise ValueError('Inputs x and y should have the same length')
    global _it
    n_bins     = len(x)
    array_type = ctypes.c_int * n_bins
    MI         = _it.BinMutualInformation(array_type(*x), array_type(*y),ctypes.c_int(n_bins))
    return float(MI)

def BinLaggedMutualInformation(x,y,lag=0):
    if len(x) != len(y):
        raise ValueError('Inputs x and y should have the same length')
    global _it
    if lag == 0:
        return BinMutualInformation(x,y)
    else:
        if lag > 0:
            x_lagged = x[lag:]
            y_lagged = y[:-lag]
        elif lag < 0:
            x_lagged = x[:lag]
            y_lagged = y[-lag:]            
        return BinMutualInformation(x_lagged,y_lagged)
