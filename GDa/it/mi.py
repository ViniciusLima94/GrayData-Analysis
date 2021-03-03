import ctypes
import numpy             as np

_mi = ctypes.CDLL('src/libmi.so')
_mi.st_mi.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int)
_mi.st_mi.restype  = ctypes.c_float

# Defining wrapper for C function in python
def st_mi(x, y):
    global _mi
    n_bins = len(x)
    array_type = ctypes.c_int * n_bins
    MI = _mi.st_mi(array_type(*x), array_type(*y), ctypes.c_int(n_bins))
    return float(MI)
