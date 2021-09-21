import numpy as np 
import numba as nb

nb.jit(nopython=True)
def custom_mean(array, axis=None):
    r'''
    Custom mean function that returns zero in case the array is a empty slice.
    > INPUTS:
    - array: Input array to compute mean from.
    - axis: axis in which to take the mean
    > OUTPUTS:
    Average of the array in the desired axis.
    '''
    if array==[]:
        return 0
    else:
        return np.nanmean(array, axis=axis)

nb.jit(nopython=True)
def custom_std(array, axis=None):
    r'''
    Custom std function that returns zero in case the array is a empty slice.
    > INPUTS:
    - array: Input array to compute mean from.
    - axis: axis in which to take the mean
    > OUTPUTS:
    Average of the array in the desired axis.
    '''
    if array==[]:
        return 0
    else:
        return np.nanstd(array, axis=axis)
