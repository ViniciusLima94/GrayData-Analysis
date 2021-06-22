# Functional computation as describe in Lombardo et. al.
import numpy as np 

def lombardo(x_i, x_j, W):
    N = x_i.shape[-1] # Time-series length
    if N%W != 0:
        raise ValueError("The sliding window should be multiple of the time-series length")
    # Dimensions of the reshaped time-series
    r, c = int(N/W), W
    x_i = x_i.reshape((r,c))
    x_j = x_j.reshape((r,c))
    FC = np.mean(x_i*x_j, axis=1)
    return FC

