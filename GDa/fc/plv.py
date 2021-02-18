import numpy as np 
import scipy

def plv(x_i, x_j, W = None):

    N = x_i.shape[-1] # Time-series length
    h1 = scipy.signal.hilbert(x_i)
    h2 = scipy.signal.hilbert(x_j)
    theta1 = np.unwrap(np.angle(h1))
    theta2 = np.unwrap(np.angle(h2))

    if W == None:
        complex_phase_diff = np.exp(np.complex(0,1)*(theta1 - theta2))
        PLV = np.abs(np.sum(complex_phase_diff))/len(theta1)
        return PLV
    else:
        if N%W != 0:
            raise ValueError("The sliding window should be multiple of the time-series length")

        r, c = int(N/W), W
        theta1 = theta1.reshape((r,c))
        theta2 = theta2.reshape((r,c))
        complex_phase_diff = np.exp(np.complex(0,1)*(theta1 - theta2))
        PLV = np.abs(np.sum(complex_phase_diff,axis=1))/c
        return PLV
