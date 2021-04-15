import scipy.signal
import numpy         as np

def smooth_spectra(spectra, win_time, win_freq, fft=False, axes = 0):
    r'''
    Smooth multidimensional spectra.
    > INPUTS:
    - spectra: Multidim. array containing the spectra either [freq,time,trials] or [freq,time]
    - win_time: Filter length in time axis
    - win_freq: Filter length in freq axis
    - fft: Wheter to use fft or not to convolve
    - axes: Axes to perform the convolution
    > OUTPUTS:
    - smoothed spectra
    '''
    if len(spectra.shape) == 2:
        kernel = np.ones([win_freq, win_time])/(win_freq*win_time)
    elif len(spectra.shape) == 3:
        kernel = np.ones([1, win_freq, win_time])/(win_freq*win_time)

    if fft == True:
        return scipy.signal.fftconvolve(spectra, kernel, mode='same', axes= axes)
    else:
        return scipy.signal.convolve(spectra, kernel, mode='same') 
