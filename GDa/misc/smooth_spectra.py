import scipy.signal
import numpy as np

def smooth_spectra(spectra, win_time, win_freq, fft=False, axes = 0):
    kernel = np.ones([1, win_freq, win_time])
    if fft == True:
        return scipy.signal.fftconvolve(spectra, kernel, mode='same', axes= axes)
    else:
        return scipy.signal.convolve2d( spectra, kernel, mode='same', axes = axes) 