import scipy.signal
import numpy as np

def smooth_spectra(spectra, win_time, win_freq, fft=False):
	kernel = np.ones([win_time, win_freq])	
	if fft == True:
		return scipy.signal.fftconvolve(spectra, kernel, mode='same')
	else:
		return scipy.signal.convolve2d(spectra, kernel, mode='same')
