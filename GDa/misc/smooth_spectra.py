import scipy.signal
import numpy as np

def smooth_spectra(spectra, win_time, win_freq, fft=False, axes = 0):
	if len(spectra.shape) == 2:
		kernel = np.ones([win_freq, win_time])/(win_freq*win_time)
	elif len(spectra.shape) == 3:
		kernel = np.ones([1, win_freq, win_time])/(win_freq*win_time)

	#print(kernel.shape)

	if fft == True:
		return scipy.signal.fftconvolve(spectra, kernel, mode='same', axes= axes)# + 1j*scipy.signal.fftconvolve(spectra.imag, kernel, mode='same', axes= axes)
	else:
		return scipy.signal.convolve(spectra, kernel, mode='same')# + 1j*scipy.signal.convolve( spectra.imag, kernel, mode='same') 
