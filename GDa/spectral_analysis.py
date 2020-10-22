#####################################################################################################
# Class to perform spectral analysis on the LFP data
#####################################################################################################
import numpy            as     np
import mne.filter
import os
import multiprocessing
from   joblib           import Parallel, delayed
from   .misc            import smooth_spectra, downsample   

class spectral_analysis():

	def __init__(self,):
		None

	def filter(self, data = None, fs = 20, f_low = 30, f_high = 60, n_jobs = 1):

		signal_filtered = mne.filter.filter_data(data, fs, f_low, f_high,
		                                         method = 'iir', verbose=False, n_jobs=n_jobs)

		return signal_filtered

	def wavelet_transform(self, data = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0, 
		                  time_bandwidth = None, delta = 1, method = 'morlet', n_jobs = 1):
		if method == 'morlet':
			out = mne.time_frequency.tfr_array_morlet(data, fs, freqs, n_cycles = n_cycles, zero_mean=False,
				                                      output='complex', decim = delta, n_jobs=n_jobs)
		if method == 'multitaper':
			out = mne.time_frequency.tfr_array_multitaper(data, fs, freqs, n_cycles = n_cycles, zero_mean=False,
													      time_bandwidth = time_bandwidth, output='complex', 
													      decim = delta, n_jobs=n_jobs)
		return out

	def wavelet_coherence(self, data = None, pairs = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0, 
		                  time_bandwidth = None, delta = 1, method = 'morlet', win_time = 1, win_freq = 1, 
		                  dir_out = None, n_jobs = 1):

		# Data dimension
		T, C, L = data.shape
		# All possible pairs of channels

		# Computing wavelets
		W = self.wavelet_transform(data = data, fs = fs, freqs = freqs, n_cycles = n_cycles, 
		                           time_bandwidth = time_bandwidth, delta = delta, 
		                           method = method, n_jobs = -1)
		# Auto spectra
		S_auto = W * np.conj(W)

		def pairwise_coherence(trial_number, channel1, channel2, win_time, win_freq):
			#channel1, channel2 = pairs[index_pair,0], pairs[index_pair,1]
			#print(str(channel1) + ', ' + str(channel2))
			Sxy = W[trial_number, channel1, :, :] * np.conj(W[trial_number, channel2, :, :])
			Sxx = smooth_spectra.smooth_spectra(S_auto[trial_number,channel1, :, :].T, win_time, win_freq, fft=True).T
			Syy = smooth_spectra.smooth_spectra(S_auto[trial_number,channel2, :, :].T, win_time, win_freq, fft=True).T
			Sxy = smooth_spectra.smooth_spectra(Sxy.T, win_time, win_freq, fft=True).T
			coh = Sxy * np.conj(Sxy) / (Sxx * Syy)
			# Saving to file
			file_name = os.path.join( dir_out, 
				'trial_' +str(trial_number) + '_ch1_' + str(channel1) + '_ch2_' + str(channel2) +'.npy')
			#print(file_name)
			np.save(file_name, {'coherence' : np.abs(coh).astype(np.float32) })

		for trial_index in range(T):
			Parallel(n_jobs=n_jobs, backend='loky', timeout=1e6)(
	        delayed(pairwise_coherence)(trial_index, pair[0], pair[1], win_time, win_freq)
	                for pair in pairs )
	

	def gabor_transform(self, signal = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0):
		n      = len(signal)
		sigma2 = 1
		omega  = np.concatenate( (np.arange(0, n/2), np.arange(-np.ceil(n/2)+1, 0) ) ) * fs/n

		fftx   = np.fft.fft(signal)

		tolerance = 0.5

		mincenterfreq = 2*tolerance*np.sqrt(sigma2)*fs*n_cycles/n
		maxcenterfreq = fs*n_cycles/(n_cycles+tolerance/np.sqrt(sigma2))

		s_array  = n_cycles/freqs
		minscale = n_cycles/maxcenterfreq
		maxscale = n_cycles/mincenterfreq

		nscale = len(freqs)
		wt     = np.zeros([n,nscale]) * (1+1j)
		scaleindices = np.arange(0,len(s_array))[(s_array>=minscale)*(s_array<=maxscale)]
		psi_array = np.zeros([n, nscale]) * (1+1j)

		for kscale in scaleindices:
			s    = s_array[kscale]
			freq = (s*omega - n_cycles)
			Psi  = (4*np.pi*sigma2)**(1/4) * np.sqrt(s) * np.exp(-sigma2/2*freq**2)
			wt[:,kscale] = np.fft.ifft(fftx*Psi)
			psi_array[:,kscale]=np.fft.ifft(Psi)

		return wt

	def gabor_spectrum(self, signal1 = None, signal2 = None, fs = 20, freqs = np.arange(6,60,1),  
		                win_time = 1, win_freq = 1, n_cycles = 7.0):
		if type(signal2) != np.ndarray:
			wt1    = self.gabor_transform(signal=signal1,fs=fs,freqs=freqs,n_cycles=n_cycles)
			Sxx    = wt1*np.conj(wt1)
			#kernel = np.ones([win_time, win_freq])		
			return smooth_spectra.smooth_spectra(Sxx, win_time, win_freq, fft=True)#sig.convolve2d(Sxx, kernel, mode='same').T
		else:
			wt1 = self.gabor_transform(signal=signal1,fs=fs,freqs=freqs,n_cycles=n_cycles)
			wt2 = self.gabor_transform(signal=signal2,fs=fs,freqs=freqs,n_cycles=n_cycles)

			npts  = wt1.shape[0]
			nfreq = len(freqs)

			Sxy    = wt1*np.conj(wt2)
			Sxx    = wt1*np.conj(wt1)
			Syy    = wt2*np.conj(wt2)

			# Smoothing spectra
			#kernel = np.ones([win_time, win_freq])		
			Sxx = smooth_spectra.smooth_spectra(Sxx, win_time, win_freq, fft=True)#sig.convolve2d(Sxx, kernel, mode='same')
			Syy = smooth_spectra.smooth_spectra(Syy, win_time, win_freq, fft=True)#sig.convolve2d(Syy, kernel, mode='same')
			Sxy = smooth_spectra.smooth_spectra(Sxy, win_time, win_freq, fft=True)#sig.convolve2d(Sxy, kernel, mode='same')
			return Sxx.T, Syy.T, Sxy.T

	def gabor_coherence(self, signal1 = None, signal2 = None, fs = 20, freqs = np.arange(6,60,1),  
		                 win_time = 1, win_freq = 1, n_cycles = 7.0):
		Sxx, Syy, Sxy = self.gabor_spectrum(signal1 = signal1, signal2 = signal2, fs = fs, freqs = freqs,  
		                win_time = win_time, win_freq = win_freq, n_cycles = n_cycles)

		return Sxy * np.conj(Sxy) / (Sxx * Syy)