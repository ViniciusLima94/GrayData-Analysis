#####################################################################################################
# Class to perform spectral analysis on the LFP data
#####################################################################################################
import numpy            as np
import scipy.signal     as sig
import mne.filter
import os
import multiprocessing
from   joblib           import Parallel, delayed
from   .misc            import smooth_spectra, downsample   

class spectral():

	def __init__(self, ):
		None

	def filter(self, signal = None, fs = 20, f_low = 30, f_high = 60, n_jobs = 1):
		signal_filtered = mne.filter.filter_data(signal, fs, f_low, f_high,
							 				     method = 'iir', verbose=False, n_jobs=n_jobs)

		return signal_filtered

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


	def wavelet_transform(self, signal = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0, 
		                  time_bandwidth = None, method = 'morlet', n_jobs = 1):
		if method == 'morlet':
			out = mne.time_frequency.tfr_array_morlet(signal, fs, freqs, n_cycles = n_cycles, 
				                                      output='complex', n_jobs=n_jobs)
		if method == 'multitaper':
			out = mne.time_frequency.tfr_array_multitaper(signal, fs, freqs, n_cycles = n_cycles, 
													      time_bandwidth = time_bandwidth, output='complex', 
													      n_jobs=n_jobs)
		return out

	def wavelet_spectrum(self, signal1 = None, signal2 = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0, 
						 win_time = 1, win_freq = 1, time_bandwidth = None, method = 'morlet', n_jobs = 1):

		# If signal2 is None compute the autospectra of signal1
		if type(signal2) != np.ndarray:
			if method == 'morlet':
				Wx = self.wavelet_transform(signal = signal1, fs = fs, freqs = freqs, 
					                 n_cycles = n_cycles, method = 'morlet', n_jobs = n_jobs)	

			if method == 'multitaper':
				Wx = self.wavelet_transform(signal = signal1, fs = fs, freqs = freqs, time_bandwidth = time_bandwidth,
					                 n_cycles = n_cycles, method = 'multitaper', n_jobs = n_jobs)		
			Wx = np.squeeze(Wx)
			# Computing spectra
			Sxx = Wx * np.conj(Wx)
			# Smoothing spectra
			#kernel = np.ones([win_time, win_freq])
			Sxx = smooth_spectra.smooth_spectra(Sxx.T, win_time, win_freq, fft=True).T#sig.convolve2d(Sxx.T, kernel, mode='same').T
			return Sxx
		# Otherwise compute the autospectra of signal1, signal2, and their cros-spectra
		else:
			if method == 'morlet':
				Wx = self.wavelet_transform(signal = signal1, fs = fs, freqs = freqs, 
					                 n_cycles = n_cycles, method = 'morlet', n_jobs = n_jobs)
				Wy = self.wavelet_transform(signal = signal2, fs = fs, freqs = freqs, 
					                 n_cycles = n_cycles, method = 'morlet', n_jobs = n_jobs)			
			if method == 'multitaper':
				Wx = self.wavelet_transform(signal = signal1, fs = fs, freqs = freqs, time_bandwidth = time_bandwidth,
					                 n_cycles = n_cycles, method = 'multitaper', n_jobs = n_jobs)
				Wy = self.wavelet_transform(signal = signal2, fs = fs, freqs = freqs, time_bandwidth = time_bandwidth,
					                 n_cycles = n_cycles, method = 'multitaper', n_jobs = n_jobs)	

			Wx = np.squeeze(Wx)
			Wy = np.squeeze(Wy)

			# Computing spectra
			Sxx = Wx * np.conj(Wx)
			Syy = Wy * np.conj(Wy)
			Sxy = Wx * np.conj(Wy)

			# Smoothing spectra
			#kernel = np.ones([win_time, win_freq])

			Sxx = smooth_spectra.smooth_spectra(Sxx.T, win_time, win_freq, fft=True).T#sig.convolve2d(Sxx.T, kernel, mode='same').T
			Syy = smooth_spectra.smooth_spectra(Syy.T, win_time, win_freq, fft=True).T#sig.convolve2d(Syy.T, kernel, mode='same').T
			Sxy = smooth_spectra.smooth_spectra(Sxy.T, win_time, win_freq, fft=True).T#sig.convolve2d(Sxy.T, kernel, mode='same').T
			return Sxx, Syy, Sxy

	def wavelet_coherence(self, signal1 = None, signal2 = None, fs = 20, freqs = np.arange(6,60,1), 
						  n_cycles = 7.0, win_time = 1, win_freq = 1, time_bandwidth = None, 
						  method = 'morlet', n_jobs = 1):

		# Compute auto- and cross-spectra
		Sxx, Syy, Sxy = self.wavelet_spectrum(signal1 = signal1, signal2 = signal2, fs = fs, 
			 								  freqs = freqs, n_cycles = n_cycles, 
									          win_time = win_time, win_freq = win_freq, time_bandwidth = time_bandwidth, 
									          method = method, n_jobs = n_jobs)
		# Return coherence
		return Sxy * np.conj(Sxy) / (Sxx * Syy)

class spectral_analysis(spectral):

	def __init__(self, session = None, path = None, freqs = np.arange(6,60,1), delta=1):

		self.freqs = freqs
		self.delta = delta

		if session == None:
			session = np.load(path, allow_pickle=True).item()
			self.nP      = session['info']['nP']
			self.nT      = session['info']['nT']
			self.pairs   = session['info']['pairs']
			self.tarray  = session['info']['tarray'][::delta]
			self.fsample = float(session['info']['fsample'])
			self.data    = session['data']
			self.dir     = session['path']['dir']
			self.dir_out = session['path']['dir_out']
		else:
			self.nP      = session.nP
			self.nT      = session.nT
			self.pairs   = session.pairs
			self.tarray  = session.time[0][::delta]
			self.fsample = float(session.recording_info['fsample'])
			self.data    = session.data
			self.dir     = session.dir
			self.dir_out = session.dir_out

	def _filter(self, trial = None, index_channel = None, apply_to_all = False, f_low = 30, f_high = 60, n_jobs = 1):

		if apply_to_all == True:
			signal_filtered = super().filter(signal = self.data, 
											 fs = self.fsample, f_low = f_low, f_high = f_high, n_jobs = n_jobs)

		else:
			signal_filtered = super().filter(signal = self.data[trial, index_channel, :], 
											 fs = self.fsample, f_low = f_low, f_high = f_high, n_jobs = n_jobs)

		return signal_filtered

	def _gabor_transform(self, trial = None, index_channel = None, n_cycles = 7.0):
		wt = super().gabor_transform(signal = self.data[trial, index_channel, :], 
									 fs = self.fsample, freqs = self.freqs, n_cycles = n_cycles)
		wt = downsample.downsample(wt,self.delta, axis=1)#wt[:,::self.delta]
		return wt

	def _gabor_spectrum(self, trial = None, index_channel1 = None, index_channel2 = None, 
		                win_time = 1, win_freq = 1, n_cycles = 7.0):
		if type(index_channel2) == type(None):
			Sxx = super().gabor_spectrum(signal1 = self.data[trial, index_channel1, :], signal2 = None, fs = self.fsample, 
						                 freqs = self.freqs, win_time = win_time, win_freq = win_freq, n_cycles = n_cycles)
			Sxx = Sxx[:,::self.delta]
			return Sxx

		else:
			Sxx, Syy, Sxy = super().gabor_spectrum(signal1 = self.data[trial, index_channel1, :], 
											       signal2 = self.data[trial, index_channel2, :], 
											       fs = self.fsample, freqs = self.freqs, win_time = win_time, 
											       win_freq = win_freq, n_cycles = n_cycles)

			Sxx = downsample.downsample(np.squeeze(Sxx),self.delta,axis=1)#Sxx[:,::self.delta]
			Syy = downsample.downsample(np.squeeze(Syy),self.delta,axis=1)#Syy[:,::self.delta]
			Sxy = downsample.downsample(np.squeeze(Sxy),self.delta,axis=1)#Sxy[:,::self.delta]
			return Sxx, Syy, Sxy	

	def _gabor_coherence(self, trial = None, index_pair = None, 
		                 win_time = 1, win_freq = 1, n_cycles = 7.0, save_to_file=False):
		index_channel1 = self.pairs[index_pair, 0]
		index_channel2 = self.pairs[index_pair, 1]
		coh = super().gabor_coherence(signal1 = self.data[trial, index_channel1, :], 
									  signal2 = self.data[trial, index_channel2, :], 
									  fs = self.fsample, freqs = self.freqs,  
		                 			  win_time = win_time, win_freq = win_freq, n_cycles = n_cycles)
		coh = downsample.downsample(coh, self.delta, axis=1)#coh[:,::self.delta]
		if save_to_file == False:
			return coh
		else:
			#file_name = os.path.join( self.dir_out, 
			#				          'trial_' +str(trial) + '_pair_' + str(self.pairs[index_pair, 0]) + '_' + str(self.pairs[index_pair, 1]) + '.npy')
			file_name = os.path.join( self.dir_out, 
				'trial_' +str(trial) + '_pair_' + str(int(index_pair)) + '.npy')
			np.save(file_name, {'coherence' : coh, 'freqs': self.freqs, 'time': self.tarray})

	def _wavelet_transform(self, trial = None, index_channel = None, n_cycles = 7.0, 
		           		   time_bandwidth = None, method = 'morlet', n_jobs = 1):
		if method == 'morlet':
			out = super().wavelet_transform(signal = self.data[trial, index_channel, :][np.newaxis, np.newaxis, :], 
											fs = self.fsample, freqs = self.freqs, n_cycles = n_cycles,
											time_bandwidth = None, method = 'morlet', n_jobs = n_jobs)
		if method == 'multitaper':
			out = super().wavelet_transform(signal = self.data[trial, index_channel, :][np.newaxis, np.newaxis, :], 
											fs = self.fsample, freqs = self.freqs, n_cycles = n_cycles,
											time_bandwidth = time_bandwidth, method = 'multitaper', n_jobs = n_jobs)
		# Resize time axis
		out = downsample.downsample(np.squeeze(out), self.delta, axis=1)#np.squeeze(out)[:,::self.delta]
		return out

	def _wavelet_spectrum(self, trial = None, index_channel1 = None, index_channel2 = None, 
						  n_cycles = 7.0, win_time = 1, win_freq = 1, time_bandwidth = None, method = 'morlet', n_jobs = 1):
		if type(index_channel2) == type(None):
			if method == 'morlet':
				#print(self.data[trial, index_channel1, :][np.newaxis, np.newaxis, :].shape)
				Sxx = super().wavelet_spectrum(signal1 = self.data[trial, index_channel1, :][np.newaxis, np.newaxis, :], 
											   signal2 = None, fs = self.fsample, freqs = self.freqs, n_cycles = n_cycles, 
							 				   win_time = win_time, win_freq = win_freq, time_bandwidth = None, 
							 				   method = 'morlet', n_jobs = n_jobs)

			if method == 'multitaper':
				Sxx = super().wavelet_spectrum(signal1 = self.data[trial, index_channel1, :][np.newaxis, np.newaxis, :], 
											   signal2 = None, fs = self.fsample, freqs = self.freqs, n_cycles = n_cycles, 
							 				   win_time = win_time, win_freq = win_freq, time_bandwidth = time_bandwidth, 
							 				   method = 'multitaper', n_jobs = n_jobs)
			# Resize time axis
			Sxx = downsample.downsample(np.squeeze(Sxx), self.delta, axis=1)#np.squeeze(Sxx)[:,::self.delta]
			return Sxx
		else:
			if method == 'morlet':
				Sxx, Syy, Sxy = super().wavelet_spectrum(signal1 = self.data[trial, index_channel1, :][np.newaxis, np.newaxis, :], 
														 signal2 = self.data[trial, index_channel2, :][np.newaxis, np.newaxis, :],
														 fs = self.fsample, freqs = self.freqs, n_cycles = n_cycles, 
							 							 win_time = win_time, win_freq = win_freq, time_bandwidth = None, 
							 							 method = 'morlet', n_jobs = n_jobs)

			if method == 'multitaper':
				Sxx, Syy, Sxy = super().wavelet_spectrum(signal1 = self.data[trial, index_channel1, :][np.newaxis, np.newaxis, :], 
														 signal2 = self.data[trial, index_channel2, :][np.newaxis, np.newaxis, :],
														 fs = self.fsample, freqs = self.freqs, n_cycles = n_cycles, 
							 							 win_time = win_time, win_freq = win_freq,  time_bandwidth = time_bandwidth, 
							 							 method = 'multitaper', n_jobs = n_jobs)
			# Resize time axis
			Sxx = downsample.downsample(np.squeeze(Sxx),self.delta,axis=1)#np.squeeze(Sxx)[:,::self.delta]
			Syy = downsample.downsample(np.squeeze(Syy),self.delta,axis=1)#np.squeeze(Syy)[:,::self.delta]
			Sxy = downsample.downsample(np.squeeze(Sxy),self.delta,axis=1)#np.squeeze(Sxy)[:,::self.delta]
			return Sxx, Syy, Sxy			

	def _wavelet_coherence(self, trial = None, index_pair = None, n_cycles = 7.0, win_time = 1, win_freq = 1,
		           time_bandwidth = None, method = 'morlet', save_to_file = False, n_jobs = 1):
		print('index pair =' + str(index_pair))
		index_channel1 = self.pairs[index_pair, 0]
		index_channel2 = self.pairs[index_pair, 1]
		out = super().wavelet_coherence(signal1 = self.data[trial, index_channel1, :][np.newaxis, np.newaxis, :], 
								signal2 = self.data[trial, index_channel2, :][np.newaxis, np.newaxis, :], 
								fs = self.fsample, freqs = self.freqs, n_cycles = n_cycles, win_time = win_time, win_freq = win_freq, 
								time_bandwidth = None, method = method, n_jobs = n_jobs)
		# Resize time axis
		out = downsample.downsample(np.squeeze(out), self.delta, axis=1)#[:,::self.delta]

		if save_to_file == False:
			return out
		else:
			#file_name = os.path.join( self.dir_out, 
			#				          'trial_' +str(trial) + '_pair_' + str(self.pairs[index_pair, 0]) + '_' + str(self.pairs[index_pair, 1]) + '.npy')
			file_name = os.path.join( self.dir_out, 
				'trial_' +str(trial) + '_pair_' + str(int(index_pair)) + '.npy')
			np.save(file_name, {'coherence' : out, 'freqs': self.freqs, 'time': self.tarray})

	def parallel_wavelet_coherence(self, n_cycles = 7.0, win_time = 1, win_freq = 1,
		           	               time_bandwidth = None, method = 'morlet', backend=None, n_jobs=1):
		if method == 'morlet':
			for trial in range(self.nT):
				Parallel(n_jobs=n_jobs,  prefer="threads", backend=backend, timeout=1e6)(
				     delayed(self._wavelet_coherence)
				     (trial = trial, index_pair = index_pair, n_cycles = n_cycles, 
				      win_time = win_time, win_freq = win_freq, 
				      method = 'morlet', save_to_file = True, n_jobs = 1)
				     for index_pair in range(self.nP)
					 )
		elif method == 'multitaper':
			for trial in range(self.nT):
				Parallel(n_jobs=n_jobs,  prefer="threads", backend=backend, timeout=1e6)(
				     delayed(self._wavelet_coherence)
				     (trial = trial, index_pair = index_pair, n_cycles = n_cycles, 
				      win_time = win_time, win_freq = win_freq, time_bandwidth = time_bandwidth,
				      method = 'multitaper', save_to_file = True, n_jobs = 1)
				     for index_pair in range(self.nP)
					 )
		elif method == 'gabor':
			for trial in range(self.nT):
				Parallel(n_jobs=n_jobs,  prefer="threads", backend=backend, timeout=1e6)(
				     delayed(self._gabor_coherence)
				     (trial = trial, index_pair = index_pair, n_cycles = n_cycles, 
				      win_time = win_time, win_freq = win_freq, 
				      save_to_file = True)
				     for index_pair in range(self.nP)
					 )			

