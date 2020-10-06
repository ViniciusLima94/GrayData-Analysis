#####################################################################################################
# Class to perform spectral analysis on the LFP data
#####################################################################################################
import numpy         as np
import scipy.signal  as sig
import mne.filter
import os
from   quantities    import s, Hz
from   joblib        import Parallel, delayed
import multiprocessing

class spectral():

	def __init__(self, ):
		None

	def filter(self, signal = None, fs = 20, f_low = 30, f_high = 60, n_jobs = 1):
		signal_filtered = mne.filter.filter_data(signal, fs, f_low, f_high,
							 				     method = 'iir', verbose=False, n_jobs=n_jobs)

		return signal_filtered

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
						 smooth_window = 1, time_bandwidth = None, method = 'morlet', n_jobs = 1):

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
			kernel = np.ones([smooth_window, 1])
			Sxx = sig.convolve2d(Sxx.T, kernel, mode='same').T
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
			kernel = np.ones([smooth_window, 1])

			Sxx = sig.convolve2d(Sxx.T, kernel, mode='same').T
			Syy = sig.convolve2d(Syy.T, kernel, mode='same').T
			Sxy = sig.convolve2d(Sxy.T, kernel, mode='same').T
			return Sxx, Syy, Sxy

	def coherence(self, signal1 = None, signal2 = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0, smooth_window = 1,
				  time_bandwidth = None, method = 'morlet', n_jobs = 1):

		# Compute auto- and cross-spectra
		Sxx, Syy, Sxy = self.wavelet_spectrum(signal1 = signal1, signal2 = signal2, fs = fs, 
			 								  freqs = freqs, n_cycles = n_cycles, 
									          smooth_window = smooth_window, time_bandwidth = time_bandwidth, 
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

		#self.results = {}
		#self.results['coherence'] = {}

	def _filter(self, trial = None, index_channel = None, apply_to_all = False, f_low = 30, f_high = 60, n_jobs = 1):

		if apply_to_all == True:
			signal_filtered = super().filter(signal = self.data, 
											 fs = self.fsample, f_low = f_low, f_high = f_high, n_jobs = n_jobs)

		else:
			signal_filtered = super().filter(signal = self.data[trial, index_channel, :], 
											 fs = self.fsample, f_low = f_low, f_high = f_high, n_jobs = n_jobs)

		return signal_filtered

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
		out = np.squeeze(out)[:,::self.delta]
		return out

	def _wavelet_spectrum(self, trial = None, index_channel1 = None, index_channel2 = None, 
						  n_cycles = 7.0, smooth_window = 1, time_bandwidth = None, method = 'morlet', n_jobs = 1):
		if type(index_channel2) == type(None):
			if method == 'morlet':
				#print(self.data[trial, index_channel1, :][np.newaxis, np.newaxis, :].shape)
				Sxx = super().wavelet_spectrum(signal1 = self.data[trial, index_channel1, :][np.newaxis, np.newaxis, :], 
											   signal2 = None, fs = self.fsample, freqs = self.freqs, n_cycles = n_cycles, 
							 				   smooth_window = smooth_window, time_bandwidth = None, 
							 				   method = 'morlet', n_jobs = n_jobs)

			if method == 'multitaper':
				Sxx = super().wavelet_spectrum(signal1 = self.data[trial, index_channel1, :][np.newaxis, np.newaxis, :], 
											   signal2 = None, fs = self.fsample, freqs = self.freqs, n_cycles = n_cycles, 
							 				   smooth_window = smooth_window, time_bandwidth = time_bandwidth, 
							 				   method = 'multitaper', n_jobs = n_jobs)
			# Resize time axis
			Sxx = np.squeeze(Sxx)[:,::self.delta]
			return Sxx
		else:
			if method == 'morlet':
				Sxx, Syy, Sxy = super().wavelet_spectrum(signal1 = self.data[trial, index_channel1, :][np.newaxis, np.newaxis, :], 
														 signal2 = self.data[trial, index_channel2, :][np.newaxis, np.newaxis, :],
														 fs = self.fsample, freqs = self.freqs, n_cycles = n_cycles, 
							 							 smooth_window = smooth_window, time_bandwidth = None, 
							 							 method = 'morlet', n_jobs = n_jobs)

			if method == 'multitaper':
				Sxx, Syy, Sxy = super().wavelet_spectrum(signal1 = self.data[trial, index_channel1, :][np.newaxis, np.newaxis, :], 
														 signal2 = self.data[trial, index_channel2, :][np.newaxis, np.newaxis, :],
														 fs = self.fsample, freqs = self.freqs, n_cycles = n_cycles, 
							 							 smooth_window = smooth_window, time_bandwidth = time_bandwidth, 
							 							 method = 'multitaper', n_jobs = n_jobs)
			# Resize time axis
			Sxx = np.squeeze(Sxx)[:,::self.delta]
			Syy = np.squeeze(Syy)[:,::self.delta]
			Sxy = np.squeeze(Sxy)[:,::self.delta]
			return Sxx, Syy, Sxy			

	def _coherence(self, trial = None, index_pair = None, n_cycles = 7.0, smooth_window = 1,
		           time_bandwidth = None, method = 'morlet', save_to_file = False, n_jobs = 1):
		index_channel1 = self.pairs[index_pair, 0]
		index_channel2 = self.pairs[index_pair, 1]
		out = super().coherence(signal1 = self.data[trial, index_channel1, :][np.newaxis, np.newaxis, :], 
								signal2 = self.data[trial, index_channel2, :][np.newaxis, np.newaxis, :], 
								fs = self.fsample, freqs = self.freqs, n_cycles = n_cycles, smooth_window = smooth_window,
								time_bandwidth = None, method = method, n_jobs = n_jobs)
		# Resize time axis
		out = np.squeeze(out)[:,::self.delta]

		if save_to_file == False:
			return out
		else:
			file_name = os.path.join( self.dir_out, 
							          'trial_' +str(trial) + '_pair_' + str(self.pairs[index_pair, 0]) + '_' + str(self.pairs[index_pair, 1]) + '.npy')
			np.save(file_name, {'coherence' : coh, 'freqs': self.freqs, 'time': self.tarray})

	def session_coherence(self, n_cycles = 7.0, smooth_window = 1, time_bandwidth = None, method = 'morlet'):
		for trial in range(self.nT):
			Parallel(n_jobs=n_jobs, backend='loky', max_nbytes=1e6)(
				     delayed(self._coherence)
				     (trial = trial, index_pair = index_pair, n_cycles = n_cycles, 
				      smooth_window = smooth_window, time_bandwidth = time_bandwidth, 
				      method = method, save_to_file = True, n_jobs = 1)
				     for index_pair in range(self.nP)
					 )