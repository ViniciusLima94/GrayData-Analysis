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

	def spectogram(self, signal = None, fs = 20, freqs = np.arange(6,60,1), method = 'morlet', n_jobs = 1):
		if method == 'morlet':
			out = mne.time_frequency.tfr_array_morlet(signal, fs, freqs, output='power', n_jobs=n_jobs)
		if method == 'multitaper':
			out = mne.time_frequency.tfr_array_multitaper(signal, fs, freqs, output='power', n_jobs=n_jobs)
		return out

	def instantaneous_power(self, signal = None, fs = 20, f_low = 30, f_high = 60, n_jobs = 1):
		# Filter the signal
		signal_filtered = self.filter(signal = signal, fs = fs, f_low = f_low, f_high = f_high, n_jobs = n_jobs)
		# Hilbert transform
		S               = sig.hilbert(signal_filtered)
		# Power
		P               = np.multiply( S, np.conj(S) )
		return P

class spectral_analysis(spectral):

	def __init__(self, session = None, path = None,):


		#self.save_filtered = save_filtered
		#self.save_morlet   = save_morlet
		#self.save_coh      = save_coh

		if session == None:
			session = np.load(path, allow_pickle=True).item()
			self.nP      = session['info']['nP']
			self.nT      = session['info']['nT']
			self.pairs   = session['info']['pairs']
			self.tarray  = session['info']['tarray']
			#self.tidx    = np.arange(self.dt, session['data'].shape[2]-self.dt, self.step)
			#self.taxs    = session['info']['tarray'][self.tidx]
			self.fsample = session['info']['fsample']
			self.data    = session['data']
			self.dir     = session['path']['dir']
			self.dir_out = session['path']['dir_out']
		else:
			self.nP      = session.nP
			self.nT      = session.nT
			self.pairs   = session.pairs
			self.tarray  = session.time[0]
			#self.tidx    = np.arange(self.dt, session.data.shape[2]-self.dt, self.step)
			#self.taxs    = session.time[0][self.tidx]
			self.fsample = session.recording_info['fsample']
			self.data    = session.data
			self.dir     = session.dir
			self.dir_out = session.dir_out

		#self.results = {}
		#self.results['coherence'] = {}

	def filter(self, trial = None, index_channel = None, apply_to_all = False, f_low = 30, f_high = 60, n_jobs = 1):

		if apply_to_all == True:
			signal_filtered = super(spectral_analysis, self).filter(signal = self.data, 
																fs = self.fsample, f_low = f_low, f_high = f_high, n_jobs = n_jobs)

		else:
			signal_filtered = super(spectral_analysis, self).filter(signal = self.data[trial, index_channel, :], 
																fs = self.fsample, f_low = f_low, f_high = f_high, n_jobs = n_jobs)

		#if self.save_filtered == True:
		#	self.results['filtered_'+str(f_low)+'_'+str(f_high)][str(trial)][str(index_channel)] = signal_filtered

		return signal_filtered

	def spectogram(self, trial = None, index_channel = None,  apply_to_all = False, freqs = np.arange(6,60,1), method = 'morlet', n_jobs = 1):

		if apply_to_all == True:
			W = super(spectral_analysis, self).spectogram(signal = self.data, fs = float(self.fsample), 
									                      freqs=freqs, method=method, n_jobs = n_jobs)			
		else:
			W = super(spectral_analysis, self).spectogram(signal = self.data[trial, index_channel, :][np.newaxis, np.newaxis, :], 
														  fs = float(self.fsample), 
									                      freqs=freqs, method=method, n_jobs = n_jobs)

		#if self.save_morlet == True:
		#	self.results['morlet'][str(trial)][str(index_channel)] = W

		return W

	def instantaneous_power(self, trial = None, index_channel = None, f_low = 30, f_high = 60, n_jobs = 1):
		
		signal_filtered = self.filter(trial = trial, index_channel = index_channel, 
									  apply_to_all = False, f_low = f_low, f_high = f_high, 
									  n_jobs = n_jobs)

		# Hilbert transform
		S               = sig.hilbert(signal_filtered)
		# Power
		P               = np.multiply( S, np.conj(S) )
		return P

	'''
	def pairwise_coherence(self, trial = None, index_pair = None, step = 25, dt = 250, 
					       fc = np.arange(6, 62, 2), df = 4, n_jobs = 1, save_to_file = True):

		self.step    = step
		self.dt      = dt
		self.fc      = fc
		self.df      = df

		#self.tidx    = np.arange(self.dt, session['data'].shape[2]-self.dt, self.step)

		#print('Trial: '+str(trial)+', Pair: '+str(index_pair))
		#coh = np.empty( [len(self.taxs), len(self.fc)] )
		coh = np.empty( [len(self.tarray), len(self.fc)] )

		S = (1+1j)*np.zeros([3, self.data.shape[2]])
		for nf in range( self.fc.shape[0] ):
			bpfreq = np.array( [ self.fc[nf]-self.df, self.fc[nf]+self.df ] )
			f_low, f_high  = bpfreq[0], bpfreq[1]
			sig1f  = self.filter(trial = trial, index_channel = self.pairs[index_pair, 0], 
								 f_low = f_low, f_high = f_high, n_jobs = n_jobs)
			sig2f  = self.filter(trial = trial, index_channel = self.pairs[index_pair, 1], 
				                 f_low = f_low, f_high = f_high, n_jobs = n_jobs)
			Sx     = sig.hilbert(sig1f)
			Sy     = sig.hilbert(sig2f)
			S[0, :] = np.multiply( Sx, np.conj(Sy) )
			S[1, :] = np.multiply( Sx, np.conj(Sx) )
			S[2, :] = np.multiply( Sy, np.conj(Sy) )
			Sm = sig.convolve2d(S.T, np.ones([self.dt, 1]), mode='same')
			#Sm = Sm[self.tidx,:]
			coh[:, nf] = ( Sm[:, 0]*np.conj(Sm[:, 0]) /(Sm[:, 1]*Sm[:,2]) ).real

		#if self.save_coh == True:
		#	self.results['coherence'][str(trial)] = {}
	#		self.results['coherence'][str(trial)][str(index_pair)] = coh

		if save_to_file == True:
			file_name = self.dir_out + 'trial_' +str(trial) + '_pair_' + str(self.pairs[index_pair, 0]) + '_' + str(self.pairs[index_pair, 1]) + '.npy'
			np.save(file_name, {'coherence' : coh})

		else:
			return coh
		'''

	def pairwise_coherence(self, trial = None, index_pair = None, step = 25, dt = 250, 
				       	   fc = np.arange(6, 62, 2), df = 4, n_jobs = 1, save_to_file = True):

		self.step    = step
		self.dt      = dt
		self.fc      = fc
		self.df      = df

		self.tidx    = np.arange(self.dt, self.data.shape[2]-self.dt, self.step)

		# Coherence matrix
		coh = np.empty( [len(self.tarray), len(self.fc)] )
		# Instantaneous power spectrum matrices
		S = (1+1j)*np.zeros([3, self.data.shape[2]])
		# Computing coherences
		for i in range( self.fc.shape[0] ):
			# Low and high frequency
			f_low, f_high = self.fc[i]-self.df, self.fc[i]+self.df
			# Channel 1 and 2 indexis
			ch1, ch2 = self.pairs[index_pair, 0], self.pairs[index_pair, 1]
			# Filtering each signal
			sig1f  = self.filter(trial = trial, index_channel = ch1, 
							 f_low = f_low, f_high = f_high, n_jobs = n_jobs)
			sig2f  = self.filter(trial = trial, index_channel = ch2, 
			                 f_low = f_low, f_high = f_high, n_jobs = n_jobs)
			Sx     = sig.hilbert(sig1f)
			Sy     = sig.hilbert(sig2f)
			# Computing instantaneous power spectrum
			S[0, :] = np.multiply( Sx, np.conj(Sy) )
			S[1, :] = np.multiply( Sx, np.conj(Sx) )
			S[2, :] = np.multiply( Sy, np.conj(Sy) )
			Sm = sig.convolve2d(S.T, np.ones([self.dt, 1]), mode='same')
			Sm = Sm[self.tidx,:]
			coh[:, i] = ( Sm[:, 0]*np.conj(Sm[:, 0]) / (Sm[:, 1]*Sm[:,2]) ).real

		if save_to_file == True:
			file_name = os.path.join( self.dir_out[:-1], 
				'trial_' +str(trial) + '_pair_' + str(self.pairs[index_pair, 0]) + '_' + str(self.pairs[index_pair, 1]) + '.npy')
			np.save(file_name, {'coherence' : coh})
		else:
			return coh

	def compute_coherences(self, n_jobs = 1):
		for trial in range(self.nT):
			Parallel(n_jobs=n_jobs, backend='loky', max_nbytes=1e6)(
				delayed(self.pairwise_coherence)
				(trial, index_pair, n_jobs = 1, save_to_file = True)
				for index_pair in range(self.nP)
				)
