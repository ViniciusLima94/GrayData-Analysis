#####################################################################################################
# Library with methods to perform spetral analysis
#####################################################################################################
import numpy         as np
import scipy.signal  as sig
import mne.filter
import neo
import elephant
from   quantities    import s, Hz


def compute_freq(N, fs):
	# Time length
	T = N / fs
	# Frequency array
	f = np.linspace(1/T, fs/2-1/T, N/2+1)

	return f

def bp_filter(signal = None, fs = 20, f_low = 30, f_high = 60, n_jobs = 1):
	signal_filtered = mne.filter.filter_data(signal, fs, f_low, f_high, 
											 method = 'iir', verbose=False, n_jobs=n_jobs)

	return signal_filtered

def wavelet_morlet(signal = None, fs = 20):

	N = signal.shape[0]

	X = neo.AnalogSignal(signal, t_start = 0*s, sampling_rate = fs*Hz, units='dimensionless')
	W = elephant.signal_processing.wavelet_transform(X, np.range(100), fs=self.fsample).reshape((N,len(f)))

	return W

def pairwise_coherence(data, trial, index_pair, taxs, tidx, fc, df, dt, fsample, pairs, dir_out, n_jobs = 1, save_to_file = True, return_coh=False):
		print('Trial: '+str(trial)+', Pair: '+str(index_pair))
		coh = np.empty( [len(taxs), len(fc)] )

		# First LFP
		sig1 = data[trial, pairs[index_pair, 0], :].copy()
		# Second LFP
		sig2 = data[trial, pairs[index_pair, 1], :].copy()

		S = (1+1j)*np.zeros([3, data.shape[2]])
		for nf in range( fc.shape[0] ):
			bpfreq = np.array( [ fc[nf]-df, fc[nf]+df ] )
			f_low, f_high  = bpfreq[0], bpfreq[1]
			sig1f  = bp_filter(signal = sig1, fs = fsample, f_low = f_low, f_high = f_high, n_jobs = n_jobs)
			sig2f  = bp_filter(signal = sig2, fs = fsample, f_low = f_low, f_high = f_high, n_jobs = n_jobs)
			Sx     = sig.hilbert(sig1f)
			Sy     = sig.hilbert(sig2f)
			S[0, :] = np.multiply( Sx, np.conj(Sy) )
			S[1, :] = np.multiply( Sx, np.conj(Sx) )
			S[2, :] = np.multiply( Sy, np.conj(Sy) )
			Sm = sig.convolve2d(S.T, np.ones([dt, 1]), mode='same') 
			Sm = Sm[tidx,:]
			coh[:, nf] = ( Sm[:, 0]*np.conj(Sm[:, 0]) /(Sm[:, 1]*Sm[:,2]) ).real

		if save_to_file == True:
			file_name = dir_out + 'trial_' +str(trial) + '_pair_' + str(pairs[index_pair, 0]) + '_' + str(pairs[index_pair, 1]) + '.npy'
			np.save(file_name, {'coherence' : coh})
		if return_coh == True:
			return coh