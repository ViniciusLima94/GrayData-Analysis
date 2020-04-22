import numpy         as np
import scipy.signal  as sig
import mne.filter

class spectral_analysis():

	def __init__(self, LFP = None, path = None, step = 25, dt = 250, fc = np.arange(6, 62, 2), df = 4):

		self.step    = step
		self.dt      = dt
		self.fc      = fc
		self.df      = df

		if LFP == None:
			LFP = np.load(path, allow_pickle=True).item()
			self.nP      = LFP['info']['nP']
			self.pairs   = LFP['info']['pairs']
			self.tarray  = LFP['info']['tarray']
			self.tidx    = np.arange(self.dt, LFP['data'].shape[2]-self.dt, self.step)
			self.taxs    = LFP['info']['tarray'][self.tidx]
			self.fsample = LFP['info']['fsample']
			self.data    = LFP['data']
		else:
			self.nP      = LFP.nP
			self.pairs   = LFP.pairs
			self.tarray  = LFP.time[0]
			self.tidx    = np.arange(self.dt, LFP.data.shape[2]-self.dt, self.step)
			self.taxs    = LFP.time[0][self.tidx]
			self.fsample = LFP.recording_info['fsample']
			self.data    = LFP.data

	def filter(self, signal = None, trial = None, index_channel = None, f_low = 30, f_high = 60, n_jobs = 1):

		if trial == None and index_channel == None:
			signal_filtered = mne.filter.filter_data(signal, self.fsample, f_low, f_high, 
												 method = 'iir', verbose=False, n_jobs=n_jobs)

		else:
			signal_filtered = mne.filter.filter_data(self.data[trial, index_channel, :], self.fsample, f_low, f_high, 
												 method = 'iir', verbose=False, n_jobs=n_jobs)


		return signal_filtered

	def pairwise_coherence(self, trial, index_pair, n_jobs = 1):
			print('Trial: '+str(trial)+', Pair: '+str(index_pair))
			self.coh = np.empty( [len(self.taxs), len(self.fc)] )
			# First LFP
			sig1 = self.data[trial, self.pairs[index_pair, 0], :].copy()
			# Second LFP
			sig2 = self.data[trial, self.pairs[index_pair, 1], :].copy()

			S = (1+1j)*np.zeros([3, self.data.shape[2]])
			for nf in range( self.fc.shape[0] ):
				bpfreq = np.array( [ self.fc[nf]-self.df, self.fc[nf]+self.df ] )
				f_low, f_high  = bpfreq[0], bpfreq[1]
				sig1f  = self.filter(signal = sig1, f_low = f_low, f_high = f_high, n_jobs = n_jobs)
				sig2f  = self.filter(signal = sig2, f_low = f_low, f_high = f_high, n_jobs = n_jobs)
				Sx     = sig.hilbert(sig1f)
				Sy     = sig.hilbert(sig2f)
				S[0, :] = np.multiply( Sx, np.conj(Sy) )
				S[1, :] = np.multiply( Sx, np.conj(Sx) )
				S[2, :] = np.multiply( Sy, np.conj(Sy) )
				Sm = sig.convolve2d(S.T, np.ones([self.dt, 1]), mode='same') 
				Sm = Sm[self.tidx,:]
				self.coh[:, nf] = ( Sm[:, 0]*np.conj(Sm[:, 0]) /(Sm[:, 1]*Sm[:,2]) ).real
				'''
				if as_mat==False:
					file_name = session['dir_out']+dirs['session']+'_'+dirs['date'][nmonkey][nses]+'_trial_'+str(nT)+'_pair_'+str(pairs[nP, 0])+'_'+str(pairs[nP, 1])+'.dat'
					np.savetxt(file_name, coh.T)
				elif as_mat==True:
					file_name = session['dir_out']+dirs['session']+'_'+dirs['date'][nmonkey][nses]+'_trial_'+str(nT)+'_pair_'+str(pairs[nP, 0])+'_'+str(pairs[nP, 1])+'.mat'
					coh = {'trial':coh.T}
					scio.savemat(file_name, coh )
				'''