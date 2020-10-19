import numpy as np 
import os
from  .io    import set_paths 

class super_tensor(set_paths):

	def __init__(self, raw_path = 'GrayLab/', monkey = 'lucy', date = '150128', session = 1, delta = 1, freqs = np.arange(6,60,1)):
		'''
		Constructor method.
		Inputs
			> raw_path : Path containing the raw data.
			> monkey   : Monkey name, should be either lucy or ethyl.
			> date     : The date of the session to use.
			> session  : The number of the session, should be either session01 or session02
		'''
		super().__init__(raw_path = raw_path, monkey = monkey, date = date, session = session)
		# Path to the raw LPF in npy format
		npy_raw_lfp_path = os.path.join('raw_lfp', monkey+'_'+'session0'+str(session)+'_'+str(date)+'.npy')
		# Loading session data
		session = np.load(npy_raw_lfp_path, allow_pickle=True).item()
		# Data is deleted to not use memory
		del session['data']
		# Loading session info
		self.nC      = session['info']['nC']
		self.nP      = session['info']['nP']
		self.nT      = session['info']['nT']
		self.pairs   = session['info']['pairs']
		self.indt    = session['info']['indt']
		self.indch   = session['info']['indch'] 
		self.areas   = session['info']['areas']  
		self.dir     = session['path']['dir']
		self.dir_out = session['path']['dir_out']
		self.dcue    = session['info']['dcue']
		self.dsm     = session['info']['dsm']
		self.stim    = session['info']['stim']
		self.fsample = float(session['info']['fsample'])
		self.freqs   = freqs
		self.tarray  = session['info']['tarray'][::delta]
		self.channel_labels = session['info']['channels_labels']
		self.t_cue_on       = session['info']['t_cue_on'] 
		self.t_cue_off      = session['info']['t_cue_off'] 
		self.t_match_on     = session['info']['t_match_on'] 
		
	def load_super_tensor(self, use_gpu = False):

		self._super_tensor = np.zeros([self.nT, self.nP, self.freqs.shape[0], self.tarray.shape[0]])
		for i in range(self.nT):
			for j in range(self.nP):
				#print('Trial = ' + str(i) + ', pair = ' + str(j))
				path                        = os.path.join( self.dir_out, 'trial_'+str(i)+'_pair_'+str(j)+'.npy' )
				self._super_tensor[i,j,:,:] = np.load(path, allow_pickle=True).item()['coherence'].real