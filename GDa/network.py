import numpy            as     np
import networkx         as     nx
import os
import h5py
import multiprocessing
from   joblib           import Parallel, delayed


class network():

	def __init__(self, monkey='lucy', date=150128, session=1):
		None

		super().__init__(raw_path = raw_path, monkey = monkey, date = date, session = session)
		###
		self.monkey   = monkey
		self.raw_path = raw_path
		self.date     = date 
		self.session  = 'session0' + str(session)

		# Path to the raw LPF in npy format
		npy_raw_lfp_path = os.path.join('raw_lfp', monkey+'_'+'session0'+str(session)+'_'+str(date)+'.npy')
		# Loading session data
		self.session_info = np.load(npy_raw_lfp_path, allow_pickle=True).item()['info']
		# Data is deleted to not use memory
		del session_data['data']
		# Copy info in session to the object
		self.session_data = session_data
		# Loading session info
		self.nP      = session_data['info']['nP']
		if trial_subset == None:
			#print(session_data['info']['nT'])
			self.nT      = session_data['info']['nT']
		else:
			self.nT = trial_subset
		self.dir_out = session_data['path']['dir_out']
		self.freqs   = freqs
		self.tarray  = session_data['info']['tarray'][::delta]
		self.pairs   = session_data['info']['pairs']
		self.areas   = session_data['info']['areas']
		self.t_cue_on   = session_data['info']['t_cue_on']
		self.t_cue_off  = session_data['info']['t_cue_off']
		self.t_match_on = session_data['info']['t_match_on']

    def convert_to_adjacency(self,):
        None

    def compute_nodes_strength(self,):
        None

    def 
