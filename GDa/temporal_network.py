import numpy            as     np
import networkx         as     nx
import os
import h5py
import multiprocessing
from   joblib           import Parallel, delayed


class temporal_network():

	def __init__(self, monkey='lucy', date=150128, session=1):

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

    def convert_to_adjacency(self,):
        None

    def compute_nodes_strength(self,):
        None

    def 
