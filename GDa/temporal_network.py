import numpy            as     np
import networkx         as     nx
import os
import h5py
import multiprocessing
from   joblib           import Parallel, delayed


class temporal_network():

    def __init__(self, monkey='lucy', date=150128, session=1):
        # Setting up mokey and recording info to load and save files
        self.monkey   = monkey
        self.raw_path = raw_path                        # Path to the session information in npy format
        self.date     = date                            npy_raw_lfp_path     = os.path.join('raw_lfp', monkey+'_'+'session0'+str(session)+'_'+str(date)+'.npy')
        self.session  = 'session0' + str(session)       h5_super_tensor_path = os.path.join('raw_lfp', monkey+'_'+'session0'+str(session)+'_'+str(date)+'.h5')

        # Loading session info
        self.session_info = np.load(npy_raw_lfp_path, allow_pickle=True).item()['info']
        # Loading super teson (temporal network)
        with h5py.File(h5_super_tensor_path, 'w'):


        
    def convert_to_adjacency(self,):
        None

    def compute_nodes_strength(self,):
        None

    def 
