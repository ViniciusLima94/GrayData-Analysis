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
        self.date     = date                            
        self.session  = 'session0' + str(session)       
        
        # Path to the session information in npy format
        npy_raw_lfp_path     = os.path.join('raw_lfp', monkey+'_'+'session0'+str(session)+'_'+str(date)+'.npy')
        # Path to the super tensor in h5 format 
        h5_super_tensor_path = os.path.join('super_tensors', monkey+'_'+'session0'+str(session)+'_'+str(date)+'.h5')
        # Loading session info
        self.session_info = np.load(npy_raw_lfp_path, allow_pickle=True).item()['info']
        # Loading super teson (temporal network)
        with h5py.File(h5_super_tensor_path, 'r') as hf:
            #  print(list(hf.keys()))
            self.super_tensor = np.array( hf.get('supertensor') )
            self.tarray       = np.array( hf.get('tarray') )
            self.freqs        = np.array( hf.get('freqs') )
            self.bands        = np.array( hf.get('bands') )
            
    def convert_to_adjacency(self,):
        self.A = np.zeros([self.session_info['nC'], self.session_info['nC'], self.session_info['nT'], len(self.bands), len(self.tarray)]) 
        for p in range(self.session_info['pairs'].shape[0]):
            i, j              = self.session_info['pairs'][p,0], self.session_info['pairs'][p,1]
            self.A[i,j,:,:,:] = self.super_tensor[p,:,:,:]

    def compute_nodes_strength(self,):
        None

    def compute_nodes_clustering(self,):
        None
