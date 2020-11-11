import numpy            as     np
import networkx         as     nx
import os
import h5py
import multiprocessing
from   scipy            import stats
from   tqdm             import tqdm
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
            
        # Concatenate trials in the super tensor
        self.super_tensor = self.super_tensor.swapaxes(1,2)
        #  self.super_tensor = self.super_tensor.reshape( (self.session_info['nP'], len(self.bands), self.session_info['nT']*len(self.tarray) ) )
        self.super_tensor = self.reshape_observations(self.super_tensor)

        #  Creating variables that will store network analysis quantities
        self.__instantiate_dictionaries()

    def convert_to_adjacency(self,):
        self.A = np.zeros([self.session_info['nC'], self.session_info['nC'], len(self.bands), self.session_info['nT']*len(self.tarray)]) 
        for p in range(self.session_info['pairs'].shape[0]):
            i, j              = self.session_info['pairs'][p,0], self.session_info['pairs'][p,1]
            self.A[i,j,:,:]   = self.super_tensor[p,:,:]
            #  self.A[j,i,:,:]   = self.super_tensor[p,:,:]

    def compute_coherence_thresholds(self, q = .80):
        #  Coherence thresholds
        self.coh_thr = {}
        for i in range(len(self.bands)):
            self.coh_thr[str(i)] = stats.mstats.mquantiles( self.super_tensor[:,i,:].flatten(), prob = q )

        #  self.node_degree = np.zeros([2, self.session_info['nC'], len(self.bands), self.super_tensor.shape[2]])
    def compute_nodes_degree(self, band = 0, thr = None):
        A_tmp = self.A[:,:,band,:] + np.transpose( self.A[:,:,band,:], (1,0,2) )
        if thr == None:
            self.node_degree[0,:,band,:] = A_tmp.sum(axis=1) 
        else:
            A_tmp = A_tmp > thr
            self.node_degree[1,:,band,:] = A_tmp.sum(axis=1)  
            #  del A_tmp
        del A_tmp

    #  def compute_nodes_degree_nx(self, band, thr = None):
    #      self.degree[str(band)] = {}
    #      if thr == None:
    #          self.degree[str(band)]['w'] = np.zeros([self.A.shape[0], self.A.shape[3]])
    #          for t in tqdm(range(self.A.shape[3])):
    #              g = nx.Graph(self.A[:,:,band,t])
    #              self.degree['w'][:,t] = list( dict( g.degree(weight='weight') ).values() )
    #      else:
    #          self.degree[str(band)]['b'] = np.zeros([self.A.shape[0], self.A.shape[3]])
    #          for t in tqdm(range(self.A.shape[3])):
    #              g = nx.Graph(self.A[:,:,band,t]>thr)
    #              self.degree[str(band)]['b'][:,t] = list( dict( g.degree() ).values() )

        #  self.node_degree = np.zeros([2, self.session_info['nC'], len(self.bands), self.super_tensor.shape[2]])
    def compute_nodes_clustering_layerwise(self, band = 0, observation = 0, thr = None):
        #  self.clustering[str(band)] = {}
        if thr == None:
            #  g = nx.Graph(self.A[:,:,band,t])
            g = self.instantiate_graph(band=band, observation=observation, thr = None)
            self.clustering[0,:,band,observation] = list( dict( nx.clustering(g, weight='weight') ).values() )
        else:
            #  self.clustering[str(band)]['b'] = np.zeros([self.A.shape[0], self.A.shape[3]])
            #  g = nx.Graph(self.A[:,:,band,t]>thr)
            g = self.instantiate_graph(band=band, observation=observation, thr = thr)
            self.clustering[1,:,band,observation] = list( dict( nx.clustering(g) ).values() )

    def compute_nodes_clustering(self, band = 0, thr = None):
        #  self.clustering[str(band)] = {}
        if thr == None:
            #  self.clustering[str(band)]['w'] = np.zeros([self.A.shape[0], self.A.shape[3]])
            for t in tqdm(range(self.A.shape[3])):
                g = nx.Graph(self.A[:,:,band,t])
                self.clustering[0,:,band,t] = list( dict( nx.clustering(g, weight='weight') ).values() )
        else:
            #  self.clustering[str(band)]['b'] = np.zeros([self.A.shape[0], self.A.shape[3]])
            for t in tqdm(range(self.A.shape[3])):
                g = nx.Graph(self.A[:,:,band,t]>thr)
                self.clustering[1,:,band,t] = list( dict( nx.clustering(g) ).values() )

        #  self.node_degree = np.zeros([2, self.session_info['nC'], len(self.bands), self.super_tensor.shape[2]])
    def compute_nodes_coreness(self, band = 0, thr = None):
        #  self.coreness[str(band)] = {}
        if thr == None:
            #  self.coreness[str(band)]['w'] = np.zeros([self.A.shape[0], self.A.shape[3]])
            for t in tqdm(range(self.A.shape[3])):
                g = nx.Graph(self.A[:,:,band,t])
                self.coreness[0,:,band,t] = list( dict( nx.core_number(g, weight='weight') ).values() )
        else:
            #  self.coreness[str(band)]['b'] = np.zeros([self.A.shape[0], self.A.shape[3]])
            for t in tqdm(range(self.A.shape[3])):
                g = nx.Graph(self.A[:,:,band,t]>thr)
                self.coreness[1,:,band,t] = list( dict( nx.core_number(g) ).values() )


        #  self.node_degree = np.zeros([2, self.session_info['nC'], len(self.bands), self.super_tensor.shape[2]])
    def compute_nodes_betweenes(self, k = None, band = 0, thr = None):
        #  self.coreness[str(band)] = {}
        if thr == None:
            #  self.coreness[str(band)]['w'] = np.zeros([self.A.shape[0], self.A.shape[3]])
            for t in tqdm(range(self.A.shape[3])):
                g = nx.Graph(self.A[:,:,band,t])
                self.betweenes[0,:,band,t] = list( dict( nx.betweenness_centrality(g, k, weight='weight') ).values() )
        else:
            #  self.coreness[str(band)]['b'] = np.zeros([self.A.shape[0], self.A.shape[3]])
            for t in tqdm(range(self.A.shape[3])):
                g = nx.Graph(self.A[:,:,band,t]>thr)
                self.betweenes[1,:,band,t] = list( dict( nx.betweenness_centrality(g, k) ).values() )

    def NMF_decomposition(self, band = 0, k = 2):
        None

    def instantiate_graph(self, band = 0, observation = 0, thr = None):
        if thr == None:
            return nx.Graph(self.A[:,:,band,observation])
        else:
            return nx.Graph(self.A[:,:,band,observation]>thr)

    def reshape_trials(self, tensor):
        #  Reshape the tensor to have trials and time as two separeted dimension
        print(len(tensor.shape))
        if len(tensor.shape) == 2:
            aux = tensor.reshape([tensor.shape[0], self.session_info['nT'], len(self.tarray)])
        if len(tensor.shape) == 3:
            aux = tensor.reshape([tensor.shape[0], tensor.shape[1], self.session_info['nT'], len(self.tarray)])
        if len(tensor.shape) == 4:
            aux = tensor.reshape([tensor.shape[0], tensor.shape[1], tensor.shape[2], self.session_info['nT'], len(self.tarray)])
        return aux

    def reshape_observations(self, tensor):
        #  Reshape the tensor to have all the trials concatenated in the same dimension
        if len(tensor.shape) == 3:
            aux = tensor.reshape([tensor.shape[0], self.session_info['nT'] * len(self.tarray)])
        if len(tensor.shape) == 4:
            aux = tensor.reshape([tensor.shape[0], tensor.shape[1], self.session_info['nT'] * len(self.tarray)])
        if len(tensor.shape) == 5:
            aux = tensor.reshape([tensor.shape[0], tensor.shape[1], tensor.shape[2], self.session_info['nt'] * len(self.tarray)])
        return aux

    def create_stages_time_grid(self, ):
        t_cue_off  = (self.session_info['t_cue_off']-self.session_info['t_cue_on'])/self.session_info['fsample']
        t_match_on = (self.session_info['t_match_on']-self.session_info['t_cue_on'])/self.session_info['fsample']
        tt         = np.tile(self.tarray, (self.session_info['nT'], 1))
        #  Create grids with starting and ending times of each stage for each trial
        self.t_baseline = ( (tt<0) ).reshape(self.session_info['nT']*len(self.tarray))
        self.t_cue      = ( (tt>=0)*(tt<t_cue_off[:,None]) ).reshape(self.session_info['nT']*len(self.tarray))
        self.t_delay    = ( (tt>=t_cue_off[:,None])*(tt<t_match_on[:,None]) ).reshape(self.session_info['nT']*len(self.tarray))
        self.t_match    = ( (tt>=t_match_on[:,None]) ).reshape(self.session_info['nT']*len(self.tarray))

    def __instantiate_dictionaries(self,):
        #  This method creates the arrays where network measurements will be stored
        #  Network degree 
        self.node_degree = np.zeros([2, self.session_info['nC'], len(self.bands), self.super_tensor.shape[2]])
        #  Network clustering
        self.clustering  = np.zeros([2, self.session_info['nC'], len(self.bands), self.super_tensor.shape[2]])
        #  Network coreness
        self.coreness    = np.zeros([2, self.session_info['nC'], len(self.bands), self.super_tensor.shape[2]])
        #  Network betweenes
        self.betweenes    = np.zeros([2, self.session_info['nC'], len(self.bands), self.super_tensor.shape[2]])

