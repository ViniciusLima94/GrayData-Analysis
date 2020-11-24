import numpy            as     np
import networkx         as     nx
import igraph           as     ig
import leidenalg
import os
import h5py
import multiprocessing
from sklearn.decomposition import NMF
from   scipy               import stats
from   tqdm                import tqdm
from   joblib              import Parallel, delayed

class temporal_network():

    def __init__(self, monkey='lucy', session=1, date=150128, wt = None, trim_borders = False):
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

        if trim_borders == True:
            self.tarray       = self.tarray[wt:-wt] 
            self.super_tensor = self.super_tensor[:,:,:,wt:-wt]

        # Concatenate trials in the super tensor
        self.super_tensor = self.super_tensor.swapaxes(1,2)
        #  self.super_tensor = self.super_tensor.reshape( (self.session_info['nP'], len(self.bands), self.session_info['nT']*len(self.tarray) ) )
        self.super_tensor = self.reshape_observations(self.super_tensor)

        #  Creating variables that will store network analysis quantities
        #  self.__instantiate_measurement_arrays()

    def convert_to_adjacency(self,):
        self.A = np.zeros([self.session_info['nC'], self.session_info['nC'], len(self.bands), self.session_info['nT']*len(self.tarray)]) 
        for p in range(self.session_info['pairs'].shape[0]):
            i, j              = self.session_info['pairs'][p,0], self.session_info['pairs'][p,1]
            self.A[i,j,:,:]   = self.super_tensor[p,:,:]
            #  self.A[j,i,:,:]   = self.super_tensor[p,:,:]

    def compute_coherence_thresholds(self, q = .80):
        #  Coherence thresholds
        self.coh_thr = np.zeros(len(self.bands)) 
        for i in range(len(self.bands)):
            self.coh_thr[i] = stats.mstats.mquantiles( self.super_tensor[:,i,:].flatten(), prob = q )

        #  self.node_degree = np.zeros([2, self.session_info['nC'], len(self.bands), self.super_tensor.shape[2]])
    def compute_nodes_degree(self, band = 0, thr = None, on_null = False):
        #  Variable to store node degree
        node_degree = np.zeros([self.session_info['nC'], self.super_tensor.shape[2]])

        if on_null == False:
            A_tmp = self.A[:,:,band,:] + np.transpose( self.A[:,:,band,:], (1,0,2) )
        else:
            A_tmp = self.A_null[:,:,band,:]

        if thr == None:
            #  self.node_degree[0,:,band,:] = A_tmp.sum(axis=1) 
            node_degree = A_tmp.sum(axis=1) 
        else:
            A_tmp = A_tmp > thr
            #  self.node_degree[1,:,band,:] = A_tmp.sum(axis=1)  
            node_degree = A_tmp.sum(axis=1)  
            #  del A_tmp
        #  del A_tmp
        return node_degree

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

    #  def compute_nodes_clustering_layerwise(self, band = 0, observation = 0, thr = None):
    #      #  self.clustering[str(band)] = {}
    #      if thr == None:
    #          #  g = nx.Graph(self.A[:,:,band,t])
    #          g = self.instantiate_graph(band=band, observation=observation, thr = None)
    #          self.clustering[0,:,band,observation] = list( dict( nx.clustering(g, weight='weight') ).values() )
    #      else:
    #          #  self.clustering[str(band)]['b'] = np.zeros([self.A.shape[0], self.A.shape[3]])
    #          #  g = nx.Graph(self.A[:,:,band,t]>thr)
    #          g = self.instantiate_graph(band=band, observation=observation, thr = thr)
    #          self.clustering[1,:,band,observation] = list( dict( nx.clustering(g) ).values() )

    def compute_nodes_clustering(self, band = 0, thr = None, use = 'networkx', on_null = False):
        #  Variable to store node clustering
        clustering  = np.zeros([self.session_info['nC'], self.super_tensor.shape[2]])

        for t in tqdm(range(self.A.shape[3])):
            g = self.instantiate_graph(band,t,thr=thr,on_null = on_null)
            if use == 'networkx':
                clustering[:,t] = list( dict( nx.clustering(g) ).values() )
            elif use=='igraph':
                clustering[:,t] = ig.Graph(self.session_info['nC'], g.edges).transitivity_local_undirected()
        return np.nan_to_num( clustering )

        #  if thr == None:
        #      #  self.clustering[str(band)]['w'] = np.zeros([self.A.shape[0], self.A.shape[3]])
        #      for t in tqdm(range(self.A.shape[3])):
        #          g = self.instantiate_graph(band,t,thr=None, on_null = on_null)#nx.Graph(self.A[:,:,band,t])
        #          #  self.clustering[0,:,band,t] = list( dict( nx.clustering(g, weight='weight') ).values() )
        #          clustering[:,t] = list( dict( nx.clustering(g, weight='weight') ).values() )
        #  else:
        #      #  self.clustering[str(band)]['b'] = np.zeros([self.A.shape[0], self.A.shape[3]])
        #      for t in tqdm(range(self.A.shape[3])):
        #          g = self.instantiate_graph(band,t,thr=thr,on_null = on_null)#nx.Graph(self.A[:,:,band,t]>thr)
        #          if use == 'networkx':
        #              #  self.clustering[1,:,band,t] = list( dict( nx.clustering(g) ).values() )
        #              clustering[:,t] = list( dict( nx.clustering(g) ).values() )
        #          elif use=='igraph':
        #              #  self.clustering[1,:,band,t] = ig.Graph(self.session_info['nC'], g.edges).transitivity_local_undirected()
        #              clustering[:,t] = ig.Graph(self.session_info['nC'], g.edges).transitivity_local_undirected()
        #  return clustering

    def compute_nodes_coreness(self, band = 0, thr = None, use='networkx', on_null = False):
        #  Variable to store coreness
        coreness    = np.zeros([self.session_info['nC'], self.super_tensor.shape[2]])
 
        for t in tqdm(range(self.A.shape[3])):
            g = self.instantiate_graph(band,t,thr=thr,on_null=on_null)
            if use=='networkx':
                coreness[:,t] = list( dict( nx.core_number(g) ).values() )
            elif use=='igraph':
                coreness[:,t] = ig.Graph(self.session_info['nC'], g.edges).coreness()
        return coreness

        #  if thr == None:
        #      #  self.coreness[str(band)]['w'] = np.zeros([self.A.shape[0], self.A.shape[3]])
        #      for t in tqdm(range(self.A.shape[3])):
        #          if   on_null==False: g = nx.Graph(self.A[:,:,band,t])
        #          elif on_null==True:  g = nx.Graph(self.A_null[:,:,band,t])
        #          #  g = self.instantiate_graph(band,t,thr=None,on_null=on_null)#nx.Graph(self.A[:,:,band,t])
        #          #  self.coreness[0,:,band,t] = list( dict( nx.core_number(g, weight='weight') ).values() )
        #          coreness[:,t] = list( dict( nx.core_number(g, weight='weight') ).values() )
        #  else:
        #      #  self.coreness[str(band)]['b'] = np.zeros([self.A.shape[0], self.A.shape[3]])
        #      for t in tqdm(range(self.A.shape[3])):
        #          if   on_null==False: g = nx.Graph(self.A[:,:,band,t]>thr)
        #          elif on_null==True:  g = nx.Graph(self.A_null[:,:,band,t]>thr)
        #          #  g = self.instantiate_graph(band,t,thr=thr,on_null=on_null)#nx.Graph(self.A[:,:,band,t]>thr)
        #          if use=='networkx':
        #              #  self.coreness[1,:,band,t] = list( dict( nx.core_number(g) ).values() )
        #              coreness[:,t] = list( dict( nx.core_number(g) ).values() )
        #          elif use=='igraph':
        #              #  self.coreness[1,:,band,t] = ig.Graph(self.session_info['nC'], g.edges).coreness() 
        #              coreness[:,t] = ig.Graph(self.session_info['nC'], g.edges).coreness()
        #  return coreness

    def compute_network_partition(self, band = 0, thr = None, use='networkx', on_null=False):
        partition = []

        for t in tqdm(range(self.A.shape[3])):
            g = self.instantiate_graph(band,t,thr=thr, on_null=on_null)
            if use=='networkx':
                partition.append(nx.algorithms.community.greedy_modularity_communities(g))
            elif use=='igraph':
                #  This one uses leidenalg
                partition.append(leidenalg.find_partition(ig.Graph(self.session_info['nC'], g.edges), leidenalg.ModularityVertexPartition))
        return partition
    
    def compute_network_modularity(self, band = 0, thr = None, use='networkx', on_null=False):
        #  Variable to store modularity
        modularity  = np.zeros(self.super_tensor.shape[2])

        for t in tqdm(range(self.A.shape[3])):
            g = self.instantiate_graph(band,t,thr=thr, on_null=on_null)#nx.Graph(self.A[:,:,band,t]>thr)
            if use=='networkx':
                comm = nx.algorithms.community.greedy_modularity_communities(g)
                modularity[t] = nx.algorithms.community.modularity(g,comm)
            elif use=='igraph':
                #  This one uses leidenalg
                modularity[t] = leidenalg.find_partition(ig.Graph(self.session_info['nC'], g.edges), leidenalg.ModularityVertexPartition).modularity
        return modularity

        #  if thr == None:
        #      #  self.coreness[str(band)]['w'] = np.zeros([self.A.shape[0], self.A.shape[3]])
        #      for t in tqdm(range(self.A.shape[3])):
        #          g = self.instantiate_graph(band,t,thr=None, on_null=on_null)#nx.Graph(self.A[:,:,band,t])
        #          #  Finding communities
        #          comm = nx.algorithms.community.greedy_modularity_communities(g)
        #          #  self.modularity[0,band,t] = nx.algorithms.community.modularity(g,comm, weight='weight') 
        #          modularity[t] = nx.algorithms.community.modularity(g,comm, weight='weight') 
        #  else:
        #      #  self.coreness[str(band)]['b'] = np.zeros([self.A.shape[0], self.A.shape[3]])
        #      for t in tqdm(range(self.A.shape[3])):
        #          g = self.instantiate_graph(band,t,thr=thr, on_null=on_null)#nx.Graph(self.A[:,:,band,t]>thr)
        #          if use=='networkx':
        #              comm = nx.algorithms.community.greedy_modularity_communities(g)
        #              #  self.modularity[1,band,t] = nx.algorithms.community.modularity(g,comm)
        #              modularity[t] = nx.algorithms.community.modularity(g,comm)
        #          elif use=='igraph':
        #              #  This one uses leidenalg
        #              #  self.modularity[1,band,t] = leidenalg.find_partition(ig.Graph(self.session_info['nC'], g.edges), leidenalg.ModularityVertexPartition).modularity
        #              modularity[t] = leidenalg.find_partition(ig.Graph(self.session_info['nC'], g.edges), leidenalg.ModularityVertexPartition).modularity
        #  return modularity

    def compute_nodes_betweenness(self, k = None, band = 0, thr = None, use='networkx', on_null=False):
        betweenness = np.zeros([self.session_info['nC'], self.super_tensor.shape[2]])

        for t in tqdm(range(self.A.shape[3])):
            g = self.instantiate_graph(band,t,thr=thr, on_null=on_null)
            if use=='networkx':
                betweenness[:,t] = list( dict( nx.betweenness_centrality(g, k) ).values() )
            elif use=='igraph':
                betweenness[:,t] = ig.Graph(self.session_info['nC'], g.edges).betweenness()
        return betweenness

        #  if thr == None:
        #      #  self.coreness[str(band)]['w'] = np.zeros([self.A.shape[0], self.A.shape[3]])
        #      for t in tqdm(range(self.A.shape[3])):
        #          g = self.instantiate_graph(band,t,thr=None, on_null=on_null)#nx.Graph(self.A[:,:,band,t])
        #          #  self.betweennes[0,:,band,t] = list( dict( nx.betweenness_centrality(g, k, weight='weight') ).values() )
        #          betweennes[:,t] = list( dict( nx.betweenness_centrality(g, k, weight='weight') ).values() )
        #  else:
        #      #  self.coreness[str(band)]['b'] = np.zeros([self.A.shape[0], self.A.shape[3]])
        #      for t in tqdm(range(self.A.shape[3])):
        #          g = self.instantiate_graph(band,t,thr=thr, on_null=on_null)#nx.Graph(self.A[:,:,band,t]>thr)
        #          if use=='networkx':
        #              #  self.betweenness[1,:,band,t] = list( dict( nx.betweenness_centrality(g, k) ).values() )
        #              betweenness[:,t] = list( dict( nx.betweenness_centrality(g, k) ).values() )
        #          elif use=='igraph':
        #              #  self.betweenness[1,:,band,t] = ig.Graph(self.session_info['nC'], g.edges).betweenness()
        #              betweenness[:,t] = ig.Graph(self.session_info['nC'], g.edges).betweenness()
        #  return betweenness

    def NMF_decomposition(self, band = 0, k = 2):

        def NMF_decomposition_single(band, k):
            model = NMF(n_components=k, init='random', random_state=0)
            W     = model.fit_transform(self.super_tensor[:,band,:])
            H     = model.components_
        
        Parallel(n_jobs=n_jobs, backend='multiprocessing', timeout=1e6)(delayed(NMF_decomposition_single)(band, k_) for k_ in k )

        return W, H

    def instantiate_graph(self, band = 0, observation = 0, thr = None, on_null = False):
        if on_null == False:
            A_tmp = self.A[:,:,band,:] #+ np.transpose( self.A[:,:,band,:], (1,0,2) )
        else:
            A_tmp = self.A_null[:,:,band,:]# + np.transpose( self.A_null[:,:,band,:], (1,0,2) )

        if thr == None:
            #  return nx.Graph(self.A[:,:,band,observation])
            return nx.Graph(A_tmp[:,:,observation])
        else:
            #  return nx.Graph(self.A[:,:,band,observation]>thr)
            return nx.Graph(A_tmp[:,:,observation]>thr)

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

    def create_null_model(self, randomize='edges'):
        self.A_null = np.zeros_like(self.A)
        if randomize=='time':
            idx = np.arange(self.session_info['nT']*len(self.tarray), dtype = int)
            np.random.shuffle(idx)
            self.A_null = ( self.A + np.transpose( self.A, (1,0,2,3) ) ).copy()
            self.A_null = self.A_null[:,:,:,idx]
        elif randomize=='edges':
            idx = np.arange(self.session_info['nC'], dtype = int)
            self.A_null = ( self.A + np.transpose( self.A, (1,0,2,3) ) ).copy()
            for t in range(self.session_info['nT']*len(self.tarray)):
                np.random.shuffle(idx)
                self.A_null[:,:,:,t] = self.A_null[:,idx,:,t]
        else:
            print('Randomize should be time or edges')

    def create_stim_grid(self, ):
        #  Number of different stimuli
        n_stim           = int((self.session_info['stim']).max()+1)
        #  Repeate each stimulus to match the length of the trial
        stim             = np.repeat(self.session_info['stim'], len(self.tarray) )
        self.stim_grid   = np.zeros([n_stim, self.session_info['nT']*len(self.tarray)])
        for i in range(n_stim):
            self.stim_grid[i] = (stim == i).astype(bool)

    def compute_temporal_correlation(self, band = 0, thr = None, tau = 1, on_null = False):
        if on_null == True:
            A = self.A_null[:,:,band,:]  
        else:
            A = self.A[:,:,band,:] + np.transpose(self.A[:,:,band,:], (1,0,2)) 
        
        if thr != None:
            A = A > thr

        if tau < 1:
            tau = 1

        num = (A[:,:,0:-tau] * A[:,:,tau:]).sum(axis=1)
        den = np.sqrt(A[:,:,0:-tau].sum(axis = 1) * A[:,:,tau:].sum(axis = 1))
        Ci  = np.nansum(( num / den ), axis=1) / (A.shape[-1] - 1)
        return np.nansum(Ci) / self.session_info['nC']

    def create_stages_time_grid(self, ):
        t_cue_off  = (self.session_info['t_cue_off']-self.session_info['t_cue_on'])/self.session_info['fsample']
        t_match_on = (self.session_info['t_match_on']-self.session_info['t_cue_on'])/self.session_info['fsample']
        tt         = np.tile(self.tarray, (self.session_info['nT'], 1))
        #  Create grids with starting and ending times of each stage for each trial
        self.t_baseline = ( (tt<0) ).reshape(self.session_info['nT']*len(self.tarray))
        self.t_cue      = ( (tt>=0)*(tt<t_cue_off[:,None]) ).reshape(self.session_info['nT']*len(self.tarray))
        self.t_delay    = ( (tt>=t_cue_off[:,None])*(tt<t_match_on[:,None]) ).reshape(self.session_info['nT']*len(self.tarray))
        self.t_match    = ( (tt>=t_match_on[:,None]) ).reshape(self.session_info['nT']*len(self.tarray))

    #  def __instantiate_measurement_arrays(self,):
    #      #  This method creates the arrays where network measurements will be stored
    #      #  Network degree 
    #      self.node_degree = np.zeros([2, self.session_info['nC'], len(self.bands), self.super_tensor.shape[2]])
    #      #  Network clustering
    #      self.clustering  = np.zeros([2, self.session_info['nC'], len(self.bands), self.super_tensor.shape[2]])
    #      #  Network coreness
    #      self.coreness    = np.zeros([2, self.session_info['nC'], len(self.bands), self.super_tensor.shape[2]])
    #      #  Network betweennes
    #      self.betweenness = np.zeros([2, self.session_info['nC'], len(self.bands), self.super_tensor.shape[2]])
    #      #  Network modularity
    #      self.modularity  = np.zeros([2, len(self.bands), self.super_tensor.shape[2]])

