import numpy            as     np
import networkx         as     nx
import igraph           as     ig
import leidenalg
import os
import h5py
import multiprocessing
import itertools
from   sklearn.decomposition import NMF
from   scipy                 import stats
from   tqdm                  import tqdm
from   joblib                import Parallel, delayed

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
            self.super_tensor = np.array( hf.get('supertensor') )
            self.tarray       = np.array( hf.get('tarray') )
            self.freqs        = np.array( hf.get('freqs') )
            self.bands        = np.array( hf.get('bands') )

        if trim_borders == True:
            self.tarray       = self.tarray[wt:-wt] 
            self.super_tensor = self.super_tensor[:,:,:,wt:-wt]

        # Concatenate trials in the super tensor
        self.super_tensor = self.super_tensor.swapaxes(1,2)
        self.super_tensor = self.reshape_observations(self.super_tensor)

    def convert_to_adjacency(self,):
        self.A = np.zeros([self.session_info['nC'], self.session_info['nC'], len(self.bands), self.session_info['nT']*len(self.tarray)]) 
        for p in range(self.session_info['pairs'].shape[0]):
            i, j              = self.session_info['pairs'][p,0], self.session_info['pairs'][p,1]
            self.A[i,j,:,:]   = self.super_tensor[p,:,:]

    def compute_coherence_thresholds(self, q = .80):
        #  Coherence thresholds
        self.coh_thr = np.zeros(len(self.bands)) 
        for i in range(len(self.bands)):
            self.coh_thr[i] = stats.mstats.mquantiles( self.super_tensor[:,i,:].flatten(), prob = q )

        #  self.node_degree = np.zeros([2, self.session_info['nC'], len(self.bands), self.super_tensor.shape[2]])
    def compute_nodes_degree(self, band = 0, thr = None, on_null = False, randomize='edges', seed = 0):
        #  Variable to store node degree
        node_degree = np.zeros([self.session_info['nC'], self.super_tensor.shape[2]])

        if on_null == False:
            A_tmp = self.A[:,:,band,:] + np.transpose( self.A[:,:,band,:], (1,0,2) )
        else:
            self.create_null_model(band = band, thr = thr, randomize=randomize, seed = seed)
            A_tmp = self.A_null

        if thr == None:
            #  self.node_degree[0,:,band,:] = A_tmp.sum(axis=1) 
            node_degree = A_tmp.sum(axis=1) 
        else:
            A_tmp = A_tmp > thr
            #  self.node_degree[1,:,band,:] = A_tmp.sum(axis=1)  
            node_degree = A_tmp.sum(axis=1)  

        return node_degree
    
    def compute_nodes_clustering(self, band = 0, thr = None, use = 'networkx', on_null = False, randomize='edges', seed=0):
        #  Variable to store node clustering
        clustering  = np.zeros([self.session_info['nC'], self.super_tensor.shape[2]])

        if on_null == True:
            self.create_null_model(band = band, randomize=randomize, seed = seed)

        for t in tqdm(range(self.A.shape[3])):
            #  g = self.instantiate_graph(band,t,thr=thr,on_null = on_null)
            if on_null == False:
                g = self.instantiate_graph(self.A[:,:,band,t], thr = thr)
            else:
                g = self.instantiate_graph(self.A_null[:,:,t], thr = thr)
   
            if use == 'networkx':
                clustering[:,t] = list( dict( nx.clustering(g) ).values() )
            elif use=='igraph':
                clustering[:,t] = ig.Graph(self.session_info['nC'], g.edges).transitivity_local_undirected()
        return np.nan_to_num( clustering )

    def compute_nodes_coreness(self, band = 0, thr = None, use='networkx', on_null = False, randomize='edges', seed = 0):
        #  Variable to store coreness
        coreness    = np.zeros([self.session_info['nC'], self.super_tensor.shape[2]])

        if on_null == True:
            self.create_null_model(band = band, randomize=randomize, seed = seed)
 
        for t in tqdm(range(self.A.shape[3])):
            #  g = self.instantiate_graph(band,t,thr=thr,on_null=on_null)
            if on_null == False:
                g = self.instantiate_graph(self.A[:,:,band,t], thr = thr)
            else:
                g = self.instantiate_graph(self.A_null[:,:,t], thr = thr)

            if use=='networkx':
                coreness[:,t] = list( dict( nx.core_number(g) ).values() )
            elif use=='igraph':
                coreness[:,t] = ig.Graph(self.session_info['nC'], g.edges).coreness()
        return coreness

    def compute_network_partition(self, band = 0, thr = None, use='networkx', on_null=False, randomize='edges', seed = 0):
        partition = []
        if on_null == True:
            self.create_null_model(band = band, randomize=randomize, seed = seed)

        for t in tqdm(range(self.A.shape[3])):
            #  g = self.instantiate_graph(band,t,thr=thr, on_null=on_null)
            if on_null == False:
                g = self.instantiate_graph(self.A[:,:,band,t], thr = thr)
            else:
                g = self.instantiate_graph(self.A_null[:,:,t], thr = thr)

            if use=='networkx':
                partition.append(nx.algorithms.community.greedy_modularity_communities(g))
            elif use=='igraph':
                #  This one uses leidenalg
                partition.append(leidenalg.find_partition(ig.Graph(self.session_info['nC'], g.edges), leidenalg.ModularityVertexPartition))
        return partition
    
    def compute_network_modularity(self, band = 0, thr = None, use='networkx', on_null=False, randomize='edges', seed = 0):
        #  Variable to store modularity
        modularity  = np.zeros(self.super_tensor.shape[2])
        if on_null == True:
            self.create_null_model(band = band, randomize=randomize, seed = seed)

        for t in tqdm(range(self.A.shape[3])):
            #  g = self.instantiate_graph(band,t,thr=thr, on_null=on_null)#nx.Graph(self.A[:,:,band,t]>thr)
            if on_null == False:
                g = self.instantiate_graph(self.A[:,:,band,t], thr = thr)
            else:
                g = self.instantiate_graph(self.A_null[:,:,t], thr = thr)

            if use=='networkx':
                comm = nx.algorithms.community.greedy_modularity_communities(g)
                modularity[t] = nx.algorithms.community.modularity(g,comm)
            elif use=='igraph':
                #  This one uses leidenalg
                modularity[t] = leidenalg.find_partition(ig.Graph(self.session_info['nC'], g.edges), leidenalg.ModularityVertexPartition).modularity
        return modularity

    def compute_nodes_betweenness(self, k = None, band = 0, thr = None, use='networkx', on_null=False, randomize='edges', seed = 0):
        betweenness = np.zeros([self.session_info['nC'], self.super_tensor.shape[2]])
        if on_null == True:
            self.create_null_model(band = band, randomize=randomize, seed = seed)

        for t in tqdm(range(self.A.shape[3])):
            #  g = self.instantiate_graph(band,t,thr=thr, on_null=on_null)
            if on_null == False:
                g = self.instantiate_graph(self.A[:,:,band,t], thr = thr)
            else:
                g = self.instantiate_graph(self.A_null[:,:,t], thr = thr)

            if use=='networkx':
                betweenness[:,t] = list( dict( nx.betweenness_centrality(g, k) ).values() )
            elif use=='igraph':
                betweenness[:,t] = ig.Graph(self.session_info['nC'], g.edges).betweenness()
        return betweenness

    #  def compute_null_statistics(self, f_name, n_stat, k = None, band = 0, thr = None, use='networkx', randomize = 'edges', seed = 1, n_jobs=1):
    #      def single_estimative(f_name, k = k, band = band, thr = thr, use=use, randomize = randomize, seed = seed):
    #          self.create_null_model(band = band, randomize=randomize, seed = seed)
    #          if f_name == self.compute_nodes_betweenness:
    #              return f_name(k = k, band = band, thr = thr, use=use, on_null=True, randomize=randomize, seed=seed)
    #          if f_name == self.compute_nodes_degree:
    #              return f_name(band = band, thr = thr, on_null=True, randomize=randomize, seed=seed)
    #          else:
    #              return f_name(band = band, thr = thr, use=use, on_null=True, randomize=randomize, seed=seed)
        
    #      measures = Parallel(n_jobs=n_jobs, backend='loky')(delayed(single_estimative)(f_name, k = k, band = band, thr = thr, use=use, randomize = randomize, seed = i*(seed+100)) for i in range(n_stat) )
    #      return np.array( measures )
    
    def compute_null_statistics(self, f_name, n_stat, band = 0, randomize='edges', n_jobs=1, seed = 0, **kwargs):

        def single_estimative(f_name, band, randomize, seed, **kwargs):
            self.create_null_model(band = band, randomize=randomize, seed = seed)
            return f_name(band = band, randomize = randomize, on_null=True, **kwargs)
        
        measures = Parallel(n_jobs=n_jobs, backend='loky')(delayed(single_estimative)(f_name, band, randomize, seed = i*(seed+100), **kwargs) for i in range(n_stat) )
        return np.array( measures )

    def NMF_decomposition(self, band = 0, k = 2):

        def NMF_decomposition_single(band, k):
            model = NMF(n_components=k, init='random', random_state=0)
            W     = model.fit_transform(self.super_tensor[:,band,:])
            H     = model.components_
        
        Parallel(n_jobs=n_jobs, backend='multiprocessing', timeout=1e6)(delayed(NMF_decomposition_single)(band, k_) for k_ in k )

        return W, H

    #  def instantiate_graph(self, band = 0, observation = 0, thr = None, on_null = False):
    #      if on_null == False:
    #          A_tmp = self.A[:,:,band,:] #+ np.transpose( self.A[:,:,band,:], (1,0,2) )
    #      else:
    #          A_tmp = self.A_null[:,:,band,:]# + np.transpose( self.A_null[:,:,band,:], (1,0,2) )

    #      if thr == None:
    #          #  return nx.Graph(self.A[:,:,band,observation])
    #          return nx.Graph(A_tmp[:,:,observation])
    #      else:
    #          #  return nx.Graph(self.A[:,:,band,observation]>thr)
    #          return nx.Graph(A_tmp[:,:,observation]>thr)

    def compute_allegiance_matrix(self, band=0, thr=None, use='networkx', per_task_stage=True, on_null=False, randomize='edges', seed = 0):

        partitions = self.compute_network_partition(band=band, thr = thr, use = use, on_null=on_null, randomize = randomize, seed = seed)
        if per_task_stage == True:
            T = np.zeros([4, self.session_info['nC'], self.session_info['nC']])
            for k in range(4):
                if k==0: p = np.array( partitions )[self.t_baseline]
                if k==1: p = np.array( partitions )[self.t_cue]
                if k==2: p = np.array( partitions )[self.t_delay]
                if k==3: p = np.array( partitions )[self.t_match]
                for i in tqdm( range(len(p)) ):
                    n_comm = len(p[i])
                    for j in range(n_comm):
                        grid = np.meshgrid(list(p[i][j]), list(p[i][j]))
                        grid = np.reshape(grid, (2, len(list(p[i][j]))**2)).T
                        T[k, grid[:,0], grid[:,1]] += 1
                T[k] = T[k] / len(p)
                np.fill_diagonal(T[k], 0)
        else:
            T = np.zeros([self.session_info['nC'], self.session_info['nC']])
            p = partitions
            for i in tqdm( range(len(p)) ):
                n_comm = len(p[i])
                for j in range(n_comm):
                    grid = np.meshgrid(list(p[i][j]), list(p[i][j]))
                    grid = np.reshape(grid, (2, len(list(p[i][j]))**2)).T
                    T[k, grid[:,0], grid[:,1]] += 1
            T[k] = T[k] / len(p)
            np.fill_diagonal(T[k], 0)

        return T
    
    def instantiate_graph(self, adjacency, thr = None):
        if thr == None:
            #  return nx.Graph(self.A[:,:,band,observation])
            return nx.Graph(adjacency)
        else:
            #  return nx.Graph(self.A[:,:,band,observation]>thr)
            return nx.Graph(adjacency>thr)

    def reshape_trials(self, tensor):
        #  Reshape the tensor to have trials and time as two separeted dimension
        print(len(tensor.shape))
        if len(tensor.shape) == 1:
            aux = tensor.reshape([self.session_info['nT'], len(self.tarray)])
        if len(tensor.shape) == 2:
            aux = tensor.reshape([tensor.shape[0], self.session_info['nT'], len(self.tarray)])
        if len(tensor.shape) == 3:
            aux = tensor.reshape([tensor.shape[0], tensor.shape[1], self.session_info['nT'], len(self.tarray)])
        if len(tensor.shape) == 4:
            aux = tensor.reshape([tensor.shape[0], tensor.shape[1], tensor.shape[2], self.session_info['nT'], len(self.tarray)])
        return aux

    def reshape_observations(self, tensor):
        #  Reshape the tensor to have all the trials concatenated in the same dimension
        if len(tensor.shape) == 2:
            aux = tensor.reshape([self.session_info['nT'] * len(self.tarray)])
        if len(tensor.shape) == 3:
            aux = tensor.reshape([tensor.shape[0], self.session_info['nT'] * len(self.tarray)])
        if len(tensor.shape) == 4:
            aux = tensor.reshape([tensor.shape[0], tensor.shape[1], self.session_info['nT'] * len(self.tarray)])
        if len(tensor.shape) == 5:
            aux = tensor.reshape([tensor.shape[0], tensor.shape[1], tensor.shape[2], self.session_info['nt'] * len(self.tarray)])
        return aux

    def create_null_model(self, band = 0, randomize='edges', thr=None, seed = 0):
        np.random.seed(seed)
        self.A_null = np.zeros_like(self.A[:,:,band,:])
        if randomize=='time':
            idx = np.arange(self.session_info['nT']*len(self.tarray), dtype = int)
            np.random.shuffle(idx)
            self.A_null = ( self.A[:,:,band,:] + np.transpose( self.A[:,:,band,:],  (1,0,2) ) ).copy()
            self.A_null = self.A_null[:,:,idx]
        elif randomize=='edges':
            #  idx = np.arange(self.session_info['nC'], dtype = int)
            if type(thr) == type(None):
                print('A threshold value should be provided')
            else:
                self.A_null = np.empty_like(self.A[:,:,band,:])
                for t in range(self.session_info['nT']*len(self.tarray)):
                    g   = self.instantiate_graph(self.A[:,:,band,t]>thr)
                    G   = ig.Graph(self.session_info['nC'], g.edges)
                    G.rewire()
                    self.A_null[:,:,t] = np.array(list(G.get_adjacency()))#nx.to_numpy_matrix(g_r)

            '''
            self.A_null = ( self.A[:,:,band,:] + np.transpose( self.A[:,:,band,:],  (1,0,2) ) ).copy()
                idx  = np.arange( self.session_info['nC'] )
            idx_ = np.squeeze([np.random.permutation(idx) for i in range(10000)]) 
            rows = np.arange( self.session_info['nC'] ).reshape(-1,1)
            for t in range(self.session_info['nT']*len(self.tarray)):
                choices            = np.random.randint(len(idx_), size=self.session_info['nC'])
                mask               = idx_[choices]
                #  Buting elements of the diagonal back to place
                i = mask == np.arange(self.session_info['nC'])[:, None]
                mask[i] = mask[range(self.session_info['nC']),range(self.session_info['nC'])]
                mask[range(self.session_info['nC']),range(self.session_info['nC'])] = range(self.session_info['nC'])
                self.A_null[:,:,t] = self.A_null[rows,mask,t] 
            '''
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

    def compute_temporal_correlation(self, band = 0, thr = None, tau = 1, on_null = False, randomize = 'edges', seed = 0):
        if on_null == True:
            #  A = self.A_null[:,:,band,:]  
            self.create_null_model(band = band, randomize=randomize, seed = seed)
            A = self.A_null
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

