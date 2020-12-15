import numpy            as     np
import networkx         as     nx
import igraph           as     ig
import leidenalg
from   tqdm                  import tqdm
from   .util                 import instantiate_graph

def compute_nodes_degree(A, thr=None):
    # Check the dimension
    assert len(A.shape)==3, "The adjacency tensor should be 3D."

    A = A + np.transpose( A, (1,0,2) )

    # Compute the node degree
    if thr is not None:
        A_tmp = A > thr
        node_degree = A_tmp.sum(axis=1)  
    else:
        node_degree = A.sum(axis=1)  

    return node_degree

def compute_nodes_clustering(A, thr = None):  
    # Check the dimension
    assert len(A.shape)==3, "The adjacency tensor should be 3D."
    #  assert thr != None, "A threshold value should be provided."

    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]
    #  Variable to store node clustering
    clustering  = np.zeros([nC,nt])

    for t in tqdm(range(nt)):
        #  Instantiate graph
        g               = instantiate_graph(A[:,:,t], thr=thr)
        if thr is None:
            clustering[:,t] = g.transitivity_local_undirected(weights="weight")
        else:
            clustering[:,t] = g.transitivity_local_undirected()

    return np.nan_to_num( clustering )

def compute_nodes_coreness(self, band = 0, thr = None, use='networkx', on_null = False, randomize='edges'):
    #  Variable to store coreness
    coreness    = np.zeros([self.session_info['nC'], self.super_tensor.shape[2]])

    #  if on_null == True:
    #      self.create_null_model(band = band, thr = thr, randomize=randomize, seed = seed)

    for t in tqdm(range(self.A.shape[3])):
        #  g = self.instantiate_graph(band,t,thr=thr,on_null=on_null)
        if on_null == False:
            g = self.instantiate_graph(self.A[:,:,band,t], thr = thr)
        else:
            try:
                g = self.instantiate_graph(self.A_null[randomize][str(band)][:,:,t], thr = thr)
            except (AttributeError, KeyError):
                print('Null model for band ' + str(band) + ' not created yet.')

        if use=='networkx':
            coreness[:,t] = list( dict( nx.core_number(g) ).values() )
        elif use=='igraph':
            coreness[:,t] = ig.Graph(self.session_info['nC'], g.edges).coreness()
    return coreness

def compute_network_partition(self, band = 0, thr = None, use='networkx', on_null=False, randomize='edges'):
    partition = []
    #  if on_null == True:
    #      self.create_null_model(band = band, randomize=randomize, seed = seed)

    for t in tqdm(range(self.A.shape[3])):
        #  g = self.instantiate_graph(band,t,thr=thr, on_null=on_null)
        if on_null == False:
            g = self.instantiate_graph(self.A[:,:,band,t], thr = thr)
        else:
            try:
                g = self.instantiate_graph(self.A_null[randomize][str(band)][:,:,t], thr = thr)
            except (AttributeError, KeyError):
                print('Null model for band ' + str(band) + ' not created yet.')

        if use=='networkx':
            partition.append(nx.algorithms.community.greedy_modularity_communities(g))
        elif use=='igraph':
            #  This one uses leidenalg
            partition.append(leidenalg.find_partition(ig.Graph(self.session_info['nC'], g.edges), leidenalg.ModularityVertexPartition))
    return partition

def compute_network_modularity(self, band = 0, thr = None, use='networkx', on_null=False, randomize='edges', seed = 0):
    #  Variable to store modularity
    modularity  = np.zeros(self.super_tensor.shape[2])
    #  if on_null == True:
    #      self.create_null_model(band = band, thr = thr, randomize=randomize, seed = seed)

    for t in tqdm(range(self.A.shape[3])):
        #  g = self.instantiate_graph(band,t,thr=thr, on_null=on_null)#nx.Graph(self.A[:,:,band,t]>thr)
        if on_null == False:
            g = self.instantiate_graph(self.A[:,:,band,t], thr = thr)
        else:
            try:
                g = self.instantiate_graph(self.A_null[randomize][str(band)][:,:,t], thr = thr)
            except (AttributeError, KeyError):
                print('Null model for band ' + str(band) + ' not created yet.')

        if use=='networkx':
            comm = nx.algorithms.community.greedy_modularity_communities(g)
            modularity[t] = nx.algorithms.community.modularity(g,comm)
        elif use=='igraph':
            #  This one uses leidenalg
            modularity[t] = leidenalg.find_partition(ig.Graph(self.session_info['nC'], g.edges), leidenalg.ModularityVertexPartition).modularity
    return modularity

def compute_nodes_betweenness(self, k = None, band = 0, thr = None, use='networkx', on_null=False, randomize='edges', seed = 0):
    betweenness = np.zeros([self.session_info['nC'], self.super_tensor.shape[2]])
    #  if on_null == True:
    #      self.create_null_model(band = band, randomize=randomize, seed = seed)

    for t in tqdm(range(self.A.shape[3])):
        #  g = self.instantiate_graph(band,t,thr=thr, on_null=on_null)
        if on_null == False:
            g = self.instantiate_graph(self.A[:,:,band,t], thr = thr)
        else:
            try:
                g = self.instantiate_graph(self.A_null[randomize][str(band)][:,:,t], thr = thr)
            except (AttributeError, KeyError):
                print('Null model for band ' + str(band) + ' not created yet.')

        if use=='networkx':
            betweenness[:,t] = list( dict( nx.betweenness_centrality(g, k) ).values() )
        elif use=='igraph':
            betweenness[:,t] = ig.Graph(self.session_info['nC'], g.edges).betweenness()
    return betweenness
