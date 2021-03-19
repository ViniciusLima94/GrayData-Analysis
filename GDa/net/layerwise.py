import numpy            as     np
import networkx         as     nx
import igraph           as     ig
import leidenalg
from   joblib                import Parallel, delayed
from   .null_models          import *
from   tqdm                  import tqdm
from   .util                 import instantiate_graph

def compute_nodes_degree(A, mirror=False):
    # Check the dimension
    assert len(A.shape)==3, "The adjacency tensor should be 3D."

    if mirror:
        A = A + np.transpose( A, (1,0,2) )

    node_degree = A.sum(axis=1)

    return node_degree

def compute_nodes_clustering(A, is_weighted=False):  
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
        g               = instantiate_graph(A[:,:,t], is_weighted=is_weighted)
        if is_weighted is True:
            clustering[:,t] = g.transitivity_local_undirected(weights="weight")
        else:
            clustering[:,t] = g.transitivity_local_undirected()

    return np.nan_to_num( clustering )

def compute_nodes_coreness(A, is_weighted=False):
    # Check the dimension
    assert len(A.shape)==3, "The adjacency tensor should be 3D."
    #assert thr != None, "A threshold value should be provided."

    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]
    #  Variable to store node coreness
    coreness  = np.zeros([nC,nt])

    for t in tqdm(range(nt)):
        g               = instantiate_graph(A[:,:,t], is_weighted=is_weighted)
        #g             = instantiate_graph(A[:,:,t], thr=thr)
        coreness[:,t] = g.coreness()

    return coreness

def compute_network_partition(A, is_weighted=False):
    # Check the dimension
    assert len(A.shape)==3, "The adjacency tensor should be 3D."
    #assert thr != None, "A threshold value should be provided."

    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]
    
    #  Save the partitions
    partition = []

    for t in tqdm(range(nt)):
        g               = instantiate_graph(A[:,:,t], is_weighted=is_weighted)
        #g = instantiate_graph(A[:,:,t], thr=thr)
        # Uses leidenalg
        partition.append(leidenalg.find_partition(g, leidenalg.ModularityVertexPartition))

    return partition

def compute_network_modularity(A, is_weighted=False):
    # Check the dimension
    assert len(A.shape)==3, "The adjacency tensor should be 3D."
    #assert thr != None, "A threshold value should be provided."

    #  Number of observations
    nt = A.shape[-1]

    #  Variable to store modularity
    modularity  = np.zeros(nt)

    for t in tqdm(range(nt)):
        #g = instantiate_graph(A[:,:,t], thr=thr)
        g               = instantiate_graph(A[:,:,t], is_weighted=is_weighted)
        modularity[t] = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition).modularity

    return modularity

def compute_nodes_betweenness(A, is_weighted=False):
    # Check the dimension
    assert len(A.shape)==3, "The adjacency tensor should be 3D."

    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]

    betweenness = np.zeros([nC, nt])

    for t in tqdm(range(nt)):
        g               = instantiate_graph(A[:,:,t], is_weighted=is_weighted)
        #g                = instantiate_graph(A[:,:,t], thr=thr)
        if thr is None:
            betweenness[:,t] = g.betweenness(weights="weight")
        else:
            betweenness[:,t] = g.betweenness()

    return betweenness

def compute_allegiance_matrix(A, is_weighted=False):
    # Check the dimension
    assert len(A.shape)==3, "The adjacency tensor should be 3D."
    #assert thr != None, "A threshold value should be provided."

    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]

    #  Find the partitions
    p = compute_network_partition(A, is_weighted=is_weighted)

    T = np.zeros([nC, nC])
    for i in tqdm( range(len(p)) ):
        n_comm = len(p[i])
        for j in range(n_comm):
            grid = np.meshgrid(list(p[i][j]), list(p[i][j]))
            grid = np.reshape(grid, (2, len(list(p[i][j]))**2)).T
            T[grid[:,0], grid[:,1]] += 1
    T = T / len(p)
    np.fill_diagonal(T, 0)

    return T

def null_model_statistics(A, f_name, n_stat, n_rewires=1000, n_jobs=1, seed=0, **kwargs):
    assert len(A.shape)==3, "The adjacency tensor should be 3D."
    #assert thr != None, "A threshold value should be provided."

    def single_estimative(A, f_name, n_rewires, seed, **kwargs):
        #  Create randomized model
        A_null = randomize_edges(A,n_rewires=n_rewires,seed=seed)
        return f_name(A_null, is_weighted=False, **kwargs)
    
    measures = Parallel(n_jobs=n_jobs, backend='loky')(delayed(single_estimative)(A, f_name, n_rewires, seed = i*(seed+100), **kwargs) for i in range(n_stat) )
    return np.array( measures )
