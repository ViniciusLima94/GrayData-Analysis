import numpy            as     np
import xarray           as     xr
import igraph           as     ig
import leidenalg
from   frites.utils          import parallel_func
from   joblib                import Parallel, delayed
from   .null_models          import *
from   tqdm                  import tqdm
from   .util                 import instantiate_graph

def _check_inputs(array, dims):
    r'''
    Check the input type and size.
    > INPUT:
    - array: The data array.
    - dims: The number of dimensions the array should have.
    '''
    assert isinstance(dims, int)
    assert isinstance(array, (np.ndarray, xr.DataArray))
    assert len(array.shape)==dims, f"The adjacency tensor should be {dims}D."

def compute_nodes_degree(A, mirror=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time), the strength (weighted) degree (binary) of each
    node is computed.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - mirror: If True will mirror the adjacency matrix (should be used if only the upper/lower triangle is given.
    > OUTPUTS:
    - node_degree: A matrix containing the nodes degree with shape (roi,roi,trials,time).
    '''
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray): A = A.values

    if mirror:
        A = A + np.transpose( A, (1,0,2,3) )

    node_degree = A.sum(axis=1)

    return node_degree

def compute_nodes_clustering(A, is_weighted=False, verbose=False):  
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time), the clustering coefficient for each
    node is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    > OUTPUTS:
    - clustering: A matrix containing the nodes clustering with shape (roi,time).
    '''
    # Check inputs
    _check_inputs(A, 3)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray): A = A.values

    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]
    #  Variable to store node clustering
    clustering  = np.zeros([nC,nt])

    itr = range(nt)
    for t in (tqdm(itr) if verbose else itr):
        #  Instantiate graph
        g               = instantiate_graph(A[:,:,t], is_weighted=is_weighted)
        if is_weighted is True:
            clustering[:,t] = g.transitivity_local_undirected(weights="weight")
        else:
            clustering[:,t] = g.transitivity_local_undirected()

    return np.nan_to_num( clustering )

def compute_nodes_coreness(A, is_weighted=False, verbose=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time), the coreness for each
    node is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    > OUTPUTS:
    - coreness: A matrix containing the nodes coreness with shape (roi,time).
    '''
    # Check inputs
    _check_inputs(A, 3)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray): A = A.values

    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]
    #  Variable to store node coreness
    coreness  = np.zeros([nC,nt])

    itr = range(nt)
    for t in (tqdm(itr) if verbose else itr):
        g               = instantiate_graph(A[:,:,t], is_weighted=is_weighted)
        coreness[:,t] = g.coreness()

    return coreness

def compute_nodes_betweenness(A, is_weighted=False, verbose=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time), the betweenness for each
    node is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    > OUTPUTS:
    - betweenness: A matrix containing the nodes betweenness with shape (roi,time).
    '''
    # Check inputs
    _check_inputs(A, 3)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray): A = A.values

    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]

    betweenness = np.zeros([nC, nt])

    itr = range(nt)
    for t in (tqdm(itr) if verbose else itr):
        g               = instantiate_graph(A[:,:,t], is_weighted=is_weighted)
        if is_weighted:
            betweenness[:,t] = g.betweenness(weights="weight")
        else:
            betweenness[:,t] = g.betweenness()

    return betweenness

def compute_network_partition(A, is_weighted=False, verbose=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time), the network partition for each
    node is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    > OUTPUTS:
    - partition: A list with the all the partition found for each layer of the matrix (for each observation).
    '''
    # Check inputs
    _check_inputs(A, 3)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray): A = A.values

    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]
    
    #  Save the partitions
    partition = []

    itr = range(nt)
    for t in (tqdm(itr) if verbose else itr):
        g               = instantiate_graph(A[:,:,t], is_weighted=is_weighted)
        # Uses leidenalg
        partition += [leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)]

    return partition

def compute_network_modularity(A, is_weighted=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time), the modularity of the  
    network for each layer/time is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    > OUTPUTS:
    - modularity: Modularity for each time frame of the temporal network.
    '''
    # Check inputs
    _check_inputs(A, 3)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray): A = A.values

    #  Number of observations
    nt = A.shape[-1]

    #  Variable to store modularity
    modularity  = np.zeros(nt)

    itr = range(nt)
    for t in (tqdm(itr) if verbose else itr):
        g               = instantiate_graph(A[:,:,t], is_weighted=is_weighted)
        modularity[t] = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition).modularity

    return modularity

def compute_allegiance_matrix(A, is_weighted=False, verbose=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time), the allegiance matrix for  
    the whole period provided will be computed.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    > OUTPUTS:
    - T: The allegiance matrix between all nodes with shape (roi, roi)
    '''
    # Check inputs
    _check_inputs(A, 3)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray): A = A.values

    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]

    #  Find the partitions
    p = compute_network_partition(A, is_weighted=is_weighted)

    T = np.zeros([nC, nC])

    itr = range( len(p) )
    for i in (tqdm(itr) if verbose else itr):
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
