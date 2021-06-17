import numpy            as     np
import xarray           as     xr
import igraph           as     ig
import leidenalg
from   frites.utils          import parallel_func
from   joblib                import Parallel, delayed
from   .null_models          import *
from   tqdm                  import tqdm
from   .util  import instantiate_graph, _check_inputs, _unwrap_inputs

def compute_nodes_degree(A, mirror=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time), the strength (weighted) degree (binary) of each
    node is computed.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - mirror: If True will mirror the adjacency matrix (should be used if only the upper/lower triangle is given.
    > OUTPUTS:
    - node_degree: A matrix containing the nodes degree with shape (roi,trials,time).
    '''
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A,concat_trials=False)

    if mirror:
        A = A + np.transpose( A, (1,0,2,3) )

    node_degree = A.sum(axis=1)

    # Convert to xarray
    node_degree = xr.DataArray(node_degree, dims=("roi","trials","time"),
                               coords={"roi": roi, "time": time, "trials": trials} )

    return node_degree

def compute_nodes_clustering(A, is_weighted=False, verbose=False):  
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time), the clustering coefficient for each
    node is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    > OUTPUTS:
    - clustering: A matrix containing the nodes clustering with shape (roi,trials,time).
    '''
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A,concat_trials=True)

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

    # Unstack trials and time
    clustering = clustering.reshape( (len(roi),len(trials),len(time)) )
    # Convert to xarray
    clustering = xr.DataArray(np.nan_to_num(clustering), dims=("roi","trials","time"),
                              coords={"roi": roi, "time": time, "trials": trials} )

    return clustering

def compute_nodes_coreness(A, is_weighted=False, verbose=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time), the coreness for each
    node is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    > OUTPUTS:
    - coreness: A matrix containing the nodes coreness with shape (roi,trials,time).
    '''
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A,concat_trials=True)

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

    # Unstack trials and time
    coreness = coreness.reshape( (len(roi),len(trials),len(time)) )
    # Convert to xarray
    coreness = xr.DataArray(coreness, dims=("roi","trials","time"),
                              coords={"roi": roi, "time": time, "trials": trials} )

    return coreness

def compute_nodes_betweenness(A, is_weighted=False, verbose=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time), the betweenness for each
    node is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    > OUTPUTS:
    - betweenness: A matrix containing the nodes betweenness with shape (roi,time).
    '''
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A,concat_trials=True)

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

    # Unstack trials and time
    betweenness = betweenness.reshape( (len(roi),len(trials),len(time)) )
    # Convert to xarray
    betweenness = xr.DataArray(betweenness, dims=("roi","trials","time"),
                              coords={"roi": roi, "time": time, "trials": trials} )

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
    - partition: A list with the all the partition found for each layer of the 
    matrix (for each observation or trials,time if flatten is False).
    '''
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A,concat_trials=True)

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

    # Reshape back to trials and time
    partition = np.reshape(partition, (len(trials),len(time)))
    # Conversion to xarray
    partition = xr.DataArray(partition, dims=("trials","time"),
                             coords={"trials":trials,"time":time})

    return partition

def compute_network_modularity(A, is_weighted=False, verbose=False):
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
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A,concat_trials=True)

    #  Number of observations
    nt = A.shape[-1]

    #  Variable to store modularity
    modularity  = np.zeros(nt)

    itr = range(nt)
    for t in (tqdm(itr) if verbose else itr):
        g               = instantiate_graph(A[:,:,t], is_weighted=is_weighted)
        modularity[t] = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition).modularity

    # Unstack trials and time 
    modularity = modularity.reshape( (len(trials),len(time)) )
    # Convert to xarray
    modularity = xr.DataArray(modularity, dims=("trials","time"),
                              coords={"time": time, "trials": trials} )

    return modularity

def compute_allegiance_matrix(A, concat=False, is_weighted=False, verbose=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time), the allegiance matrix for  
    the whole period provided will be computed.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - concat: Wheter trials are concatenated or not.
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    > OUTPUTS:
    - T: The allegiance matrix between all nodes with shape (roi, roi)
    '''
    #  Find the partitions
    if verbose: print("Finding network partitions.\n")
    p = compute_network_partition(A, is_weighted=is_weighted, verbose=verbose)

    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A,concat_trials=True)
    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]

    T = np.zeros([nC, nC])

    if verbose: print("Computing allegiance matrix.\n")
    itr = range( len(p) )
    for i in (tqdm(itr) if verbose else itr):
        n_comm = len(p[i])
        for j in range(n_comm):
            grid = np.meshgrid(list(p[i][j]), list(p[i][j]))
            grid = np.reshape(grid, (2, len(list(p[i][j]))**2)).T
            T[grid[:,0], grid[:,1]] += 1
    T = T / nt 
    np.fill_diagonal(T, 0)

    # Converting to xarray
    T = xr.DataArray(T, dims=("roi_1","roi_2"),
                     coords={"roi_1":roi, "roi_2": roi})
    return T

def windowed_allegiance_matrix(A, times=None, is_weighted=False, verbose=False, win_args=None, n_jobs=1):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time), the windowed allegiance matrix.
    For each window the observations are concatenated for all trials and then the allegiance matrix is estimated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - times: Time array to construct the windows.
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    - win_args: Dict. with arguments to be passed to define_windows (for more details see frites.conn.conn_sliding_windows)
    - n_jobs: Number of jobs to use when parallelizing over windows.
    > OUTPUTS:
    - T: The allegiance matrix between all nodes with shape (len(window), roi, roi)
    '''
    from frites.conn.conn_sliding_windows import define_windows

    assert isinstance(win_args, dict)
    assert isinstance(times, (list, np.ndarray, xr.DataArray))

    win, t_win = define_windows(times, **win_args)
    def _for_win(w):
        T=compute_allegiance_matrix(A.isel(time=slice(w[0],w[1])),
                                    is_weighted=is_weighted, verbose=verbose)
        return T

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_win, n_jobs=n_jobs, verbose=verbose,
        total=len(win))
    # compute the single trial coherence
    T = parallel(p_fun(w) for w in win)

    # Concatenating
    T = xr.concat(T, dim="time")
    # Ordering dimensions
    T = T.transpose("roi_1","roi_2","time")
    # Assign time axis
    T = T.assign_coords({"time":t_win})
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
