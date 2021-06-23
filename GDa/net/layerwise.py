import numpy            as     np
import xarray           as     xr
import igraph           as     ig
import leidenalg
from   frites.utils          import parallel_func
from   joblib                import Parallel, delayed
from   .null_models          import *
from   tqdm                  import tqdm
from   .util                 import instantiate_graph, _check_inputs, _unwrap_inputs, _reshape_list

_DEFAULT_TYPE = np.float32

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
    node_degree = xr.DataArray(node_degree.astype(_DEFAULT_TYPE), dims=("roi","trials","time"),
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
    clustering = xr.DataArray(np.nan_to_num(clustering).astype(_DEFAULT_TYPE), dims=("roi","trials","time"),
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
    coreness = xr.DataArray(coreness.astype(_DEFAULT_TYPE), dims=("roi","trials","time"),
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
    betweenness = xr.DataArray(betweenness.astype(_DEFAULT_TYPE), dims=("roi","trials","time"),
                              coords={"roi": roi, "time": time, "trials": trials} )

    return betweenness

def compute_network_partition(A,  kw_leiden={}, is_weighted=False, verbose=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time), the network partition for each
    node is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - kw_leiden: Parameters to be passed to leindelalg (for frther info see: https://leidenalg.readthedocs.io/en/stable/reference.html)
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
    #  partition = []
    partition = np.empty(nt, dtype=object)

    itr = range(nt)
    for t in (tqdm(itr) if verbose else itr):
        g          = instantiate_graph(A[:,:,t], is_weighted=is_weighted)
        # Uses leidenalg
        #  partition += [leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)]
        partition[t] = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, **kw_leiden)

    # Reshape back to trials and time
    partition = np.reshape(partition, (len(trials),len(time)))
    # Reshaping like this because for some reason the above was given error sometimes.
    #  partition = _reshape_list(partition, (len(trials),len(time)), leidenalg.VertexPartition.ModularityVertexPartition)
    # Conversion to xarray
    partition = xr.DataArray(partition, dims=("trials","time"),
                             coords={"trials":trials,"time":time})

    return partition

def compute_network_modularity(A, kw_leiden={}, is_weighted=False, verbose=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time), the modularity of the  
    network for each layer/time is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - kw_leiden: Parameters to be passed to leindelalg (for frther info see: https://leidenalg.readthedocs.io/en/stable/reference.html)
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    > OUTPUTS:
    - modularity: Modularity for each time frame of the temporal network.
    '''
    # Check inputs
    #_check_inputs(A, 4)
    # Get values in case it is an xarray
    #A, roi, trials, time = _unwrap_inputs(A,concat_trials=True)

    #  Number of observations
    #nt = A.shape[-1]
    # Finding parititions
    if verbose: print("Finding network partitions.\n")
    partition = compute_network_partition(A, kw_leiden, is_weighted=is_weighted, verbose=verbose)
    # Getting dimension arrays
    trials, time = partition.trials.values, partition.time.values
    nt           = len(trials)*len(time)
    # Stack paritions 
    partition    = partition.stack(observations=("trials","time"))

    #  Variable to store modularity
    modularity  = np.zeros(nt)

    if verbose: print("Computing modularity.\n")
    itr = range(nt)
    for t in (tqdm(itr) if verbose else itr):
        #  g               = instantiate_graph(A[:,:,t], is_weighted=is_weighted)
        #  modularity[t] = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition).modularity
        modularity[t] = partition.values[t].modularity

    # Unstack trials and time 
    modularity = modularity.reshape( (len(trials),len(time)) )
    # Convert to xarray
    modularity = xr.DataArray(modularity.astype(_DEFAULT_TYPE), dims=("trials","time"),
                              coords={"time": time, "trials": trials} )
    return modularity

def compute_allegiance_matrix(A, kw_leiden={}, concat=False, is_weighted=False, verbose=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time), the allegiance matrix for  
    the whole period provided will be computed.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - kw_leiden: Parameters to be passed to leindelalg (for frther info see: https://leidenalg.readthedocs.io/en/stable/reference.html)
    - concat: Wheter trials are concatenated or not.
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    > OUTPUTS:
    - T: The allegiance matrix between all nodes with shape (roi, roi)
    '''
    #  Find the partitions
    if verbose: print("Finding network partitions.\n")
    p = compute_network_partition(A, kw_leiden, is_weighted=is_weighted, verbose=verbose)
    # Getting dimension arrays
    trials, time = p.trials.values, p.time.values
    # Total number of observations
    nt           = len(trials)*len(time)
    # Stack paritions 
    p            = p.stack(observations=("trials","time"))
    # Number of ROI
    nC           = A.shape[0]

    if isinstance(A, xr.DataArray):
        roi = A.roi_1.values
    else:
        roi = np.arange(nC, dtype=int)

    T = np.zeros([nC, nC])

    itr = range( nt )
    for i in (tqdm(itr) if verbose else itr):
        p_lst  = list(p.values[i])
        n_comm = len(p_lst)
        for j in range(n_comm):
            grid = np.meshgrid(p_lst[j],p_lst[j])
            grid = np.reshape(grid, (2, len(p_lst[j])**2)).T
            T[grid[:,0],grid[:,1]] += 1
    T = T / nt
    np.fill_diagonal(T , 0)

    # Converting to xarray
    T = xr.DataArray(T.astype(_DEFAULT_TYPE), dims=("roi_1","roi_2"),
                     coords={"roi_1":roi, "roi_2": roi})
    return T

def windowed_allegiance_matrix(A, times=None, is_weighted=False, verbose=False, kw_leiden=None, win_args=None, n_jobs=1):
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
    - T: The allegiance matrix between all nodes with shape (roi, roi, trials, time)
    '''
    from frites.conn.conn_sliding_windows import define_windows

    assert isinstance(win_args, dict)
    assert isinstance(A, xr.DataArray)
    assert ('time' in A.dims) and ('trials' in A.dims) and ('roi_1' in A.dims) and ('roi_2' in A.dims)

    # Number of regions
    nC         = A.shape[0]
    # ROIs
    roi        = A.roi_1.values
    # Define windows
    win, t_win = define_windows(times, **win_args)
    # For a given trial computes windowed allegiance
    def _for_win(trial, win):
        T = xr.DataArray(np.zeros((nC,nC,len(win))), 
                         dims=("roi_1","roi_2","time"),
                         coords={"roi_1":roi, "roi_2": roi, "time":t_win})
        for i_w, w in enumerate(win):
            T[...,i_w]=compute_allegiance_matrix(A.isel(trials=[trial],time=slice(w[0],w[1])),
                                                 kw_leiden, is_weighted=is_weighted, verbose=verbose)
        return T

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_win, n_jobs=n_jobs, verbose=verbose,
        total=A.shape[2])
    # compute the single trial coherence
    T = parallel(p_fun(trial,win) for trial in range(A.shape[2]))
    # Concatenating
    T = xr.concat(T.astype(_DEFAULT_TYPE), dim="trials")
    # Ordering dimensions
    T = T.transpose("roi_1","roi_2","trials","time")
    # Assign time axis
    T = T.assign_coords({"trials":A.trials.values})
    return T

def null_model_statistics(A, f_name, n_stat, n_rewires=1000, seed=0, n_jobs=1,  **kwargs):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time), compute the null-statistic
    of a given measurement for different repetitions/seeds.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - f_name: The name of the function of wich the null-statistic should be computed.
    - n_stat: The number of different random seeds to use to compute the null-statistic.
    - seed: Initial seed to set other seeds.
    - n_rewires: The number of rewires to be applied to the binary adjacency matrix,
    - n_jobs: Number of jobs to use when parallelizing over windows.
    > OUTPUTS:
    - T: The allegiance matrix between all nodes with shape (roi, roi, trials, time)
    '''

    assert f_name.__name__ in ['compute_nodes_degree','compute_nodes_clustering','compute_nodes_coreness','compute_nodes_betweenness','compute_network_modularity']

    # Compute the null statistics for a given seed
    def _single_estimative(A, f_name, n_rewires, seed, **kwargs):
        #  Create randomized model
        A_null = randomize_edges(A,n_rewires=n_rewires,seed=seed)
        return f_name(A_null, **kwargs)

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _single_estimative, n_jobs=n_jobs, verbose=False,
        total=n_stat)
    # compute the single trial coherence
    measures = parallel(p_fun(A,f_name,n_rewires,i*(seed+100),**kwargs) for i in range(n_stat))

    # Converting to xarray
    measures = xr.concat(measures, dim='seeds')
    
    return measures
    
    #  measures = Parallel(n_jobs=n_jobs, backend='loky')(delayed(single_estimative)(A, f_name, n_rewires, seed = i*(seed+100), **kwargs) for i in range(n_stat) )
    #  return np.array( measures )
