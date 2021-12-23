import numpy as np
import xarray as xr
import brainconn as bc
import leidenalg
from frites.utils import parallel_func
from tqdm import tqdm
from .util import (instantiate_graph, _check_inputs,
                   _unwrap_inputs, _is_binary, MODquality)

_DEFAULT_TYPE = np.float32


def compute_nodes_degree(A, mirror=False):
    """
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time),
    the strength (weighted) degree (binary) of each
    node is computed.

    Parameters
    ----------
    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    mirror: bool | False
        If True will mirror the adjacency matrix (should be used if only the
        upper/lower triangle is given).

    Returns
    -------
    node_degree: array_like
        A matrix containing the nodes degree with shape (roi,trials,time).
    """

    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A, concat_trials=False)

    if mirror:
        A = A + np.transpose(A, (1, 0, 2, 3))

    node_degree = A.sum(axis=1)

    # Convert to xarray
    node_degree = xr.DataArray(node_degree.astype(_DEFAULT_TYPE),
                               dims=("roi", "trials", "times"),
                               coords={"roi": roi,
                                       "times": time,
                                       "trials": trials})

    return node_degree


def compute_nodes_clustering(A, verbose=False, backend='igraph', n_jobs=1):
    """
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time),
    the clustering coefficient for each node is computed for all the
    trials concatenated.

    Parameters
    ----------
    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    backend: string | "igraph"
        Wheter to use igraph or brainconn package.
    n_jobs: int | 1
        Number of jobs to use when parallelizing in observations.

    Returns
    -------
    clustering: array_like
        A matrix containing the nodes clustering with shape (roi,trials,time).
    """
    assert backend in ['igraph', 'brainconn']
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A, concat_trials=True)
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)
    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]
    #  Variable to store node clustering
    clustering = np.zeros([nC, nt])

    # Compute for a single observation
    def _for_frame(t):
        #  Instantiate graph
        if is_weighted:
            if backend == 'igraph':
                g = instantiate_graph(A[..., t], is_weighted=is_weighted)
                clustering = g.transitivity_local_undirected(weights="weight")
            elif backend == 'brainconn':
                clustering = bc.clustering.clustering_coef_wu(A[..., t])
        else:
            if backend == 'igraph':
                g = instantiate_graph(A[..., t], is_weighted=is_weighted)
                clustering = g.transitivity_local_undirected()
            elif backend == 'brainconn':
                clustering = bc.clustering.clustering_coef_bu(A[..., t])
        return clustering

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_frame, n_jobs=n_jobs, verbose=verbose,
        total=nt)
    # Compute the single trial coherence
    clustering = parallel(p_fun(t) for t in range(nt))
    # Convert to numpy array
    clustering = np.asarray(clustering).T

    # Unstack trials and time
    clustering = clustering.reshape((len(roi), len(trials), len(time)))
    # Convert to xarray
    clustering = xr.DataArray(np.nan_to_num(clustering).astype(_DEFAULT_TYPE),
                              dims=("roi", "trials", "times"),
                              coords={"roi": roi,
                                      "times": time,
                                      "trials": trials})

    return clustering


def compute_nodes_coreness(A, verbose=False, n_jobs=1):
    """
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time),
    the coreness for each node is computed for all the trials concatenated.

    Parameters
    ----------
    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    backend: string | "igraph"
        Wheter to use igraph or brainconn package.
    n_jobs: int | 1
        Number of jobs to use when parallelizing in observations.

    Returns
    -------
    coreness: array_like
        A matrix containing the nodes coreness with shape (roi,trials,time).
    """

    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A, concat_trials=True)
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)
    #  Number of observations
    nt = A.shape[-1]

    # Compute for a single observation
    def _for_frame(t):
        g = instantiate_graph(A[..., t], is_weighted=is_weighted)
        coreness = g.coreness()
        return coreness

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_frame, n_jobs=n_jobs, verbose=verbose,
        total=nt)
    # Compute the single trial coherence
    coreness = parallel(p_fun(t) for t in range(nt))
    # Convert to numpy array
    coreness = np.asarray(coreness).T

    # Unstack trials and time
    coreness = coreness.reshape((len(roi), len(trials), len(time)))
    # Convert to xarray
    coreness = xr.DataArray(coreness.astype(_DEFAULT_TYPE),
                            dims=("roi", "trials", "times"),
                            coords={"roi": roi,
                                    "times": time,
                                    "trials": trials})

    return coreness


def compute_nodes_coreness_bc(A, delta=1, return_degree=False,
                              verbose=False, n_jobs=1):
    """
    The same as 'compute_nodes_coreness' but based on brainconnectivity
    toolbox method can be either for binary or weighted undirected graphs.
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time),
    the coreness for each node is computed for all the trials concatenated.

    Parameters
    ----------
    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    delta: float | 1
        Increment of the core size.
    return_degree: bool | False
        Return the strength/degree of the node if True.
    n_jobs: int | 1
        Number of jobs to use when parallelizing in observations.

    Returns
    -------
    coreness: array_like
        A matrix containing the nodes coreness with shape (roi,trials,time).
    """

    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A, concat_trials=True)
    #  Number of observations
    nt = A.shape[-1]
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)

    # If it is binary use k-core otherwise use s-core
    if is_weighted:
        core_func = bc.core.score_wu
    else:
        core_func = bc.core.kcore_bu

    ##################################################################
    # Computes nodes' k-coreness
    #################################################################
    def _nodes_core(A):
        # Number of nodes
        n_nodes = len(A)
        # Initial coreness
        k = 0
        # Store each node's coreness
        k_core = np.zeros(n_nodes)
        # Iterate until get a disconnected graph
        while True:
            # Get coreness matrix and level of k-core
            C, kn = core_func(A, k)
            if kn == 0:
                break
            # Assigns coreness level to nodes
            s = C.sum(1)
            idx = s > 0
            if return_degree:
                k_core[idx] = s[idx]
            else:
                k_core[idx] = k
            k += delta
        return k_core

    # Compute for a single observation
    def _for_frame(t):
        coreness = _nodes_core(A[..., t])
        return coreness

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_frame, n_jobs=n_jobs, verbose=verbose,
        total=nt)
    # Compute the single trial coherence
    coreness = parallel(p_fun(t) for t in range(nt))
    # Convert to numpy array
    coreness = np.asarray(coreness).T

    # Unstack trials and time
    coreness = coreness.reshape((len(roi), len(trials), len(time)))
    # Convert to xarray
    coreness = xr.DataArray(coreness.astype(_DEFAULT_TYPE),
                            dims=("roi", "trials", "times"),
                            coords={"roi": roi,
                                    "times": time,
                                    "trials": trials})

    return coreness


def compute_nodes_efficiency(A, verbose=False, n_jobs=1):
    """
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time),
    the efficiency for each node is computed for all the trials concatenated.

    Parameters
    ----------
    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    n_jobs: int | 1
        Number of jobs to use when parallelizing in observations.

    Returns
    -------
    coreness: array_like
        A matrix containing the nodes coreness with shape (roi,trials,time).
    """

    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A, concat_trials=True)
    #  Number of observations
    nt = A.shape[-1]
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)

    # If it is binary use k-core otherwise use s-core
    if is_weighted:
        eff_func = bc.distance.efficiency_wei
    else:
        eff_func = bc.distance.efficiency_bin

    ##################################################################
    # Computes nodes' efficiency
    #################################################################

    # Compute for a single observation
    def _for_frame(t):
        _, eff = eff_func(A[..., t], local=True)
        return eff

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_frame, n_jobs=n_jobs, verbose=verbose,
        total=nt)
    # Compute the single trial coherence
    eff = parallel(p_fun(t) for t in range(nt))
    # Convert to numpy array
    eff = np.asarray(eff).T

    # Unstack trials and time
    eff = eff.reshape((len(roi), len(trials), len(time)))
    # Convert to xarray
    eff = xr.DataArray(eff.astype(_DEFAULT_TYPE),
                       dims=("roi", "trials", "times"),
                       coords={"roi": roi,
                               "times": time,
                               "trials": trials})

    return eff


def compute_nodes_betweenness(A, verbose=False, backend='igraph', n_jobs=1):
    """
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time),
    the betweenness for each node is computed for all the trials concatenated.

    Parameters
    ----------
    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    backend: string | "igraph"
        Wheter to use igraph or brainconn package.
    n_jobs: int | 1
        Number of jobs to use when parallelizing in observations.

    Returns
    -------
    betweenness: array_like
        A matrix containing the nodes betweenness with shape (roi,time).
    """

    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A, concat_trials=True)
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)
    #  Number of observations
    nt = A.shape[-1]

    # Compute for a single observation
    def _for_frame(t):
        if is_weighted:
            if backend == 'igraph':
                g = instantiate_graph(A[..., t], is_weighted=is_weighted)
                betweenness = g.betweenness(weights="weight")
            elif backend == 'brainconn':
                betweenness = bc.centrality.betweenness_wei(A[..., t])
        else:
            if backend == 'igraph':
                g = instantiate_graph(A[..., t], is_weighted=is_weighted)
                betweenness = g.betweenness()
            elif backend == 'brainconn':
                betweenness = bc.centrality.betweenness_bin(A[..., t])
        return betweenness

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_frame, n_jobs=n_jobs, verbose=verbose,
        total=nt)
    # Compute the single trial coherence
    betweenness = parallel(p_fun(t) for t in range(nt))
    # Convert to numpy array
    betweenness = np.asarray(betweenness).T

    # Unstack trials and time
    betweenness = betweenness.reshape((len(roi), len(trials), len(time)))
    # Convert to xarray
    betweenness = xr.DataArray(betweenness.astype(_DEFAULT_TYPE),
                               dims=("roi", "trials", "times"),
                               coords={"roi": roi,
                                       "times": time,
                                       "trials": trials})

    return betweenness


def compute_network_partition(A,  kw_louvain={}, kw_leiden={},
                              backend='igraph', n_jobs=1, verbose=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time),
    the network partition for each node is computed for all
    the trials concatenated.

    Parameters
    ----------
    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    kw_louvain: dict | {}
        Parameters to be passed to louvain alg from BrainConnectivity toolbox
    kw_leiden: dict | {}
        Parameters to be passed to leindelalg
        https://leidenalg.readthedocs.io/en/stable/reference.html
    backend: string | "igraph"
        Wheter to use igraph or brainconn package.
    n_jobs: int | 1
        Number of jobs to use when parallelizing in observations.

    Returns
    -------
    partition:
        A list with the all the partition found for each layer of the
    '''

    assert backend in ['igraph', 'brainconn']

    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A, concat_trials=True)
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)
    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]

    # Using igraph
    if backend == 'igraph':
        #  Save the partitions
        # Store nodes' membership
        partition = np.zeros((nC, nt))
        # Store network modularity
        modularity = np.zeros(nt)

        itr = range(nt)
        for t in (tqdm(itr) if verbose else itr):
            g = instantiate_graph(A[..., t], is_weighted=is_weighted)
            # Uses leidenalg
            if is_weighted:
                weights = "weight"
            else:
                weights = None

            optimizer = leidenalg.ModularityVertexPartition
            # Find partitions
            partition[:, t] = leidenalg.find_partition(
                g, optimizer, weights=weights, **kw_leiden).membership
            # Compute modularity
            modularity[t] = MODquality(A[..., t], partition[:, t], 1)

    # Using brainconn
    elif backend == 'brainconn':

        # Check if the network have negative weights
        has_negative_weights = A.min() < 0

        if has_negative_weights:
            mod_func = bc.modularity.modularity_louvain_und_sign
        else:
            mod_func = bc.modularity.modularity_louvain_und

        def _for_frame(t):
            partition, modularity = mod_func(A[..., t], **kw_louvain)
            #  return partition-1, modularity
            return np.concatenate((partition-1, [modularity]))

        # define the function to compute in parallel
        parallel, p_fun = parallel_func(
            _for_frame, n_jobs=n_jobs, verbose=verbose,
            total=nt)

        # Compute the single trial coherence
        out = np.squeeze(parallel(p_fun(t) for t in range(nt)))
        partition, modularity = np.asarray(
            out[:, :-1]).T, np.asarray(out[:, -1])

    # Reshape partition and modularity back to trials and time
    partition = np.reshape(partition, (nC, len(trials), len(time)))
    # Conversion to xarray
    partition = xr.DataArray(partition.astype(int),
                             dims=("roi", "trials", "times"),
                             coords={"roi": roi,
                                     "trials": trials,
                                     "times": time})

    # Unstack trials and time
    modularity = modularity.reshape((len(trials), len(time)))
    # Convert to xarray
    modularity = xr.DataArray(modularity.astype(_DEFAULT_TYPE),
                              dims=("trials", "times"),
                              coords={"times": time,
                                      "trials": trials})

    return partition, modularity


def compute_allegiance_matrix(A, kw_louvain={}, kw_leiden={}, concat=False,
                              verbose=False, backend='igraph', n_jobs=1):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time),
    the allegiance matrix for the whole period provided will be computed.

    Parameters
    ----------
    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    kw_louvain: dict | {}
        Parameters to be passed to louvain alg from BrainConnectivity toolbox
    kw_leiden: dict | {}
        Parameters to be passed to leindelalg
        https://leidenalg.readthedocs.io/en/stable/reference.html
    concat: bool | False
        Wheter trials are concatenated or not.
    backend: string | "igraph"
        Wheter to use igraph or brainconn package.
    n_jobs: int | 1
        Number of jobs to use when parallelizing in observations.

    Returns
    -------
    T: array_like
        The allegiance matrix between all nodes with shape (roi, roi)
    '''

    assert backend in ['igraph', 'brainconn']

    # Number of ROI
    nC = A.shape[0]
    # Getting roi names
    if isinstance(A, xr.DataArray):
        roi = A.sources.values
    else:
        roi = np.arange(nC, dtype=int)

    # Using igraph
    if backend == 'igraph':
        assert n_jobs == 1, "For backend igraph n_jobs is not allowed"
        #  Find the partitions
        if verbose:
            print("Finding network partitions.\n")
        p, _ = compute_network_partition(
            A, kw_leiden, verbose=verbose, backend='igraph')
    # Using brainconn
    elif backend == 'brainconn':
        #  Find the partitions
        if verbose:
            print("Finding network partitions.\n")
        p, _ = compute_network_partition(
            A, kw_louvain, verbose=verbose, backend='brainconn', n_jobs=n_jobs)

    # Getting dimension arrays
    trials, time = p.trials.values, p.times.values
    # Total number of observations
    nt = len(trials)*len(time)
    # Stack paritions
    p = p.stack(observations=("trials", "times"))

    def _for_frame(t):
        # Allegiance for a frame
        T = np.zeros((nC, nC))
        # Affiliation vector
        av = p.isel(observations=t).values
        # For now convert affiliation vector to igraph format
        n_comm = int(av.max()+1)
        for j in range(n_comm):
            p_lst = np.arange(nC, dtype=int)[av == j]
            grid = np.meshgrid(p_lst, p_lst)
            grid = np.reshape(grid, (2, len(p_lst)**2)).T
            T[grid[:, 0], grid[:, 1]] = 1
        np.fill_diagonal(T, 1)
        return T

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_frame, n_jobs=n_jobs, verbose=verbose,
        total=nt)
    # Compute the single trial coherence
    T = parallel(p_fun(t) for t in range(nt))
    T = np.nanmean(T, 0)

    # Converting to xarray
    T = xr.DataArray(T.astype(_DEFAULT_TYPE), dims=("sources", "targets"),
                     coords={"sources": roi,
                             "targets": roi})
    return T


def windowed_allegiance_matrix(A, kw_louvain={}, kw_leiden={}, times=None,
                               verbose=False, win_args=None, backend='igraph',
                               n_jobs=1):
    """
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time),
    the windowed allegiance matrix. For each window the observations are
    concatenated for all trials and then the allegiance matrix is estimated.

    Parameters
    ----------
    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    kw_louvain: dict | {}
        Parameters to be passed to louvain alg from BrainConnectivity toolbox
    kw_leiden: dict | {}
        Parameters to be passed to leindelalg
        https://leidenalg.readthedocs.io/en/stable/reference.html
    times: array_like
        Time array to construct the windows.
    win_args: dict
        Which arguments to be passed to define_windows
        :py: `frites.conn.conn_sliding_windows`
    concat: bool | False
        Wheter trials are concatenated or not.
    backend: string | "igraph"
        Wheter to use igraph or brainconn package.
    n_jobs: int | 1
        Number of jobs to use when parallelizing in observations.

    Returns
    -------
    T: array_like
        The allegiance matrix between all nodes with shape
        (roi, roi, trials, time)
    """

    from frites.conn.conn_sliding_windows import define_windows

    assert isinstance(win_args, dict)
    assert isinstance(A, xr.DataArray)
    assert ('times' in A.dims) and ('trials' in A.dims) and (
        'sources' in A.dims) and ('targets' in A.dims)

    # Number of regions
    nC = A.shape[0]
    # ROIs
    roi = A.sources.values
    # Define windows
    win, t_win = define_windows(times, **win_args)
    # For a given trial computes windowed allegiance

    def _for_win(trial, win):
        T = xr.DataArray(np.zeros((nC, nC, len(win))),
                         dims=("sources", "targets", "times"),
                         coords={"sources": roi,
                                 "targets": roi,
                                 "times": t_win})
        for i_w, w in enumerate(win):
            T[..., i_w] = compute_allegiance_matrix(A.isel(trials=[trial],
                                                           times=slice(w[0],
                                                                       w[1])),
                                                    kw_louvain, kw_leiden,
                                                    verbose=verbose,
                                                    backend=backend, n_jobs=1)
        return T.astype(_DEFAULT_TYPE)

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_win, n_jobs=n_jobs, verbose=verbose,
        total=A.shape[2])
    # compute the single trial coherence
    T = parallel(p_fun(trial, win) for trial in range(A.shape[2]))
    # Concatenating
    T = xr.concat(T, dim="trials")
    # Ordering dimensions
    T = T.transpose("sources", "targets", "trials", "times")
    # Assign time axis
    T = T.assign_coords({"trials": A.trials.values})
    return T
