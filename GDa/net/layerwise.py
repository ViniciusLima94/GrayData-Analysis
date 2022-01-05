import numpy as np
import xarray as xr

from frites.utils import parallel_func
from .io import (_check_inputs, _unwrap_inputs)
from .static import (_degree, _clustering, _coreness,
                     _shortest_path, _betweenness,
                     _modularity, _efficiency)
from ..config import _DEFAULT_TYPE


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

    node_degree = _degree(A)

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
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A, concat_trials=True)
    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]
    #  Variable to store node clustering
    clustering = np.zeros([nC, nt])

    # Compute for a single observation
    def _for_frame(t):
        # Call core function
        clustering = _clustering(A[..., t], backend=backend)
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


def compute_nodes_coreness(A, kw_bc={}, verbose=False, backend='igraph',
                           n_jobs=1):
    """
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time),
    the coreness for each node is computed for all the trials concatenated.

    Parameters
    ----------
    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    kw_bc: dict | {}
        Parameters to be passed to brainconn implementation
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
    #  Number of observations
    nt = A.shape[-1]

    # Compute for a single observation
    def _for_frame(t):
        # Call core function
        coreness = _coreness(A[..., t], kw_bc=kw_bc, backend=backend)
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


def compute_nodes_distances(A, backend="igraph", verbose=False, n_jobs=1):
    """
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time),
    the lengths of shortest paths between all pairs of nodes for each node is
    computed for all the trials concatenated.

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
    distances: array_like
        A matrix containing the nodes coreness with shape
        (roi,roi,trials,time).
    """

    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A, concat_trials=True)
    #  Number of observations
    nt = A.shape[-1]

    ##################################################################
    # Computes nodes' shortest paths
    #################################################################

    # Compute for a single observation
    def _for_frame(t):
        dist = _shortest_path(A[..., t], backend=backend)
        return dist

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_frame, n_jobs=n_jobs, verbose=verbose,
        total=nt)
    # Compute the single trial coherence
    dist = parallel(p_fun(t) for t in range(nt))
    # Convert to numpy array
    dist = np.asarray(dist)

    # Unstack trials and time
    dist = dist.reshape((len(roi), len(roi), len(trials), len(time)))
    # Convert to xarray
    dist = xr.DataArray(dist.astype(_DEFAULT_TYPE),
                        dims=("sources", "targets", "trials", "times"),
                        coords={"sources": roi,
                                "targets": roi,
                                "times": time,
                                "trials": trials})

    return dist


def compute_nodes_efficiency(A, backend="igraph", verbose=False, n_jobs=1):
    """
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time),
    the efficiency for each node is computed for all the trials concatenated.

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
    #  Number of observations
    nt = A.shape[-1]

    ##################################################################
    # Computes nodes' efficiency
    #################################################################

    # Compute for a single observation
    def _for_frame(t):
        eff = _efficiency(A[..., t], backend=backend)
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
    #  Number of observations
    nt = A.shape[-1]

    # Compute for a single observation
    def _for_frame(t):
        betweenness = _betweenness(A[..., t], backend=backend)
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


def compute_network_partition(A,  kw_bc={}, backend='igraph',
                              n_jobs=1, verbose=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time),
    the network partition for each node is computed for all
    the trials concatenated.

    Parameters
    ----------
    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    kw_bc: dict | {}
        Parameters to be passed to brainconn implementation
    backend: string | "igraph"
        Wheter to use igraph or brainconn package.
    n_jobs: int | 1
        Number of jobs to use when parallelizing in observations.

    Returns
    -------
    partition:
        A list with the all the partition found for each layer of the
    '''
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A, concat_trials=True)
    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]

    def _for_frame(t):
        # Call core function
        partition, modularity = _modularity(A[..., t], kw_bc=kw_bc,
                                            backend=backend)
        #  return partition-1, modularity
        return np.concatenate((partition, [modularity]))

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
