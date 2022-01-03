import numpy as np
import xarray as xr
import numba as nb

from .util import _check_inputs
from frites.utils import parallel_func
from .layerwise import compute_network_partition
from ..config import _DEFAULT_TYPE


def compute_allegiance_matrix(A, kw_bc={}, backend='igraph',
                              n_jobs=1, verbose=False):
    """
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time),
    the allegiance matrix for the whole period provided will be computed.

    Parameters
    ----------
    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    kw_bc: dict | {}
        Parameters to be passed to louvain alg from BrainConnectivity toolbox
        https://leidenalg.readthedocs.io/en/stable/reference.html
    backend: string | "igraph"
        Wheter to use igraph or brainconn package.
    n_jobs: int | 1
        Number of jobs to use when parallelizing in observations.

    Returns
    -------
    T: array_like
        The allegiance matrix between all nodes with shape (roi, roi)
    """

    assert backend in ['igraph', 'brainconn']

    # Number of ROI
    nC = A.shape[0]
    # Getting roi names
    if isinstance(A, xr.DataArray):
        roi = A.sources.values
    else:
        roi = np.arange(nC, dtype=int)

    # Get the partitions
    p, _ = compute_network_partition(A,  kw_bc=kw_bc, backend=backend,
                                     n_jobs=n_jobs, verbose=verbose)

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
    T = xr.DataArray(T.astype(_DEFAULT_TYPE),
                     dims=("sources", "targets"),
                     coords={"sources": roi,
                             "targets": roi})
    return T


def windowed_allegiance_matrix(A, kw_bc={}, times=None,
                               win_args=None, backend='igraph',
                               n_jobs=1, verbose=False):
    """
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time),
    the windowed allegiance matrix. For each window the observations are
    concatenated for all trials and then the allegiance matrix is estimated.

    Parameters
    ----------
    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    kw_bc: dict | {}
        Parameters to be passed to louvain alg from BrainConnectivity toolbox
    times: array_like
        Time array to construct the windows.
    win_args: dict
        Which arguments to be passed to define_windows
        :py: `frites.conn.conn_sliding_windows`
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
                                                    kw_bc=kw_bc,
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


@nb.jit(nopython=True)
def _nan_pad(x, new_size, pad_value):
    pad_array = pad_value*np.ones(new_size-len(x), dtype=x.dtype)
    return np.hstack((x, pad_array))


@nb.jit(nopython=True)
def compute_icts(times):
    """ Given the activation times compute the ICT """
    ict = np.diff(times)
    return ict


@nb.jit(nopython=True)
def array_icts(array, times, pad=False, pad_value=np.nan):
    """ Given the activation times compute the ICT """

    act_times = times[array]
    ict = np.diff(act_times)
    if not pad:
        return ict
    else:
        new_size = len(times) - 1
        return _nan_pad(ict, new_size, pad_value)


def tensor_icts(tensor, times, n_jobs=1, verbose=False):
    """
    Computes the ICTS for all edges in the temporal network.
    """

    if not tensor.dtype == bool:
        tensor = tensor.astype(bool)

    n_edges, n_trials, n_times = tensor.shape
    _new_size = n_times - 1

    @nb.jit(nopython=True)
    def _edgewise(e):
        ict = np.empty((n_trials, _new_size))
        # For each trial
        for i in range(n_trials):
            ict[i, :] = array_icts(
                tensor[e, i],
                times,
                pad=True, pad_value=0)
        return ict

    # Computed in parallel for each edge
    parallel, p_fun = parallel_func(
        _edgewise, n_jobs=n_jobs, verbose=verbose,
        total=n_edges)

    ict = parallel(p_fun(e) for e in range(n_edges))
    ict = np.stack(ict, axis=0)

    return ict


def compute_temporal_correlation(A, tau=1, mirror=False):
    # Check inputs
    _check_inputs(A, 3)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray):
        A = A.values

    #  Number of channels
    nC = A.shape[0]

    if mirror:
        A = A + np.transpose(A, (1, 0, 2))

    if tau < 1:
        tau = 1

    num = (A[:, :, 0:-tau] * A[:, :, tau:]).sum(axis=1)
    den = np.sqrt(A[:, :, 0:-tau].sum(axis=1) * A[:, :, tau:].sum(axis=1))
    Ci = np.nansum((num / den), axis=1) / (A.shape[-1] - 1)
    return np.nansum(Ci) / nC


def cosine_similarity(A, thr=None, mirror=False):
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray):
        A = A.values

    if mirror:
        A = A + np.transpose(A, (1, 0, 2, 3))

    num = (A[:, :, :, :-1]*A[:, :, :, 1:]).sum(axis=1)
    den = np.sqrt(np.sum(A[:, :, :, :-1]**2, axis=1)) * \
        np.sqrt(np.sum(A[:, :, :, 1:]**2, axis=1))

    return num / den


def jaccard_index(A, mirror=False):
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray):
        A = A.values

    if mirror:
        A = A + np.transpose(A, (1, 0, 2, 3))

    num = (A[:, :, :, :-1] * A[:, :, :, 1:]).sum(axis=1)
    den = (A[:, :, :, :-1] + A[:, :, :, 1:])
    #  The union is the number of elements in A plus the elements in B
    #  minus the number of elements A and B have in common
    den[den == 2] = 1
    den = den.sum(axis=1)
    J = num/den
    J[J == np.inf] = np.nan
    return J
