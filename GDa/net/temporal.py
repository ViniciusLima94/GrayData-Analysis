import numpy as np
import xarray as xr
import numba as nb

from .util import _check_inputs
from frites.utils import parallel_func


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

    #  Number of channels
    nC = A.shape[0]

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
