import igraph as ig
import numpy as np
import numba as nb
import xarray as xr

from scipy import stats
from frites.utils import parallel_func
from ..util import _extract_roi


@nb.njit
def _is_binary(matrix):
    """
    Check if a matrix is binary or weighted.

    Parameters
    ----------
    matrix: array_like
        The adjacency matrix or tensor.
    """
    is_binary = True
    for v in np.nditer(matrix):
        if v.item() != 0 and v.item() != 1:
            is_binary = False
            break
    return is_binary


@nb.jit(nopython=True)
def _is_disconnected(matrix):
    """
    Check if a graph is disconnected or not.

    Parameters
    ----------
    matrix: array_like
        The adjacency matrix or tensor.
    """
    is_disconnected = False
    if np.sum(matrix) == 0.0:
        is_disconnected = True
    return is_disconnected


def _convert_to_membership(n_nodes, partitions):
    """ Convert partitions from obtained from louvain igraph to membership. """
    # Extact list of partitions
    av = np.zeros(n_nodes)  # Affiliation vector
    for comm_i, comm in enumerate(partitions):
        av[comm] = comm_i
    return av+1


def _convert_to_affiliation_vector(n_nodes, partitions):
    """
    Convert partitions in leidenalg format to array.
    Parameters
    ----------
    n_nodes: int
        The number of nodes.
    partitions: ModularityVertexPartition
        Parition objects of type
        "leidenalg.VertexPartition.ModularityVertexPartition"
    Returns
    -------
    av: array_like
        Affiliation vector.
    """
    # Extact list of partitions
    n_times = len(partitions)  # Number of time points
    av = np.zeros((n_nodes, n_times))  # Affiliation vector
    for t in range(n_times):
        for comm_i, comm in enumerate(partitions[t]):
            av[comm, t] = comm_i
    return av


def _reshape_list(array, shapes, dtype):
    assert isinstance(shapes, tuple)
    assert isinstance(array,  list)
    idx = 0
    container = np.zeros(shapes, dtype=dtype)
    for i in range(shapes[0]):
        for j in range(shapes[1]):
            container[i, j] = array[idx]
            idx += 1
    return container


def convert_to_adjacency(tensor, sources, targets, dtype=np.float32):
    """
    Convert the tensor with the edge time-series to a matrix representations.

    Parameters
    ----------
    tensor: array_like
        The tensor with the edge time series (roi,freqs,trials,times).
    sources: array_like
        list of source nodes.
    targets: array_like
        list of target nodes.

    Returns
    -------
    The adjacency matrix (roi,roi,freqs,trials,times).
    """

    assert tensor.ndim == 4
    assert tensor.shape[0] == len(sources) == len(targets)

    # Number of pairs
    n_pairs, n_bands, n_trials, n_times = tensor.shape[:]
    # Number of channels
    n_channels = np.max(np.stack((sources, targets))) + 1

    # Adjacency tensor
    A = np.zeros([n_channels, n_channels, n_bands,
                 n_trials, n_times], dtype=dtype)

    for p in range(n_pairs):
        i, j = sources[p], targets[p]
        A[i, j, ...] = A[j, i, ...] = tensor[p, ...]
    return A

def convert_to_stream(adj, sources, targets, dtype=np.float32):
    """
    Convert adjacency tensor to edge time-series tensor.


    Parameters:
    ----------
    adj: array_like#### 
        The adjacency tensor (roi, roi, freqs, trials, times)
    sources: array_like
        list of source nodes.
    targets: array_like
        list of target nodes.

    Returns
    -------
    The edge time-series tensor (roi,freqs,trials,times).
    """

    assert adj.ndim == 5

    # Adjacency tensor dimensions
    n_roi, _, n_bands, n_trials, n_times = adj.shape[:]
    # Number of edges
    n_pairs = int(n_roi * (n_roi - 1) / 2)

    assert n_pairs == len(sources) == len(targets)

    # Edge time-series tensor
    tensor = np.zeros([n_pairs,  n_bands,
                       n_trials, n_times], dtype=dtype)

    for p in range(n_pairs):
        i, j = sources[p], targets[p]
        tensor[p, ...] = adj[i, j, ...]

    # In case adj is and DataArray the output 
    # will also be
    if isinstance(adj, xr.DataArray):
        # Check if dimensions have the apropriate labels
        dims = ['sources', 'targets', 'freqs', 'trials', 'times']
        np.testing.assert_array_equal(
            adj.dims, dims)
        # Get sources and targets roi names
        x_s, x_t \
        = adj.sources.data, adj.targets.data
        roi_c = np.c_[x_s[sources], x_t[targets]]
        idx = np.argsort(np.char.lower(roi_c.astype(str)), axis=1)
        roi = np.c_[[r[i] for r, i in zip(roi_c, idx)]]
        rois = []
        rois += [f'{s}-{t}' for s, t in roi]
        # Convert to DataArray
        tensor = xr.DataArray(tensor,
                              dims=('roi', 'freqs', 'trials', 'times'),
                              coords = {
                                  'roi': rois,
                                  'freqs': adj.freqs.data,
                                  'trials': adj.trials.data,
                                  'times': adj.times.data
                              })
    return tensor


def instantiate_graph(A, is_weighted=False):
    """
    Convert a numpy array adjacency matrix into a igraph object

    Parameters
    ----------
    A: array_like
        Adjacency matrix (roi,roi).
    is_weighted: bool | False
        Wheter the matrix is weighted or not.

    Returns
    -------
    The adjacency matrix as an igraph object.
    """
    if is_weighted:
        g = ig.Graph.Weighted_Adjacency(
            A.tolist(), attr="weight", loops=False, mode=ig.ADJ_UNDIRECTED)
    else:
        g = ig.Graph.Adjacency(A.tolist(), mode=ig.ADJ_UNDIRECTED)
    return g


def compute_quantile_thresholds(tensor, q=0.8, relative=False, verbose=False,
                                n_jobs=1):
    """
    Compute the power/coherence thresholds for the data

    Parameters
    ----------
    tensor: array_like
        Data with dimensions (nodes/links,bands,observations)
        or (nodes/links,bands,trials,time)
    q: array_like | 0.8
        Quantile value to use as threshold
    relative: bool | False
        If True compute one threshold for each node/link
        in each band (defalta False)

    Returns
    -------
    thr: array_like
        Threshold values, if realtive is True it will have
        dimensions ("links","bands","trials") otherwise ("bands","trials")
        (if tensor shape is 3 there is no "trials" dimension)
    """
    n_nodes, n_bands = tensor.shape[0], tensor.shape[1]
    # To compute in parallel for each band

    def _for_band(b):
        if relative:
            out = np.squeeze(stats.mstats.mquantiles(
                tensor[:, b, :], prob=q, axis=-1))
        else:
            out = stats.mstats.mquantiles(tensor[:, b, :].flatten(), prob=q)
        return out

    # Create containers
    if relative:
        thr = xr.DataArray(
            np.zeros([n_nodes, n_bands]), dims=("roi", "freqs"))
    else:
        thr = xr.DataArray(np.zeros(n_bands), dims=("freqs"))

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_band, n_jobs=n_jobs, verbose=verbose,
        total=n_bands)
    # Compute the single trial coherence
    out = np.squeeze(parallel(p_fun(t) for t in range(n_bands)))
    thr.values = np.stack(out, -1)
    return thr


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
    Check if the matrix a is symmetric
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
