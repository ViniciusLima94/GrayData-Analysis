import igraph as ig
import numpy as np
import numba as nb
import xarray as xr
from scipy import stats
from frites.utils import parallel_func


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
    #  partitions = partitions.values[0]
    # Number of time points
    n_times = len(partitions)
    # Affiliation vector
    av = np.zeros((n_nodes, n_times))
    for t in range(n_times):
        for comm_i, comm in enumerate(partitions[t]):
            av[comm, t] = comm_i
    return av


@nb.jit(nopython=True)
def MODquality(A, av, gamma=1):
    """
    Given an affiliation vector compute the modularity of the graph
    given by A (adapted from brainconn).

    Parameters
    ----------
    A: array_like
        Adjacency matrix.
    av: array_like
        Affiliation vector of size (n_nodes) containing the label of the
        community of each node.
    gamma: float
        Value of gamma to use.

    Returns
    -------
    The modularity index of the network.
    """
    # Number of nodes
    n_nodes = A.shape[0]
    # Degrees
    d = A.sum(0)
    # Number of edges
    n_edges = np.sum(d)
    # Initial modularity matrix
    B = A - gamma * np.outer(d, d) / n_edges
    #  B       = B[av][:, av]
    # Tile affiliation vector
    s = av.repeat(n_nodes).reshape((-1, n_nodes))
    return np.sum(np.logical_not(s - s.T) * B / n_edges)


def CPMquality(A, av, gamma=1):
    """
    Constant Potts Model (CPM) quality function.

    Parameters
    ----------
    A: array_like
        Adjacency matrix.
    av: array_like
        Affiliation vector of size (n_nodes) containing the label of the
        community of each node.
    gamma: float
        Value of gamma to use.

    Returns
    -------
    H: float
        The quality given by the CPM model.
    """
    av = av.astype(int)
    # Total number of communities
    n_comm = int(np.max(av))+1
    # Quality index
    H = 0
    for c in range(n_comm):
        # Find indexes of channels in the commucnit C
        idx = av == c
        # Number of nodes in community C
        n_c = np.sum(idx)
        H = H + np.sum(A[np.ix_(av[idx], av[idx])])-gamma*n_c*(n_c-1)/2
    return H


def _check_inputs(array, dims):
    """
    Check the input type and size.

    Parameters
    ----------
    array: array_lie
        The data array.
    dims: int
        The number of dimensions the array should have.
    """
    assert isinstance(dims, int)
    assert isinstance(array, (np.ndarray, xr.DataArray))
    assert len(array.shape) == dims, f"The adjacency tensor should be {dims}D."


def _unwrap_inputs(array, concat_trials=False):
    """
    Unwrap array and its dimensions for further manipulation.

    Parameters
    ----------
    array: array_like
        The data array (roi,roi,trials,time).
    concat_trials: bool | False
        Wheter to concatenate or not trials of the values in the array.

    Returns
    -------
    array values concatenated or not and the values for each of its dimensions.
    """
    if isinstance(array, xr.DataArray):
        # Concatenate trials and time axis
        try:
            roi = array.roi_1.values
            trials = array.trials.values
            time = array.time.values
        except:
            roi = np.arange(0, array.shape[0])
            trials = np.arange(0, array.shape[2])
            time = np.arange(0, array.shape[3])
        if concat_trials:
            array = array.stack(observations=("trials", "times"))
        array = array.values
    else:
        roi = np.arange(0, array.shape[0])
        trials = np.arange(0, array.shape[2])
        time = np.arange(0, array.shape[3])
        if concat_trials:
            array = array.reshape((len(roi), len(roi), len(trials)*len(time)))
    return array, roi, trials, time


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
    n_channels = int(np.roots([1, -1, -2*n_pairs])[0])

    # Adjacency tensor
    A = np.zeros([n_channels, n_channels, n_bands,
                 n_trials, n_times], dtype=dtype)

    for p in range(n_pairs):
        i, j = sources[p], targets[p]
        A[i, j, ...] = A[j, i, ...] = tensor[p, ...]
    return A


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


def compute_coherence_thresholds(tensor, q=0.8, relative=False, verbose=False,
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
    coh_thr: array_like
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
        coh_thr = xr.DataArray(
            np.zeros([n_nodes, n_bands]), dims=("roi", "freqs"))
    else:
        coh_thr = xr.DataArray(np.zeros(n_bands), dims=("freqs"))

    # itr = range(n_bands)  # Iterator
    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_band, n_jobs=n_jobs, verbose=verbose,
        total=n_bands)
    # Compute the single trial coherence
    out = np.squeeze(parallel(p_fun(t) for t in range(n_bands)))
    coh_thr.values = np.stack(out, -1)
    return coh_thr


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
    Check if the matrix a is symmetric
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
