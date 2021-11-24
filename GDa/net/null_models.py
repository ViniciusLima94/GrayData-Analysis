import random
import numpy as np
import numba as nb
import xarray as xr
from tqdm import tqdm
from util import instantiate_graph, _check_inputs, _unwrap_inputs

_DEFAULT_TYPE = np.float32


def shuffle_frames(A, seed=0):

    # Checking inputs
    _check_inputs(A, 3)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray):
        A = A.values

    #  Number of observations
    nt = A.shape[-1]
    #  Set randomization seed
    np.random.seed(seed)
    #  Observation indexes
    idx = np.arange(nt, dtype=int)
    np.random.shuffle(idx)
    A_null = (A + np.transpose(A,  (1, 0, 2))).copy()
    A_null = A_null[:, :, idx]

    return A_null.astype(_DEFAULT_TYPE)


def randomize_edges(A, n_rewires=100, seed=0, verbose=False):
    """
    Randomize an adjacency matrix while maintaining its nodes' degrees
    and undirectionality.

    Parameters:
    ----------

    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    n_rewires: int | 100
        Number of rewires to be performed in each adjacency matrix.
    seed: int | 0
        Seed for random edge selection.

    Returns:
    -------
    A_null: array_like
        The randomized multiplex adjacency matrix
        with shape (roi,roi,trials,time).
    """
    # Set random seed (for python's default random and numpy)
    random.seed(seed)
    np.random.seed(seed)

    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A, concat_trials=True)
    #  Number of channels
    # nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]

    A_null = np.empty_like(A)

    itr = range(nt)
    for t in (tqdm(itr) if verbose else itr):
        g = instantiate_graph(A[..., t], is_weighted=False)
        G = g.copy()
        G.rewire(n=n_rewires)
        A_null[..., t] = np.array(list(G.get_adjacency()))

    # Unstack trials and time
    A_null = A_null.reshape((len(roi), len(roi), len(trials), len(time)))
    # Convert to xarray
    A_null = xr.DataArray(A_null.astype(_DEFAULT_TYPE),
                          dims=("sources", "targets", "trials", "times"),
                          coords={"sources": roi,
                                  "targets": roi,
                                  "times": time,
                                  "trials": trials})
    return A_null


def dragon_head_randomizer(A, k=2, seed=0, verbose=False):
    """
    Randomize an adjacency matrix by shuffling the edges
    for a subgraph formed by k-nodes. The nodes are selected
    in a way that try to minimize the difference of the weighth
    values of the edges beling to the subgraph.

    This is done by first sorting the edges according to weights,
    then the first k strongest edges are selected and have are
    "rewired". At the end the null adjancency matrix if built
    by projecting the rewired edges to the (n_nodes, n_nodes)
    matrix.

    Parameters:
    ----------

    A: array_like
        Multiplex adjacency matrix with shape (roi,roi,trials,time).
    k: int | 2
        Size of the k-plets to use should be at least 2.
    seed: int | 0
        Seed for random edge selection.

    Returns:
    -------
    A_null: array_like
        The randomized multiplex adjacency matrix
        with shape (roi,roi,trials,time).
    """
    np.random.seed(seed)

    # Get number of nodes, trials and edges
    n_nodes, n_trials, n_times = A.shape[0], A.shape[2], A.shape[3]
    n_edges = int(n_nodes*(n_nodes-1)/2)

    # The number of k-plets should be proportional to
    # k, the begginiing of the sorted edges is removed
    # in order to achieve #k-plets%k = 0.
    rem = int((np.arange(n_edges) % k)[-1]+1)

    # Total number of observations
    nt = n_trials * n_times

    # Containner for the null matrix
    Anull = A.copy()
    Anull = Anull.reshape((n_nodes, n_nodes, nt))

    itr = range(nt)
    for t in (tqdm(itr) if verbose else itr):
        # Get edges
        B = Anull[..., t][np.triu_indices(n_nodes, 1)]

        # Sort edges and weights
        sort = np.argsort(B)
        wsort = np.sort(B)

        # Reshape in kplets form
        sorted_nodes = sort[rem:].reshape((-1, k))
        sorted_wei = wsort[rem:].reshape((-1, k))
        _sorted_nodes, _sorted_wei = _shuffle_kplets(sorted_nodes, sorted_wei)

        # Bring back to matrix form
        out = _sorted_wei.flatten()[np.argsort(sort[rem:])]
        print(f"{out.shape}")
        print(f"{wsort[:rem].shape}")
        out = np.hstack((wsort[:rem], out))
        # Modify Anull inplace
        # Anull[np.triu_indices(n_nodes), t] = out
        Anull[..., t][np.triu_indices(n_nodes, 1)] = out
        # N = np.zeros((n_nodes, n_nodes))
        # N[np.triu_indices(n_nodes, 1)] = out
    Anull = Anull.reshape((n_nodes, n_nodes, n_trials, n_times))
    return Anull+Anull.transpose((1, 0, 2, 3))


@nb.jit(nopython=True)
def _shuffle_kplets(sorted_nodes, sorted_wei):
    """
    Receives an array with the nodes sorted by weights
    and the weights organized in "k-plets", i.e., an
    array with n rows and k columns.

    Parameters:
    ----------
    sorted_nodes: array_like
        Index of the nodes sorted by weights
        with shape (n,k).
    sorted_wei: array_like
        Weights sorted with shape (n,k).
    """
    # Store the shuffled version of sorted_nodes
    _sorted_nodes = sorted_nodes.copy()
    # Store the shuffled version of sorted_wei
    _sorted_wei = sorted_wei.copy()
    for idx in range(sorted_nodes.shape[0]):
        np.random.shuffle(_sorted_nodes[idx, :])
        ###############################
        np.random.shuffle(_sorted_wei[idx, :])
    return _sorted_nodes, _sorted_wei
