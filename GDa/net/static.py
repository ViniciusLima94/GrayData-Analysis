import numpy as np
import numba as nb

from .io import get_weights_params
from .util import instantiate_graph
from .fnames import _get_func

###########################################################
# Core methods
###########################################################


@nb.jit(nopython=True)
def _degree(A: np.ndarray, axis: int = 1):
    """ Compute the degree from and adjacency matrix """
    return A.sum(axis)


def _clustering(A: np.ndarray, backend: str = "igraph"):
    """ Compute the clustering from and adjacency matrix """
    # Check backend
    assert backend in ["igraph", "brainconn"]
    # Check if the matrix is weighted or binary
    is_weighted, weights = get_weights_params(A)
    # Get the function
    func = _get_func(backend, "clustering", is_weighted)
    if backend == 'igraph':
        g = instantiate_graph(A, is_weighted=is_weighted)
        clustering = func(g, weights=weights)
        clustering = np.nan_to_num(clustering, copy=False, nan=0.0)
    elif backend == 'brainconn':
        clustering = func(A)
    return np.asarray(clustering)


def _coreness(A: np.ndarray, kw_bc: dict = {}, backend: str = "igraph"):
    """ Compute the coreness from and adjacency matrix """
    # Check backend
    assert backend in ["igraph", "brainconn"]
    # Check if the matrix is weighted or binary
    is_weighted, _ = get_weights_params(A)
    # Get the function
    func = _get_func(backend, "coreness", is_weighted)
    if backend == 'igraph':
        g = instantiate_graph(A, is_weighted=is_weighted)
        coreness = func(g)
    elif backend == 'brainconn':
        coreness = func(A, **kw_bc)
    return np.asarray(coreness)


def _shortest_path(A: np.ndarray, backend: str = "igraph"):
    """ Compute the shortest_path from and adjacency matrix """
    # Check backend
    assert backend in ["igraph", "brainconn"]
    # Check if the matrix is weighted or binary
    is_weighted, weights = get_weights_params(A)
    # Get the function
    func = _get_func(backend, "shortest_path", is_weighted)
    if backend == 'igraph':
        g = instantiate_graph(A, is_weighted=is_weighted)
        shortest_path = func(g, weights=weights)
    elif backend == 'brainconn':
        shortest_path = func(A, is_weighted)
    return np.asarray(shortest_path)


def _betweenness(A: np.ndarray, backend: str = "igraph"):
    """ Compute the betweenness from and adjacency matrix """
    # Check backend
    assert backend in ["igraph", "brainconn"]
    # Check if the matrix is weighted or binary
    is_weighted, weights = get_weights_params(A)
    # Get the function
    func = _get_func(backend, "betweenness", is_weighted)
    if backend == 'igraph':
        g = instantiate_graph(A, is_weighted=is_weighted)
        # Multiply by 2 to match brainconn
        betweenness = func(g, directed=False, weights=weights)
        betweenness = 2*np.asarray(betweenness)
    elif backend == 'brainconn':
        betweenness = func(A)
        betweenness = np.asarray(betweenness)
    return betweenness


def _modularity(A: np.ndarray, kw_bc: dict = {}, backend: str = "igraph"):
    """ Compute the modularity from and adjacency matrix """
    # Check backend
    assert backend in ["igraph", "brainconn"]
    # Check if the matrix is weighted or binary
    is_weighted, _ = get_weights_params(A)
    # Get the function
    func = _get_func(backend, "modularity", is_weighted)
    if backend == 'igraph':
        g = instantiate_graph(A, is_weighted=is_weighted)
        membership, mod = func(g, is_weighted, True)
    elif backend == 'brainconn':
        membership, mod = func(A, **kw_bc)
    return membership, mod


def _efficiency(A: np.ndarray, backend: str = "igraph"):
    """ Compute the efficiency from and adjacency matrix """
    # Check backend
    assert backend in ["igraph", "brainconn"]
    # Check if the matrix is weighted or binary
    is_weighted, _ = get_weights_params(A)
    # Get the function
    func = _get_func(backend, "efficiency", is_weighted)
    if backend == 'igraph':
        eff = func(A)
    elif backend == 'brainconn':
        eff = func(A, local=True)
    return eff
