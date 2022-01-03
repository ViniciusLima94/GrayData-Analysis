import numpy as np
import numba as nb
import brainconn as bc
import igraph as ig

from functools import partial
from brainconn.utils.matrix import invert
from brainconn.utils.misc import cuberoot
from .util import (instantiate_graph, _is_binary,
                   _convert_to_membership)


###########################################################
# Utilities functions
###########################################################


def invalid_graph(val):
    msg = "Method not implelented for weighted graph and required backend"
    raise ValueError(msg)


def get_weights_params(A):
    is_weighted = not _is_binary(A)
    if is_weighted:
        weights = "weight"
    else:
        weights = None
    return is_weighted, weights

###########################################################
# Brainconn method names
###########################################################


def _get_func(backend, metric, is_weighted):
    funcs = {}

    funcs["igraph"] = {}
    funcs["brainconn"] = {}

    # Clustering
    funcs["igraph"]["clustering"] = {
        False: ig.Graph.transitivity_local_undirected,
        True: ig.Graph.transitivity_local_undirected,
    }

    funcs["brainconn"]["clustering"] = {
        False: bc.clustering.clustering_coef_bu,
        True: bc.clustering.clustering_coef_wu
    }

    # Coreness
    funcs["igraph"]["coreness"] = {
        False: ig.Graph.coreness,
        True: invalid_graph,
    }

    funcs["brainconn"]["coreness"] = {
        False: coreness_bc,
        True: coreness_bc
    }

    # Distances
    funcs["igraph"]["shortest_path"] = {
        False: ig.Graph.shortest_paths_dijkstra,
        True: ig.Graph.shortest_paths_dijkstra,
    }

    funcs["brainconn"]["shortest_path"] = {
        False: dijkstra_bc,
        True: dijkstra_bc
    }

    # Betweenness
    funcs["igraph"]["betweenness"] = {
        False: ig.Graph.betweenness,
        True: ig.Graph.betweenness,
    }

    funcs["brainconn"]["betweenness"] = {
        False: bc.centrality.betweenness_bin,
        True: bc.centrality.betweenness_wei
    }

    # Modularity
    funcs["igraph"]["modularity"] = {
        False: louvain_ig,
        True: louvain_ig,
    }

    funcs["brainconn"]["modularity"] = {
        False: bc.modularity.modularity_louvain_und,
        True: bc.modularity.modularity_louvain_und
    }

    # Efficiency
    funcs["igraph"]["efficiency"] = {
        False: local_efficiency_bin_ig,
        True: local_efficiency_wei_ig
    }

    funcs["brainconn"]["efficiency"] = {
        False: bc.distance.efficiency_bin,
        True: bc.distance.efficiency_wei
    }

    return funcs[backend][metric][is_weighted]

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
        betweenness = func(g, weights=weights)
    elif backend == 'brainconn':
        betweenness = func(A)
    return np.asarray(betweenness)


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

###########################################################
# Igraph wrappers
###########################################################


def louvain_ig(g, is_weighted, membership=False):
    """ Determines louvain modularity algorithm using igraph """
    if is_weighted:
        weights = "weight"
    else:
        weights = None
    louvain_partition = ig.Graph.community_multilevel(g, weights=weights)
    mod = g.modularity(louvain_partition, weights=weights)
    if membership:
        return _convert_to_membership(g.vcount(), list(louvain_partition)), mod
    else:
        return louvain_partition, mod


def dijkstra_bc(A, is_weighted):
    """ Switch for Dijkstra alg. in BrainConn """
    if is_weighted:
        D, _ = bc.distance.distance_wei(A)
    else:
        D = bc.distance.distance_bin(A)
    return D


def distance_inv(g, is_weighted):
    """ igraph inverse distance """
    if is_weighted:
        weights = "weight"
    else:
        weights = None
    D = np.asarray(g.shortest_paths_dijkstra(weights=weights))
    np.fill_diagonal(D, 1)
    D = 1 / D
    np.fill_diagonal(D, 0)
    return D


def local_efficiency_bin_ig(G):
    """ BrainConn implementation of binary 
    efficiency but using igraph Dijkstra algorithm 
    for speed increase
    :py:`brainconn.distance.efficiency_bin`
    """

    # The matrix should be binary
    is_weighted = False
    assert _is_binary(G) == True

    n = len(G)
    E = np.zeros((n,))  # local efficiency
    for u in range(n):
        # find pairs of neighbors
        (V,) = np.where(np.logical_or(G[u, :], G[:, u].T))
        # Check if is disconnected graph
        is_desconnected = np.sum(G[np.ix_(V, V)]) == 0.0
        if is_desconnected:
            E[u] = 0
            continue
        g = instantiate_graph(G[np.ix_(V, V)], is_weighted=is_weighted)
        # inverse distance matrix
        e = distance_inv(g, is_weighted)
        # symmetrized inverse distance matrix
        se = e + e.T
        # symmetrized adjacency vector
        sa = G[u, V] + G[V, u].T
        numer = np.sum(np.outer(sa.T, sa) * se) / 2
        if numer != 0:
            denom = np.sum(sa) ** 2 - np.sum(sa * sa)
            # print numer,denom
            E[u] = numer / denom  # local efficiency
    return E

def local_efficiency_wei_ig(Gw):
    """ BrainConn implementation of binary 
    efficiency but using igraph Dijkstra algorithm 
    for speed increase
    :py:`brainconn.distance.efficiency_wei`
    """

    n = len(Gw)
    Gl = invert(Gw, copy=True)  # connection length matrix
    A = np.array((Gw != 0), dtype=int)

    E = np.zeros((n,))  # local efficiency
    for u in range(n):
        # find pairs of neighbors
        (V,) = np.where(np.logical_or(Gw[u, :], Gw[:, u].T))
        # symmetrized vector of weights
        sw = cuberoot(Gw[u, V]) + cuberoot(Gw[V, u].T)
        # Check if is disconnected graph
        is_desconnected = np.sum(Gl[np.ix_(V, V)]) == 0.0
        if is_desconnected:
            E[u] = 0
            continue
        g = instantiate_graph(Gl[np.ix_(V, V)], is_weighted=True)
        # inverse distance matrix
        e = distance_inv(g, True)
        # symmetrized inverse distance matrix
        se = cuberoot(e) + cuberoot(e.T)

        numer = np.sum(np.outer(sw.T, sw) * se) / 2
        if numer != 0:
            # symmetrized adjacency vector
            sa = A[u, V] + A[V, u].T
            denom = np.sum(sa) ** 2 - np.sum(sa * sa)
            # print numer,denom
            E[u] = numer / denom  # local efficiency
    return E


def coreness_bc(A, delta=1, return_degree=False):
    """ Wrapper for brainconn coreness """
    # Get function
    is_weighted, _ = get_weights_params(A)
    # Get the function
    if is_weighted:
        func = bc.core.score_wu
    else:
        func = bc.core.kcore_bu

    # Number of nodes
    n_nodes = len(A)
    # Initial coreness
    k = 0
    # Store each node's coreness
    k_core = np.zeros(n_nodes)
    # Iterate until get a disconnected graph
    while True:
        # Get coreness matrix and level of k-core
        C, kn = func(A, k)
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
