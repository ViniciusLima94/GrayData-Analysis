import numpy as np 
import numba as nb
import brainconn as bc 
import igraph as ig 

from .util import (instantiate_graph, _is_binary,
                   _convert_to_membership)


def invalid_graph(**args):
    msg = "Method not implelented for weighted graph and required backend"
    raise ValueError(msg)

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
        False: bc.core.kcore_bu,
        True: bc.core.score_wu
    }

    # Distances
    funcs["igraph"]["shortest_path"] = {
        False: ig.Graph.shortest_paths_dijkstra,
        True: ig.Graph.shortest_paths_dijkstra,
    }

    funcs["brainconn"]["shortest_path"] = {
        False: bc.distance.distance_bin,
        True: bc.distance.distance_wei
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
        False: local_efficiency_ig,
        True: invalid_graph ,
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
def _degree(A):
    """ Compute the degree from and adjacency matrix """
    return A.sum(-1)

def _clustering(A, backend="igraph"):
    """ Compute the clustering from and adjacency matrix """
    # Check backend
    assert backend in ["igraph", "brainconn"]
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)
    # Get the function
    func = _get_func(backend, "clustering", is_weighted)
    if backend == 'igraph':
        g = instantiate_graph(A, is_weighted=is_weighted)
        clustering = func(g, weights="weight")
    elif backend == 'brainconn':
        clustering = func(A) 
    return clustering


def _coreness(A, backend="igraph"):
    """ Compute the coreness from and adjacency matrix """
    # Check backend
    assert backend in ["igraph", "brainconn"]
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)
    # Get the function
    func = _get_func(backend, "coreness", is_weighted)
    if backend == 'igraph':
        g = instantiate_graph(A, is_weighted=is_weighted)
        coreness = func(g, weights="weight")
    elif backend == 'brainconn':
        coreness = func(A) 
    return coreness


def _shortest_path(A, backend="igraph"):
    """ Compute the shortest_path from and adjacency matrix """
    # Check backend
    assert backend in ["igraph", "brainconn"]
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)
    # Get the function
    func = _get_func(backend, "shortest_path", is_weighted)
    if backend == 'igraph':
        g = instantiate_graph(A, is_weighted=is_weighted)
        shortest_path = func(g, weights="weight")
    elif backend == 'brainconn':
        shortest_path = func(A) 
    return shortest_path


def _betweenness(A, backend="igraph"):
    """ Compute the betweenness from and adjacency matrix """
    # Check backend
    assert backend in ["igraph", "brainconn"]
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)
    # Get the function
    func = _get_func(backend, "betweenness", is_weighted)
    if backend == 'igraph':
        g = instantiate_graph(A, is_weighted=is_weighted)
        betweenness = func(g, weights="weight")
    elif backend == 'brainconn':
        betweenness = func(A) 
    return betweenness


def _modularity(A, backend="igraph"):
    """ Compute the modularity from and adjacency matrix """
    # Check backend
    assert backend in ["igraph", "brainconn"]
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)
    # Get the function
    func = _get_func(backend, "modularity", is_weighted)
    if backend == 'igraph':
        g = instantiate_graph(A, is_weighted=is_weighted)
        membership, mod = func(g, is_weighted, True)
    elif backend == 'brainconn':
        membership, mod = func(A) 
    return membership, mod

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

def local_efficiency_ig(Gw):
    """ BrainConn implementation of efficiency but 
    using igraph Dijkstra algorithm for speed 
    :py:`brainconn.distance.efficiency_wei`
    """
    from bc.utils.matrix import invert
    from bc.utils.misc import cuberoot

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
        is_weighted = not _is_binary(Gl[np.ix_(V, V)].sum())
        g = instantiate_graph(Gl[np.ix_(V, V)], is_weighted=is_weighted)
        # inverse distance matrix
        e = distance_inv(g, is_weighted)
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
