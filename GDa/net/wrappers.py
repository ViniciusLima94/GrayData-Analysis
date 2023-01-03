"""
Define wrapppers for igraph and brainconn some
functions to standardize their input/outputs
"""

import numpy as np
import igraph as ig
import brainconn as bc

from brainconn.utils.matrix import invert
from brainconn.utils.misc import cuberoot
from .io import get_weights_params
from .util import (instantiate_graph, _is_binary,
                   _is_disconnected, _convert_to_membership)


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
    assert _is_binary(G) is True

    n = len(G)
    E = np.zeros((n,))  # local efficiency
    for u in range(n):
        # find pairs of neighbors
        (V,) = np.where(np.logical_or(G[u, :], G[:, u].T))
        # Check if is disconnected graph
        if _is_disconnected(G[np.ix_(V, V)]):
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
        if _is_disconnected(Gl[np.ix_(V, V)]):
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
