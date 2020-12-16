import igraph as ig

def instantiate_graph(A, thr = None):
    if thr is None:
        g = ig.Graph.Weighted_Adjacency(A.tolist(), attr="weight", loops=False, mode=ig.ADJ_UNDIRECTED)
    else:
        g = ig.Graph.Adjacency((A>thr).tolist(), mode=ig.ADJ_UNDIRECTED)
    return g

def check_symmetric(a, rtol=1e-05, atol=1e-08):
        return numpy.allclose(a, a.T, rtol=rtol, atol=atol)
