import igraph as ig

def instantiate_graph(A, thr = None):
    if thr is None:
        g = ig.Graph.Weighted_Adjacency(A.tolist(), attr="weight", loops=False, mode=ig.ADJ_UNDIRECTED)
    else:
        g = ig.Graph.Adjacency((A>thr).tolist(), mode=ig.ADJ_UNDIRECTED)
    return g
