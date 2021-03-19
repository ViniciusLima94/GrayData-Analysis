import igraph as ig 
import numpy  as np
from   scipy  import stats

def instantiate_graph(A, is_weighted = False):
    if is_weighted is False:
        g = ig.Graph.Weighted_Adjacency(A.tolist(), attr="weight", loops=False, mode=ig.ADJ_UNDIRECTED)
    else:
        g = ig.Graph.Adjacency(A.tolist(), mode=ig.ADJ_UNDIRECTED)
    return g

def compute_thresholds(tensor, q=0.8, relative=False):
    r'''
    Compute the power/coherence thresholds for the data
    > INPUTS:
    - tensor: Data with dimensions [nodes/links,bands,observations]
    - q: Quartile value to use as threshold
    - relative: If True compute one threshold for each node/link in each band (defalta False)
    > OUTPUTS:
    - coh_thr: Threshold values, if realtive is True it will have dimensions [bands, nodes/links] otherwise [bands]
    '''
    n_nodes, n_bands, n_obs = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    if relative:
        coh_thr = np.zeros([n_bands, n_nodes])
        for i in range( n_bands ):
            coh_thr[i] = np.squeeze( stats.mstats.mquantiles(tensor[:,i,:], prob=q, axis=1) )
    else:
        coh_thr = np.zeros( n_bands )
        for i in range( n_bands ):
            coh_thr[i] = stats.mstats.mquantiles(tensor[:,i,:].flatten(), prob=q)
    return coh_thr

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    r'''
    Check if the matrix a is symmetric
    '''
    return numpy.allclose(a, a.T, rtol=rtol, atol=atol)
