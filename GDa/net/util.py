import igraph as ig 
import numpy  as np
import xarray as xr
from   scipy  import stats

def instantiate_graph(A, is_weighted = False):
    if is_weighted is False:
        g = ig.Graph.Weighted_Adjacency(A.tolist(), attr="weight", loops=False, mode=ig.ADJ_UNDIRECTED)
    else:
        g = ig.Graph.Adjacency(A.tolist(), mode=ig.ADJ_UNDIRECTED)
    return g

def compute_coherence_thresholds(tensor, q=0.8, relative=False):
    r'''
    Compute the power/coherence thresholds for the data
    > INPUTS:
    - tensor: Data with dimensions [nodes/links,bands,observations]
    - q: Quartile value to use as threshold
    - relative: If True compute one threshold for each node/link in each band (defalta False)
    > OUTPUTS:
    - coh_thr: Threshold values, if realtive is True it will have dimensions ["links","bands","trials"] otherwise ["bands","trials"]
    '''
    n_nodes, n_bands, n_trials, n_obs = tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]
    if relative:
        coh_thr = np.zeros([n_nodes, n_bands, n_trials])
        for t in range( n_trials ):
            for i in range( n_bands ):
                coh_thr[:,i,t] = np.squeeze( stats.mstats.mquantiles(tensor[:,i,t,:], prob=q, axis=-1) )
        coh_thr = xr.DataArray(coh_thr, dims=("links","bands","trials"))
    else:
        coh_thr = np.zeros( [n_bands, n_trials] )
        for t in range( n_trials ):
            for i in range( n_bands ):
                coh_thr[i,t] = stats.mstats.mquantiles(tensor[:,i,t,:].flatten(), prob=q)
        coh_thr = xr.DataArray(coh_thr, dims=("bands","trials"))
    
    return coh_thr

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    r'''
    Check if the matrix a is symmetric
    '''
    return numpy.allclose(a, a.T, rtol=rtol, atol=atol)
