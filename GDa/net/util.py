import igraph as ig 
import numpy  as np
import xarray as xr
from   scipy  import stats
from   tqdm   import tqdm

def _check_inputs(array, dims):
    r'''
    Check the input type and size.
    > INPUT:
    - array: The data array.
    - dims: The number of dimensions the array should have.
    '''
    assert isinstance(dims, int)
    assert isinstance(array, (np.ndarray, xr.DataArray))
    assert len(array.shape)==dims, f"The adjacency tensor should be {dims}D."

def convert_to_adjacency(tensor,):
    # Number of pairs
    n_pairs    = tensor.shape[0]
    # Number of channels
    n_channels = int(np.roots([1,-1,-2*n_pairs])[0])
    # Number of bands
    n_bands    = tensor.shape[1]
    # Number of trials
    n_trials   = tensor.shape[2]
    # Number of time points
    n_times    = tensor.shape[3]
    # Channels combinations 
    pairs      = np.transpose( np.tril_indices(n_channels, k = -1) )

    # Adjacency tensor
    A = np.zeros([n_channels, n_channels, n_bands, n_trials, n_times])

    for p in range(n_pairs):
        i, j          = int(pairs[p,0]), int(pairs[p,1])
        A[i,j,:,:,:]  = A[j,i,:,:,:] = tensor[p,:,:,:]

    return A

def instantiate_graph(A, is_weighted = False):
    if is_weighted is False:
        g = ig.Graph.Weighted_Adjacency(A.tolist(), attr="weight", loops=False, mode=ig.ADJ_UNDIRECTED)
    else:
        g = ig.Graph.Adjacency(A.tolist(), mode=ig.ADJ_UNDIRECTED)
    return g

def compute_coherence_thresholds(tensor, q=0.8, relative=False, verbose=False):
    r'''
    Compute the power/coherence thresholds for the data
    > INPUTS:
    - tensor: Data with dimensions [nodes/links,bands,observations] or [nodes/links,bands,trials,time]
    - q: Quartile value to use as threshold
    - relative: If True compute one threshold for each node/link in each band (defalta False)
    > OUTPUTS:
    - coh_thr: Threshold values, if realtive is True it will have dimensions ["links","bands","trials"] otherwise ["bands","trials"] (if tensor shape is 3 there is no "trials" dimension)
    '''
    if len(tensor.shape)==4: 
        n_nodes, n_bands, n_trials, n_obs = tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]
        if relative:
            coh_thr = np.zeros([n_nodes, n_bands, n_trials])
            itr = range(n_trials) # Iterator
            for t in (tqdm(itr) if verbose else itr):
                for i in range( n_bands ):
                    coh_thr[:,i,t] = np.squeeze( stats.mstats.mquantiles(tensor[:,i,t,:], prob=q, axis=-1) )
            coh_thr = xr.DataArray(coh_thr, dims=("links","bands","trials"))
        else:
            coh_thr = np.zeros( [n_bands, n_trials] )
            itr = range(n_trials) # Iterator
            for t in (tqdm(itr) if verbose else itr):
                for i in range( n_bands ):
                    coh_thr[i,t] = stats.mstats.mquantiles(tensor[:,i,t,:].flatten(), prob=q)
            coh_thr = xr.DataArray(coh_thr, dims=("bands","trials"))
    if len(tensor.shape)==3: 
        n_nodes, n_bands, n_obs = tensor.shape[0], tensor.shape[1], tensor.shape[2]
        if relative:
            coh_thr = np.zeros([n_nodes, n_bands])
            itr = range(n_bands) # Iterator
            for i in (tqdm(itr) if verbose else itr):
                coh_thr[:,i] = np.squeeze( stats.mstats.mquantiles(tensor[:,i,:], prob=q, axis=-1) )
            coh_thr = xr.DataArray(coh_thr, dims=("links","bands"))
        else:
            coh_thr = np.zeros( [n_bands] )
            itr = range(n_bands) # Iterator
            for i in (tqdm(itr) if verbose else itr):
                coh_thr[i] = stats.mstats.mquantiles(tensor[:,i,:].flatten(), prob=q)
            coh_thr = xr.DataArray(coh_thr, dims=("bands"))

    return coh_thr

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    r'''
    Check if the matrix a is symmetric
    '''
    return numpy.allclose(a, a.T, rtol=rtol, atol=atol)
