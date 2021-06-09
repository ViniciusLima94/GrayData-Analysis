import random
import numpy  as np 
import xarray as xr
import igraph as ig
from   tqdm   import tqdm
from   .util  import instantiate_graph, _check_inputs

def shuffle_frames(A, seed=0):

    # Checking inputs
    _check_inputs(A, 3)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray): A = A.values

    #  Number of observations
    nt = A.shape[-1]
    #  Set randomization seed
    np.random.seed(seed)
    #  Observation indexes
    idx = np.arange(nt, dtype = int)
    np.random.shuffle(idx)
    A_null = ( A + np.transpose( A,  (1,0,2) ) ).copy()
    A_null = A_null[:,:,idx]

    return A_null

def randomize_edges(A, n_rewires = 100, seed=0, verbose=False):

    # Checking inputs
    _check_inputs(A, 3)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray): A = A.values

    random.seed(seed)

    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]

    A_null = np.empty_like(A)

    itr = range(nt)
    for t in (tqdm(itr) if verbose else itr):
        g   = instantiate_graph(A[:,:,t], is_weighted=False)
        G   = g.copy()
        G.rewire(n=n_rewires)
        A_null[:,:,t] = np.array(list(G.get_adjacency()))
    return A_null
