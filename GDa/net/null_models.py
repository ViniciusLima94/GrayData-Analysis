import numpy  as np 
import igraph as ig
from   tqdm   import tqdm
from   .util  import instantiate_graph

def shuffle_frames(A, thr=None, seed=0):
    # Check the dimension
    assert len(A.shape)==3, "The adjacency tensor should be 3D."

    #  Number of observations
    nt = A.shape[-1]

    #  Set randomization seed
    np.random.seed(seed)
    #  Observation indexes
    idx = np.arange(nt, dtype = int)
    np.random.shuffle(idx)
    A_null = ( A + np.transpose( A,  (1,0,2) ) ).copy()
    A_null = A_null[:,:,idx]
    if thr is not None:
        A_null = (A_null > thr).astype(int)

    return A_null

def randomize_edges(A, thr=None, n_rewires = 100, seed=0):
    # Check the dimension
    assert len(A.shape)==3, "The adjacency tensor should be 3D."
    assert thr != None, "A threshold value should be provided."

    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]

    A_null = np.empty_like(A)

    for t in range(nt):
        g   = instantiate_graph(A[:,:,t], thr=thr)
        G   = g.copy()
        G.rewire(n=n_rewires)
        A_null[:,:,t] = np.array(list(G.get_adjacency()))

    return A_null