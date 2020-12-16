import numpy as np
from   .util import check_symmetric

def compute_temporal_correlation(A, thr = None, tau = 1):
    # Check the dimension
    assert len(A.shape)==3, "The adjacency tensor should be 3D."

    #  Number of channels
    nC = A.shape[0]

    if not check_symmetric:
        A = A + np.transpose(A, (1,0,2)) 
    
    if thr is not None:
        A = A > thr

    if tau < 1:
        tau = 1

    num = (A[:,:,0:-tau] * A[:,:,tau:]).sum(axis=1)
    den = np.sqrt(A[:,:,0:-tau].sum(axis = 1) * A[:,:,tau:].sum(axis = 1))
    Ci  = np.nansum(( num / den ), axis=1) / (A.shape[-1] - 1)
    return np.nansum(Ci) / nC

def cosine_similarity(A, thr = None, mirror=False):
    # Check the dimension
    assert len(A.shape)==4, "The adjacency tensor should be 4D."

    #  Number of channels
    nC = A.shape[0]

    if mirror:
        A = A + np.transpose(A, (1,0,2,3)) 

    num = ( A[:,:,:,:-1]*A[:,:,:,1:] ).sum(axis = 1)
    den = np.sqrt( np.sum(A[:,:,:,:-1]**2, axis = 1) ) * np.sqrt( np.sum(A[:,:,:,1:]**2, axis = 1) )
    
    return num / den

def jaccard_index(A, thr = None, mirror=False):
    # Check the dimension
    assert len(A.shape)==4, "The adjacency tensor should be 4D."
    assert thr != None, "For computing the Jaccard index the threshold should be provided."

    #  Number of channels
    nC = A.shape[0]

    if type(thr) == type(None):
        raise ValueError("For computing the Jaccard index the threshold should be provided")
    else:
        if mirror:
            A = A + np.transpose(A, (1,0,2,3)) 
        num = (A[:,:,:,:-1] * A[:,:,:,1:]).sum(axis=1)
        den = (A[:,:,:,:-1] + A[:,:,:,1:])
        #  The unition in the number of elements in A plus the elements in B minus the number of elements A and B have in common
        den[den==2] = 0
        den = den.sum(axis=1)
        return num/den
