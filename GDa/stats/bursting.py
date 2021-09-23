import numpy  as np
import numba  as nb
from   frites.utils   import parallel_func
from   .util          import custom_mean, custom_std

@nb.jit(nopython=True)
def _nan_pad(x, new_size, pad_value):
    pad_array = pad_value*np.ones(new_size-len(x), dtype=x.dtype)
    return np.hstack( (x, pad_array) )

@nb.jit(nopython=True)
def find_start_end(array, find_zeros=False):
    """
    Given a binary array find the indexes where the sequences of ones start and begin if find_zeros is False.
    Otherwise it will find the indexes where the sequences of zeros start and begin.
    For instance, for the array [0,1,1,1,0,0], would return 1 and 3 respectively for find_zeros=False,
    and 1 and 2 for find_zeros=True.

    Parameters
    ----------
    array: array_like
        Binary array.
    find_zeros: bool | False
        Wheter to find a sequence of zeros or ones

    Returns
    -------
    The matrix containing the start anb ending index
    for each sequence of consecutive ones or zeros with shapes [n_seqs,2]
    where n_seqs is the number of sequences found.
    """
    if find_zeros:
        _bounds = np.array([1])
    else:
        _bounds = np.array([0])

    bounded     = np.hstack((_bounds, array, _bounds))
    difs        = np.diff(bounded)
    # get 1 at run starts and -1 at run ends if find_zeros is False
    if not find_zeros:
        run_starts, = np.where(difs > 0)
        run_ends,   = np.where(difs < 0)
    # get -1 at run starts and 1 at run ends if find_zeros is True
    else:
        run_starts, = np.where(difs < 0)
        run_ends,   = np.where(difs > 0)
    return np.vstack((run_starts,run_ends)).T

@nb.jit(nopython=True)
def find_activation_sequences(spike_train, dt=None):
    """
    Given a spike-train, it finds the length of all activations in it.
    For example, for the following spike-train: x = {0111000011000011111},
    the array with the corresponding sequences of activations (ones) will be
    returned: [3, 2, 5] (times dt if this parameter is provided).

    Parameters
    ----------
    spike_train: array_like
        The binary spike train.
    dt: int | None
        If provided the returned array with the length of activations will be given in seconds.

    Returns
    -------
    act_lengths: array_like
        Array containing the length of activations with shape [n_seqs]
        where n_seqs is the number of sequences found.
    """

    # If no dt is specified it is set to 1
    if dt is None:
        dt = 1
    out         = find_start_end(spike_train)
    act_lengths = (out[:,1]-out[:,0])*dt

    return act_lengths

@nb.jit(nopython=True)
def masked_find_activation_sequences(spike_train, mask, dt=None, drop_edges=False, pad=False):
    """
    Similar to "find_activation_sequences" but a mask is applied to the spike_train while computing
    the size of the activation sequences.'

    Parameters
    ----------
    spike_train: array_like
        The binary spike train.
    mask: array_like
        Binary mask applied to the spike-train.
    dt: int | None
        If provided the returned array with the length of activations will be given in seconds.
    drop_edges: bool | False
        If True will remove the size of the last burst size in case the spike trains ends at one.
    pad: int, float | False
        Pad the activation lengths array to have its maximum values possible.
        If the spike-train contains n samples the number of activation sequences is constrained between 
        0 (no activation of the edge during the period) and ceil(n/2) (which would correspond to an 
        activation time-series with 0s and 1s intercalated).
        If pad is True the series returned will have size ceil(n/2) by being padded with zeros to the right.

    Returns
    -------
    act_lengths: array_like
        Array containing the length of activations with shape [n_seqs]
        where n_seqs is the number of sequences found.
    """

    # Assure that mask is type bool
    #mask = mask.astype(np.bool_)

    # Find the size of the activations lengths for the masked spike_train
    act_lengths = find_activation_sequences(spike_train[mask], dt=dt)
    # If drop_edges is true it will check if activation at the
    # left and right edges crosses the mask limits.
    if drop_edges:
        idx, = np.where(mask==True)
        i,j  = idx[0], idx[-1]
        # If the mask starts at the beggining of the array
        # there is no possibility to cross from the left side
        if i>=1:
            if spike_train[i-1]==1 and spike_train[i]==1 and len(act_lengths)>0 :
                act_lengths = np.delete(act_lengths,0)
        # If the mask ends at the ending of the array
        # there is no possibility to cross from the right side
        if j<len(mask)-1:
            if spike_train[j]==1 and spike_train[j+1]==1 and len(act_lengths)>0 :
                act_lengths = np.delete(act_lengths,-1)

    if pad:
        _new_size   = len(spike_train)//2+1
        act_lengths = _nan_pad(act_lengths, _new_size, 0)

    return act_lengths

def tensor_find_activation_sequences(spike_train, mask, dt=None, drop_edges=False, n_jobs=1):
    """
    A wrapper from "masked_find_activation_sequences" to run for tensor data
    of shape [links, trials, time].

    Parameters
    ----------
    spike_train: array_like
        The binary spike train with shape [links, trials, time].
    mask: array_like
        Binary mask applied to the spike-train with size [trials, time]. For more than one mask
        a dicitionary should be provided where for each key an array with size [trials, time]
        is provided.
    dt: int | None
        If provided the returned array with the length of activations will be given in seconds.
    drop_edges: bool | False
        If True will remove the size of the last burst size in case the spike trains ends at one.
    n_jobs: int | 1
        Number of threads to use

    Returns
    -------
    act_lengths: array_like
        Array containing the length of activations for each link and trial
    """

    # Checking inputs
    assert isinstance(spike_train, np.ndarray)
    assert isinstance(mask, (dict, np.ndarray))
    assert spike_train.ndim == 3

    # Number of edges
    n_edges = spike_train.shape[0]
    # Size of act_length considering the padding
    _new_size   = spike_train.shape[-1]//2+1

    # Find the activation sequences for each edge
    @nb.jit(nopython=True)
    def _edgewise(x, m):
        act_lengths = np.empty((x.shape[0],_new_size))
        # For each trial
        for i in range(x.shape[0]):
            act_lengths[i,:] = masked_find_activation_sequences(x[i,...], m[i,...],
                                                                drop_edges=drop_edges,
                                                                pad=True, dt=dt)
        return act_lengths

    # Computed in parallel for each edge
    parallel, p_fun = parallel_func(
    _edgewise, n_jobs=n_jobs, verbose=False,
    total=n_edges)

    if isinstance(mask, np.ndarray):
        assert mask.ndim == 2
        # If it is DataArray should be converted to ndarray to be compatible with numba
        act_lengths = parallel(p_fun(spike_train[e,...], mask) for e in range(n_edges))
        act_lengths = np.stack(act_lengths,axis=0)
        #  Trade 0s for NaN
        act_lengths = act_lengths.astype(np.float)
        act_lengths[act_lengths==0] = np.nan
        # Concatenate trials
        act_lengths = act_lengths.reshape(act_lengths.shape[0],
                                          act_lengths.shape[1]*act_lengths.shape[2])
    elif isinstance(mask, dict):
        # Use the same keys as the mask
        act_lengths = dict.fromkeys(mask.keys())
        for key in mask.keys():
            act_lengths[key] = parallel(p_fun(spike_train[e,...], mask[key]) for e in range(n_edges))
            act_lengths[key] = np.stack(act_lengths[key],axis=0)
            #  Trade 0s for NaN
            act_lengths[key] = act_lengths[key].astype(np.float)
            act_lengths[key][act_lengths[key]==0] = np.nan
            # Concatenate trials
            act_lengths[key] = act_lengths[key].reshape(act_lengths[key].shape[0],
                                                        act_lengths[key].shape[1]*act_lengths[key].shape[2])


    return act_lengths

def tensor_burstness_stats(spike_train, mask, drop_edges=False, samples=None, dt=None, n_jobs=1):
    """
    Given a data tensor (links,trial,time) composed of spike trains the sequence 
    of activations of it will be determined (see tensor_find_activations_squences) and 
    the following burstness stats computed: link avg. activation
    time (mu), total act. time relative to task stage time (mu_tot), 
    CV (mean activation time over its std).

    Parameters
    ----------
    spike_train: array_like
        The binary spike train with shape [links, trials, time].
    mask: array_like
        Binary mask applied to the spike-train with size [trials, time]. For more than one mask
        a dicitionary should be provided where for each key an array with size [trials, time]
        is provided.
    drop_edges: bool | False
        If True will remove the size of the last burst size in case the spike trains ends at one.
    samples: int, array_like | None
        Total number of samples for each in the spike-train for the period
        delimitated by the mask.
    dt: int | None
        If provided the returned array with the length of activations will be given in seconds.
    n_jobs: int | 1
        Number of threads to use

    Returns
    -------
    bs_stats: array_like
        array containing mu, mu_tot, and CV computed from the activation sequences in the spike train
        with shape [links,4] or [links,stages,4].
    """
    if dt is None:
        dt = 1

    # Computing activation lengths
    out = tensor_find_activation_sequences(spike_train, mask, dt=dt, drop_edges=drop_edges, n_jobs=n_jobs)

    if isinstance(out, np.ndarray):
        # In order ot be able to use NaN
        bs_stats = np.zeros((len(out),4))
        # Computing statistics for each link
        bs_stats[:,0] = np.nanmean(out,axis=-1)
        bs_stats[:,1] = np.nanstd(out,axis=-1)
        bs_stats[:,2] = np.nansum(out,axis=-1)/(samples*dt)
        bs_stats[:,3] = bs_stats[:,1]/bs_stats[:,0] 
    elif isinstance(out, dict):
        assert len(mask)==len(samples)
        # Getting keys
        keys = list( out.keys() )
        bs_stats = np.zeros((len(out[keys[0]]),len(keys),4))
        for idx, key in enumerate(out.keys()):
            bs_stats[:,idx,0] = np.nanmean(out[key],axis=-1)
            bs_stats[:,idx,1] = np.nanstd(out[key],axis=-1)
            bs_stats[:,idx,2] = np.nansum(out[key],axis=-1)/(samples[idx]*dt)
            bs_stats[:,idx,3] = bs_stats[:,idx,1]/bs_stats[:,idx,0] 
    return bs_stats
