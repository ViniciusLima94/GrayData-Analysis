import numpy  as np
import numba  as nb
import cupy   as cp
import xarray as xr
from   frites.utils   import parallel_func
#  from   .util          import custom_mean, custom_std

def _return_target(target="cpu"):
    if target=="cpu":
        xp = np
    elif target=="gpu":
        xp = cp
    elif target=="auto":
        xp = cp.get_array_module(x)
    return xp

def find_activation_sequences(spike_train, dt=None, target="cpu"):
    r'''
    Given a spike-train, it finds the length of all activations in it.
    For example, for the following spike-train: x = {0111000011000011111},
    the array with the corresponding sequences of activations (ones) will be 
    returned: [3, 2, 5] (times dt if this parameter is provided).
    > INPUTS:
    - spike_train: The binary spike train.
    - dt: If providade the returned array with the length of activations will be given in seconds.
    > OUTPUTS:
    - act_lengths: Array containing the length of activations
    '''
    xp = _return_target(target=target)

    if dt is None:
        dt = 1
    # make sure all runs of ones are well-bounded
    bounded = xp.hstack(([0], spike_train, [0]))
    #  # get 1 at run starts and -1 at run ends
    difs        = xp.diff(bounded)
    run_starts, = xp.where(difs > 0)
    run_ends,   = xp.where(difs < 0)
    #  # Length of each activation sequence
    act_lengths =  (run_ends - run_starts)*dt
    return act_lengths

def masked_find_activation_sequences(spike_train, mask, dt=None, drop_edges=False, target="cpu"):
    r'''
    Similar to "find_activation_sequences" but a mask is applied to the spike_train while computing
    the size of the activation sequences.'
    > INPUTS:
    - spike_train: The binary spike train.
    - mask: Binary mask applied to the spike-train.
    - dt: If providade the returned array with the length of activations will be given in seconds.
    - drop_edges: If True will remove the size of the last burst size in case the spike trains ends at one.
    > OUTPUTS:
    - act_lengths: Array containing the length of activations
    '''
    xp = _return_target(target=target)
    # Certifies that both spike_train and maks are either in host or device
    assert type(spike_train)==type(mask)
    # Find the size of the activations lengths for the masked spike_train
    act_lengths = find_activation_sequences(spike_train[mask], dt=dt, target=target)
    # If drop_edges is true it will check if activation at the left and right edges crosses the mask
    # limits.
    if drop_edges:
        idx,        = xp.where(mask==True)
        i,j         = idx[0], idx[-1]
        if isinstance(act_lengths, cp.ndarray): act_lengths=cp.asnumpy(act_lengths)
        if i>=1 and len(act_lengths)>0:
            if spike_train[i-1]==1 and spike_train[i]==1:
                act_lengths = np.delete(act_lengths,0)
        if j<len(mask)-1 and len(act_lengths)>0:
            if spike_train[j]==1 and spike_train[j+1]==1:
                act_lengths = np.delete(act_lengths,-1)
    return act_lengths

def tensor_find_activation_sequences(spike_train, mask, dt=None, drop_edges=False, target="cpu", n_jobs=1):
    r'''
    A wrapper from "masked_find_activation_sequences" to run for tensor data 
    of shape [links, trials, time].
    > INPUTS:
    - spike_train: The binary spike train tensor with size [links, trials, time].
    - mask: Binary mask applied to the spike-train with size [trials, time]. For more than one mask
            a dicitionary should be provided where for each key an array with size [trials, time]
            is provided.
    - dt: If providade the returned array with the length of activations will be given in seconds.
    - drop_edges: If True will remove the size of the last burst size in case the spike trains ends at one.
    - n_jobs: Number of jobs to use
    > OUTPUTS:
    - act_lengths: Array containing the length of activations for each link and trial
    '''

    xp = _return_target(target=target)

    # Checking inputs
    assert isinstance(spike_train, (xp.ndarray, xr.DataArray))
    assert isinstance(mask, (dict, xp.ndarray, xr.DataArray))
    assert len(spike_train.shape) == 3

    n_edges  = spike_train.shape[0]
    n_trials = spike_train.shape[1]
    n_times  = spike_train.shape[2]

    @nb.jit(nopython=True)
    def _edgewise(x, m):
        act_lengths = []
        for i in range(n_trials):
            act_lengths += [xp.apply_along_axis(masked_find_activation_sequences, -1, 
                                                x[i,...], m[i,...], drop_edges=drop_edges, 
                                                dt=dt, target=target)]
        act_lengths = xp.concatenate( act_lengths, axis=0 )
        return act_lengths 

    # Computed in parallel for each edge
    parallel, p_fun = parallel_func(
    _edgewise, n_jobs=n_jobs, verbose=False,
    total=n_edges)

    if isinstance(mask, (xp.ndarray, xr.DataArray)):
        assert len(mask.shape) == 2
        act_lengths = parallel(p_fun(spike_train[e,...], mask) for e in range(n_edges))
    elif isinstance(mask, dict):
        # Use the same keys as the mask
        act_lengths = dict.fromkeys(mask.keys())
        for key in mask.keys():
            assert len(mask[key].shape) == 2
            act_lengths[key] = parallel(p_fun(spike_train[e,...], mask[key]) for e in range(n_edges))
    return act_lengths

def tensor_burstness_stats(spike_train, mask, drop_edges=False, samples=None, dt=None, n_jobs=1):
    r'''
    Given a data tensor (shape n_trials, n_roi, n_times) composed of spike trains the sequence 
    of activations of it will be determined (see tensor_find_activations_squences) and 
    the following burstness stats computed: link avg. activation
    time (mu), total act. time relative to task stage time (mu_tot), 
CV (mean activation time over its std).
    > INPUTS:
    - spike_train: The binary spike train tensor with size [links, trials, time].
    - mask: Binary mask applied to the spike-train with size [trials, time]. For more than one mask
            a dicitionary should be provided where for each key an array with size [trials, time]
            is provided.
    - dt: If providade the returned array with the length of activations will be given in seconds.
    - drop_edges: If True will remove the size of the last burst size in case the spike trains ends at one.
    - n_jobs: Number of jobs to use
    > OUTPUTS:
    - array containing mu, mu_tot, and CV computed from the activation sequences in the spike train.
    '''
    assert len(mask)==len(samples)
    if dt is None: dt = 1

    # Computing activation lengths
    out  = tensor_find_activation_sequences(spike_train, mask, dt=dt, drop_edges=drop_edges, n_jobs=n_jobs)

    if isinstance(out, (np.ndarray, xr.DataArray)):
        bs_stats = np.zeros((out.shape[0],4))
        # Computing statistics for each link
        bs_stats[:,0] = [custom_mean( v ) for v in out]
        bs_stats[:,1] = [custom_std( v )  for v in out]
        bs_stats[:,2] = [np.sum( v )/(samples*dt) for v in out]
        bs_stats[:,3] = bs_stats[:,1]/bs_stats[:,0]
    elif isinstance(out, dict):
        # Getting keys
        keys = list( out.keys() )
        bs_stats = np.zeros((len(out[keys[0]]),len(keys),4))
        for idx, key in enumerate(out.keys()):
            bs_stats[:,idx,0] = [custom_mean( v ) for v in out[key]]
            bs_stats[:,idx,1] = [custom_std( v )  for v in out[key]]
            bs_stats[:,idx,2] = [np.sum( v )/(samples[idx]*dt) for v in out[key]]
            bs_stats[:,idx,3] = bs_stats[:,idx,1]/bs_stats[:,idx,0]
    return bs_stats
