import numpy as np
from   frites.utils   import parallel_func
from   .util import custom_mean, custom_std

def find_activation_sequences(spike_train, dt=None, pad=False, max_size=None):
    r'''
    Given a spike-train, it finds the length of all activations in it.
    For example, for the following spike-train: x = {0111000011000011111},
    the array with the corresponding sequences of activations (ones) will be 
    returned: [3, 2, 5] (times dt if this parameter is provided).
    > INPUTS:
    - spike_train: The binary spike train.
    - dt: If providade the returned array with the length of activations will be given in seconds.
    - pad: Wheter to pad or not the array containing the size of the activations lengths in spike_train.
           For example for an spike-train (x) with size N, the maximum number of activations happens when 
           x=[0,1,0,1,0....], therefore the maximum size of the activations lengths array will be
           round(N/2). If the option pad is set to true the act_lengths will be padded at the right side of 
           the array with NaN in order to it have size round(N/2) or the provided max_size.
    - max_size: Max size of the returned array, if none it is set as round(N/2)
    > OUTPUTS:
    - act_lengths: Array containing the length of activations
    '''
    if pad==True and max_size is not None: 
        assert max_size>=int( np.round(len(spike_train)/2) ), "Max size should be greater or equal the maximum number of activation or None."
    if dt is None:
        dt = 1
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], spike_train, [0]))
    # get 1 at run starts and -1 at run ends
    difs        = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends,   = np.where(difs < 0)
    # Length of each activation sequence
    act_lengths =  (run_ends - run_starts)*dt
    # Padding 
    if max_size is None:
        max_size    = int( np.round(len(spike_train)/2) )
    if pad and len(act_lengths)<max_size:
        act_lengths = np.hstack( (act_lengths,np.ones(max_size-len(act_lengths))*np.nan) )
    return act_lengths

def masked_find_activation_sequences(spike_train, mask, dt=None, drop_edges=False):
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
    # Find the size of the activations lengths for the masked spike_train
    act_lengths = find_activation_sequences(spike_train[mask], dt=None, pad=False, max_size=None)
    # If drop_edges is true it will check if activation at the left and right edges crosses the mask
    # limits.
    if drop_edges:
        idx,        = np.where(mask==True)
        i,j         = idx[0], idx[-1]
        if spike_train[i-1]==1 and spike_train[i]==1:
            act_lengths = np.delete(act_lengths,0)
        if spike_train[j]==1 and spike_train[j+1]==1:
            act_lengths = np.delete(act_lengths,-1)
    return act_lengths

def tensor_find_activation_sequences(spike_train, mask, dt=None, drop_edges=False, n_jobs=1):
    r'''
    A wrapper from "masked_find_activation_sequences" to run for tensor data 
    of shape [links, trials, time].
    > INPUTS:
    - spike_train: The binary spike train tensor with size [links, trials, time].
    - mask: Binary mask applied to the spike-train with size [trials, time].
    - dt: If providade the returned array with the length of activations will be given in seconds.
    - drop_edges: If True will remove the size of the last burst size in case the spike trains ends at one.
    > OUTPUTS:
    - act_lengths: Array containing the length of activations for each link and trial
    '''
    act_lengths = []
    for i in range(spike_train.shape[0]):
        act_lengths += [np.apply_along_axis(masked_find_activation_sequences, -1, 
                        spike_train[i,...], mask[i,...], drop_edges=drop_edges, 
                        dt=dt)]
    return act_lengths

def compute_burstness_stats(spike_train, drop_edges=False, samples=None, dt=None):
    r'''
    Given a spike_train the sequence of activations of it 
    will be determined (see find_activations_squences) and 
    the following burstness stats computed: link avg. activation
    time (mu), total act. time relative to task stage time (mu_tot), 
CV (mean activation time over its std).
    > INPUTS:
    - spike_train: The binary spike train.
    - drop_edges: If True will remove the size of the last burst size in case the spike trains ends at one.
    - dt: If providade the returned array with the length of activations will be given in seconds.
    > OUTPUTS:
    array containing mu, mu_tot, and CV computed from the activation sequences in the spike train.
    '''
    if dt is None:
        dt = 1
    if samples is None:
        samples = len(spike_train)
    # Find activation lengths
    act_lengths = find_activation_sequences(spike_train,dt=dt, drop_edges=drop_edges)
    # Compute stats 
    # Mean act. time
    mu     = custom_mean(act_lengths)#act_lengths.mean()
    # Std. of act. time
    mu_st  = custom_std(act_lengths)#act_lengths.std()
    # CV (or irregularity) of links activations
    cv     = mu_st / mu
    # Normalized total act. time
    mu_tot = act_lengths.sum() / ( samples * dt )
    return np.array([mu,mu_st,mu_tot,cv])

def burstness_stats(data, drop_edges=False, samples=None, dt=None):
    r'''
    Given a data tensor (shape n_trials, n_roi, n_times) composed of spike trains the sequence 
    of activations of it will be determined (see find_activations_squences) and 
    the following burstness stats computed: link avg. activation
    time (mu), total act. time relative to task stage time (mu_tot), 
CV (mean activation time over its std).
    > INPUTS:
    - data: Tensor containing the data with shape n_trials, n_roi, n_times.
    - drop_edges: If True will remove the size of the last burst size in case the spike trains ends at one.
    - dt: If providade the returned array with the length of activations will be given in seconds.
    > OUTPUTS:
    array containing mu, mu_tot, and CV computed from the activation sequences in the spike train.
    '''

    # Checking inputs
    assert isinstance(data, np.ndarray)
    assert len(np.shape(data)) is 3

    return None

def compute_burstness_stats_from_act_seq(act_lengths, dt=None):
    r'''
    Given a activation sequence of activations  it 
    will be determined (the same as compute_burstness_stats but instead of inputing the 
    spike train the pre-computed activation sequence lengths are passed) and 
    the following burstness stats computed: link avg. activation
    time (mu), total act. time relative to task stage time (mu_tot), 
CV (mean activation time over its std).
    > INPUTS:
    - act_seq: Activation sequence for a given spi-train.
    > OUTPUTS:
    array containing mu, mu_tot, and CV computed from the activation sequences in the spike train.
    '''
    if dt is None:
        dt = 1
    # Compute stats 
    # Mean act. time
    mu     = custom_mean(act_lengths)#act_lengths.mean()
    # Std. of act. time
    mu_st  = custom_std(act_lengths)#act_lengths.std()
    # CV (or irregularity) of links activations
    cv     = mu_st / mu
    # Normalized total act. time
    mu_tot = act_lengths.sum()# / ( len(spike_train) * dt )
    return np.array([mu,mu_tot,cv])
