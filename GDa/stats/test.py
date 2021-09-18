import numpy as np 
import cupy  as cp
import matplotlib.pyplot as plt
import time

def timer(function, **kwargs):
    start = time.time()
    function(**kwargs)
    end   = time.time()
    return (end-start)

def find_activation_sequences(spike_train=None, dt=None, target="cpu"):
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
    if target=="cpu":
        _np = np
    elif target=="gpu":
        _np = cp
    elif target=="auto":
        if isinstance(spike_train, np.ndarray): _np = np
        elif isinstance(spike_train, cp.ndarray): _np = cp

    if dt is None:
        dt = 1 
    # make sure all runs of ones are well-bounded
    bounded = _np.hstack(([0], spike_train, [0]))
    #  # get 1 at run starts and -1 at run ends
    difs        = _np.diff(bounded)
    run_starts, = _np.where(difs > 0)
    run_ends,   = _np.where(difs < 0)
    #  # Length of each activation sequence
    act_lengths =  (run_ends - run_starts)*dt
    return act_lengths

def masked_find_activation_sequences(spike_train=None, mask=None, dt=None, drop_edges=False, target="cpu"):
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

    # spike_train and mask should be either on host or device
    assert type(spike_train)==type(mask)

    if target=="cpu":
        _np = np
    elif target=="gpu":
        _np = cp
    elif target=="auto":
        if isinstance(spike_train, np.ndarray): _np = np
        elif isinstance(spike_train, cp.ndarray): _np = cp

    # Find the size of the activations lengths for the masked spike_train
    act_lengths = find_activation_sequences(spike_train[mask], dt=dt, target=target)
    # If drop_edges is true it will check if activation at the left and right edges crosses the mask
    # limits.
    if drop_edges:
        idx,        = _np.where(mask==True)
        i,j         = idx[0], idx[-1]
        if i>=1 and len(act_lengths)>0:
            if spike_train[i-1]==1 and spike_train[i]==1:
                act_lengths = _np.delete(act_lengths,0)
        if j<len(mask)-1 and len(act_lengths)>0:
            if spike_train[j]==1 and spike_train[j+1]==1:
                act_lengths = _np.delete(act_lengths,-1)
    return act_lengths

if __name__ == '__main__':

    def _gen_data(n):
        return np.random.rand(n)>0.5

    # Different data sizes 
    n = np.linspace(10000, 540*1176, 50, dtype=int)

    t_cpu = []
    t_gpu = []

    #  for i in range(n.shape[0]):
    #      t_cpu += [timer(find_activation_sequences,spike_train=_gen_data(n[i]), dt=None, target="cpu")]
    #      t_gpu += [timer(find_activation_sequences,spike_train=cp.array(_gen_data(n[i])), dt=None, target="gpu")]
    
    #  def masked_find_activation_sequences(spike_train, mask, dt=None, drop_edges=False, target="cpu"):
    for i in range(n.shape[0]):
        x     = _gen_data(n[i])
        mask  = np.ones_like(x); mask[n[i]//2:]=0
        t_cpu += [timer(masked_find_activation_sequences,spike_train=x, mask=mask,dt=None, drop_edges=True, target="cpu")]
        t_gpu += [timer(masked_find_activation_sequences,spike_train=cp.array(x), mask=cp.array(mask),drop_edges=True,dt=None, target="gpu")]

    plt.plot(n,t_cpu, label='cpu')
    plt.plot(n,t_gpu, label='gpu')
    plt.legend()
    plt.show()


