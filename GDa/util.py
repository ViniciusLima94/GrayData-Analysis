import numpy as np 

def create_stages_time_grid(t_cue_on, t_cue_off, t_match_on, fsample, tarray, ntrials,flatten=False):
    r'''
    Create grids to keep track of different stages of the experiment    
    > INPUTS:
    - t_cue_on: Cue onset times
    - t_cue_off: Cue offset times 
    - t_match_on: Match onset times
    - fsample: Frequency sample
    - tarray: Time axis array
    - ntrials: Number of trials
    - flatten: Wheter to concatenate trials and time dimensions
    > OUTPUTS:
    - Dictionary with boolean masks to acess each stage of the experiment for each trial
    '''
    t_cue_off  = (t_cue_off-t_cue_on)/fsample
    t_match_on = (t_match_on-t_cue_on)/fsample
    tt         = np.tile(tarray, (ntrials, 1))
    #  Create grids with starting and ending times of each stage for each trial
    t_baseline = ( (tt<0) )
    t_cue      = ( (tt>=0)*(tt<t_cue_off[:,None]) )
    t_delay    = ( (tt>=t_cue_off[:,None])*(tt<t_match_on[:,None]) )
    t_match    = ( (tt>=t_match_on[:,None]) )
    # Stage masks
    if flatten==False:
        s_mask     = {'baseline': t_baseline, 
                      'cue':      t_cue,
                      'delay':    t_delay,
                      'match':    t_match}
    else:
        s_mask     = {'baseline': t_baseline.reshape(ntrials*len(tarray)),
                      'cue':      t_cue.reshape(ntrials*len(tarray)),
                      'delay':    t_delay.reshape(ntrials*len(tarray)),
                      'match':    t_match.reshape(ntrials*len(tarray))}

    return s_mask

def downsample(x, delta, axis=0):
    r'''
    Downsample 1D or 2D array.
    > INPUTS:
    - downsample: Downsample factor
    - axis: Which axis to downsample
    > OUTPUTS:
    - Array with the corresponding dimension downsampled
    '''
    if len(x.shape)==1:
        return x[::delta]
    else:
        if axis == 0:
            return x[::delta, :]
        elif axis ==1:
            return x[:,::delta]

def reshape_trials(tensor, nT, nt):
    r'''
    Reshape the tensor to have all the trials and time as separated dimensions.
    > INPUTS:
    - tensor: Input tensor
    - nT: Number of trials
    - nt: Number of time points
    > OUTPUTS:
    - aux: reshaped version of the tensor where the two last dimensions has nT and nt points respectively
    '''
    if len(tensor.shape) == 1:
        aux = tensor.reshape([nT, nt])
    if len(tensor.shape) == 2:
        aux = tensor.reshape([tensor.shape[0], nT, nt])
    if len(tensor.shape) == 3:
        aux = tensor.reshape([tensor.shape[0], tensor.shape[1], nT, nt])
    if len(tensor.shape) == 4:
        aux = tensor.reshape([tensor.shape[0], tensor.shape[1], tensor.shape[2], nT, nt])
    return aux

def reshape_observations(tensor, nT, nt):
    r'''
    Reshape the tensor to have all the trials  and time dimensioins concatenated.
    > INPUTS:
    - tensor: Input tensor
    - nT: Number of trials
    - nt: Number of time points
    > OUTPUTS:
    - aux: reshaped version of the tensor where the last dimensions has nT*nt points
    '''
    if len(tensor.shape) == 2:
        aux = tensor.reshape([nT * nt])
    if len(tensor.shape) == 3:
        aux = tensor.reshape([tensor.shape[0], nT * nt])
    if len(tensor.shape) == 4:
        aux = tensor.reshape([tensor.shape[0], tensor.shape[1], nT * nt])
    if len(tensor.shape) == 5:
        aux = tensor.reshape([tensor.shape[0], tensor.shape[1], tensor.shape[2], nT * nt])
    return aux

#def create_stim_grid(stim_list, nT, nt):
#    n_stim           = int(stim_list.max()) 
#    self.stim_grid   = np.zeros([n_stim, nT*nt])
#    #  Repeate each stimulus to match the length of the trial 
#    stim             = np.repeat(stim_list-1, nt )
#    for i in range(n_stim):
#        self.stim_grid[i] = (stim == i).astype(bool)
#    return self.stim_grid.astype(bool)
