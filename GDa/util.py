import numpy  as np 
import pandas as pd
import scipy

def create_stages_time_grid(t_cue_on, t_cue_off, t_match_on, fsample, tarray, ntrials,flatten=False):
    """
    Create grids to keep track of different stages of the experiment    

    Parameters
    ----------
    t_cue_on: array_like
        Cue onset times
    t_cue_off: array_like
        Cue offset times 
    t_match_on: array_like
        Match onset times
    fsample: float
        Frequency sample
    tarray: array_like
        Time axis array
    ntrials: int
        Number of trials
    flatten: bool | False
        Wheter to concatenate trials and time dimensions

    Returns
    -------
    Dictionary with boolean masks to acess each stage of the experiment for each trial
    """

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

def filter_trial_indexes(trial_info, trial_type=None, behavioral_response=None):
    """
    Filter data (can be a session, power times seires or coherence)
    by desired trials based on trial_type and behav. response.

    Parameters
    ----------
    trial_info: pandas.DataFrame
        DataFrame with metadata used to filter the desired trials
    trial_type: int | None
        the type of trial (DRT/fixation)
    behavioral_response: int | None
        Wheter to get sucessful (1) or unsucessful (0) trials
    Returns
    -------
    filtered_trials | array_like
        The number of the trials correspondent to the selected trial_type and behavioral_response
    filtered_trials_idx | array_like
        The index of the trials corresponding to the selected trial_type and behavioral_response
    """
    # Check for invalid values
    assert isinstance(trial_info, pd.core.frame.DataFrame)
    assert _check_values(trial_type,[None, 1.0, 2.0, 3.0]) is True, "Trial type should be either 1, 2, 3 or None."
    assert _check_values(behavioral_response,[None,np.nan, 0.0, 1.0]) is True, "Behavioral response should be either 0, 1, NaN or None."

    if isinstance(trial_type, np.ndarray) and behavioral_response is None:
        idx = trial_info['trial_type'].isin(trial_type)
    if trial_type is None and isinstance(behavioral_response, np.ndarray):
        idx = trial_info['behavioral_response'].isin(behavioral_response)
    if isinstance(trial_type, np.ndarray) and isinstance(behavioral_response, np.ndarray):
        idx = trial_info['trial_type'].isin(trial_type) & trial_info['behavioral_response'].isin(behavioral_response)
    filtered_trials     = trial_info[idx].trial_index.values
    filtered_trials_idx = trial_info[idx].index.values
    return filtered_trials, filtered_trials_idx

def _check_values(values, in_list):
    is_valid=True
    if values is None:
        return is_valid
    else:
        for val in values:
            if val not in in_list:
                is_valid=False
                break
        return is_valid

#  def reshape_observations(tensor, nT, nt):
#      r'''
#      Reshape the tensor to have all the trials  and time dimensioins concatenated.
#      > INPUTS:
#      - tensor: Input tensor
#      - nT: Number of trials
#      - nt: Number of time points
#      > OUTPUTS:
#      - aux: reshaped version of the tensor where the last dimensions has nT*nt points
#      '''
#      if len(tensor.shape) == 2:
#          aux = tensor.reshape([nT * nt])
#      if len(tensor.shape) == 3:
#          aux = tensor.reshape([tensor.shape[0], nT * nt])
#      if len(tensor.shape) == 4:
#          aux = tensor.reshape([tensor.shape[0], tensor.shape[1], nT * nt])
#      if len(tensor.shape) == 5:
#          aux = tensor.reshape([tensor.shape[0], tensor.shape[1], tensor.shape[2], nT * nt])
#      return aux

#  def smooth(x, w):
#      r'''
#      Auxiliary function to smooth a 1D array using a boxcar function.
#      > INPUTS:
#      - x: A 1D array
#      - w: Size of the window
#      > OUTPUTS:
#      - smoothed array.
#      '''
#      return scipy.signal.fftconvolve(x, np.ones(w)/w, mode='same')

#def create_stim_grid(stim_list, nT, nt):
#    n_stim           = int(stim_list.max()) 
#    self.stim_grid   = np.zeros([n_stim, nT*nt])
#    #  Repeate each stimulus to match the length of the trial 
#    stim             = np.repeat(stim_list-1, nt )
#    for i in range(n_stim):
#        self.stim_grid[i] = (stim == i).astype(bool)
#    return self.stim_grid.astype(bool)

#  def downsample(x, delta, axis=0):
#      r'''
#      Downsample 1D or 2D array.
#      > INPUTS:
#      - downsample: Downsample factor
#      - axis: Which axis to downsample
#      > OUTPUTS:
#      - Array with the corresponding dimension downsampled
#      '''
#      if len(x.shape)==1:
#          return x[::delta]
#      else:
#          if axis == 0:
#              return x[::delta, :]
#          elif axis ==1:
#              return x[:,::delta]

#  def reshape_trials(tensor, nT, nt):
#      if len(tensor.shape) == 1:
#          aux = tensor.reshape([nT, nt])
#      if len(tensor.shape) == 2:
#          aux = tensor.reshape([tensor.shape[0], nT, nt])
#      if len(tensor.shape) == 3:
#          aux = tensor.reshape([tensor.shape[0], tensor.shape[1], nT, nt])
#      if len(tensor.shape) == 4:
#          aux = tensor.reshape([tensor.shape[0], tensor.shape[1], tensor.shape[2], nT, nt])
#      return aux
