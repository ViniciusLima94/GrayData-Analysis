import numpy as np

def create_stim_grid(stim_list, nT, nt):
    n_stim           = int(stim_list.max()) 
    self.stim_grid   = np.zeros([n_stim, nT*nt])
    #  Repeate each stimulus to match the length of the trial 
    stim             = np.repeat(stim_list-1, nt )
    for i in range(n_stim):
        self.stim_grid[i] = (stim == i).astype(bool)
    return self.stim_grid.astype(bool)

def create_stages_time_grid(t_cue_on, t_cue_off, t_match_on, fsample, tarray, ntrials):
    t_cue_off  = (t_cue_off-t_cue_on)/fsample
    t_match_on = (t_match_on-t_cue_on)/fsample
    tt         = np.tile(tarray, (ntrials, 1))
    #  Create grids with starting and ending times of each stage for each trial
    t_baseline = ( (tt<0) ).reshape(ntrials*len(tarray))
    t_cue      = ( (tt>=0)*(tt<t_cue_off[:,None]) ).reshape(ntrials*len(tarray))
    t_delay    = ( (tt>=t_cue_off[:,None])*(tt<t_match_on[:,None]) ).reshape(ntrials*len(tarray))
    t_match    = ( (tt>=t_match_on[:,None]) ).reshape(ntrials*len(tarray))
    # Maks
    t_mask     = [t_baseline, t_cue, t_delay, t_match]
    return t_mask
