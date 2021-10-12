import numpy as np

def create_stages_time_grid(t_cue_on, t_cue_off, t_match_on, fsample, tarray, ntrials,flatten=False):
    t_cue_off  = (t_cue_off-t_cue_on)/fsample
    t_match_on = (t_match_on-t_cue_on)/fsample
    tt         = np.tile(tarray, (ntrials, 1))
    #  Create grids with starting and ending times of each stage for each trial
    t_baseline = ( (tt<0) )#.reshape(ntrials*len(tarray))
    t_cue      = ( (tt>=0)*(tt<t_cue_off[:,None]) )#.reshape(ntrials*len(tarray))
    t_delay    = ( (tt>=t_cue_off[:,None])*(tt<t_match_on[:,None]) )#.reshape(ntrials*len(tarray))
    t_match    = ( (tt>=t_match_on[:,None]) )#.reshape(ntrials*len(tarray))
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
