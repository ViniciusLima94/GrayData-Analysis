import numpy as np
import pandas as pd


def create_stages_time_grid(t_cue_on, t_cue_off, t_match_on, fsample, tarray,
                            ntrials, early_delay=None, align_to="cue",
                            flatten=False):
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
    early_delay: float | None
        Time in seconds after cue onset to be considered as
        early delay (if None no division between early and late
        delay is made)
    align_to: string | "cue"
        Wheter the data is aligned to cue or match to set
        the reference time as t_cue_on or t_match_on.
    flatten: bool | False
        Wheter to concatenate trials and time dimensions

    Returns
    -------
    Dictionary with boolean masks to acess each stage of the
    experiment for each trial
    """
    # Get the reference time
    if align_to == "cue":
        t_ref = t_cue_on
    else:
        t_ref = t_match_on

    # Check if has division in early and late delay
    has_early_delay = isinstance(early_delay, float)

    if has_early_delay:
        assert early_delay > 0.0
        mask_names = ["baseline", "cue", "delay_e", "delay_l", "match"]
    else:
        mask_names = ["baseline", "cue", "delay", "match"]

    # Get starting and ending time of each period
    # according to the reference.
    # Divides by fsample to get it in seconds
    t_cue_on = (t_cue_on - t_ref)/fsample
    t_cue_off = (t_cue_off - t_ref)/fsample
    t_match_on = (t_match_on - t_ref)/fsample
    # Convert to column vector for operations
    t_cue_on = t_cue_on[:, None]
    t_cue_off = t_cue_off[:, None]
    t_match_on = t_match_on[:, None]

    # Tile time array to get mask for each trial
    tt = np.tile(tarray, (ntrials, 1))

    # Get the mask for each stage
    # t_baseline = tt < t_cue_on
    # t_cue = ((tt >= t_cue_on)*(tt < t_cue_off))
    # t_delay = ((tt >= t_cue_off)*(tt < t_match_on))
    # t_match = ((tt >= t_match_on))
    t = []
    t += [tt < t_cue_on]
    t += [((tt >= t_cue_on)*(tt < t_cue_off))]
    # If has early delay divides it
    if not has_early_delay:
        t += [((tt >= t_cue_off)*(tt < t_match_on))]
    else:
        t += [((tt >= t_cue_off)*(tt < t_cue_off+early_delay))]
        t += [((tt >= t_cue_off+early_delay)*(tt < t_match_on))]
    t += [((tt >= t_match_on))]

    # Stage masks
    s_mask = {}
    if flatten is False:
        for i, key in enumerate(mask_names):
            s_mask[key] = t[i]
    else:
        for i, key in enumerate(mask_names):
            s_mask[key] = t[i].reshape(ntrials*len(tarray))

    # Stage masks
    # if flatten is False:
        # s_mask = {'baseline': t_baseline,
                  # 'cue':      t_cue,
                  # 'delay':    t_delay,
                  # 'match':    t_match}
    # else:
        # s_mask = {'baseline': t_baseline.reshape(ntrials*len(tarray)),
                  # 'cue':      t_cue.reshape(ntrials*len(tarray)),
                  # 'delay':    t_delay.reshape(ntrials*len(tarray)),
                  # 'match':    t_match.reshape(ntrials*len(tarray))}

    return s_mask


def filter_trial_indexes(trial_info, trial_type=None,
                         behavioral_response=None):
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
        The number of the trials correspondent to the selected
        trial_type and behavioral_response
    filtered_trials_idx | array_like
        The index of the trials corresponding to the selected
        trial_type and behavioral_response
    """
    # Check for invalid values
    assert isinstance(trial_info, pd.core.frame.DataFrame)
    assert _check_values(trial_type, [None, 1.0, 2.0, 3.0]) is True
    assert _check_values(behavioral_response, [None, np.nan, 0.0, 1.0]) is True

    if isinstance(trial_type, np.ndarray) and behavioral_response is None:
        idx = trial_info['trial_type'].isin(trial_type)
    if trial_type is None and isinstance(behavioral_response, np.ndarray):
        idx = trial_info['behavioral_response'].isin(behavioral_response)
    if isinstance(trial_type, np.ndarray) and isinstance(behavioral_response, np.ndarray):
        idx = trial_info['trial_type'].isin(
            trial_type) & trial_info['behavioral_response'].isin(behavioral_response)
    filtered_trials = trial_info[idx].trial_index.values
    filtered_trials_idx = trial_info[idx].index.values
    return filtered_trials, filtered_trials_idx


def _check_values(values, in_list):
    is_valid = True
    if values is None:
        return is_valid
    else:
        for val in values:
            if val not in in_list:
                is_valid = False
                break
        return is_valid
