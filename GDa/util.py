import numpy as np
import xarray as xr
import pandas as pd


def get_areas():
    """
    This function reads a JSON file 'areas.json' and returns a dictionary 
    mapping the names of areas to their respective regions.
    The file should be formatted like this:
        { "areas": [            
        {"name": "Area 1", "region": "Region A"},            
        {"name": "Area 2", "region": "Region B"},            
        ...          ]
        }
    """
    import json

    with open("areas.json") as f:
        areas_dict = json.load(f)["areas"]

    names = []
    regions = []
    for i in range(len(areas_dict)):

        names += [areas_dict[i]["name"].lower()]
        regions += [areas_dict[i]["region"]]

    return dict(zip(names, regions))

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


def average_stages(feature, avg, early_cue=0.2, early_delay=0.3):
    """
    Loads the network feature DataArray and average it for each task
    stage if needed (avg=1) otherwise return the feature itself
    (avg=0).

    Paramters:
    ---------
    feature: xr.DataArray
        A given feature array (power, degree, coreness...) with
        shape (roi, freqs, trials, times).
    avg: int
        Wheter to average over task stages or not
    early_cue: float | 0.2
        Final period of the baseline to condider as cue.
    early_delay: float | 0.3
        The period at the beggining of the delay that should
        be used as early delay.

    Returns:
    -------
    out: xr.DataArray
        If avg==0 returns features otherwise returns
        a version of features averaged over stages.
    """
    if avg == 1:
        out = []
        # Creates stage mask
        mask = create_stages_time_grid(feature.t_cue_on - early_cue * feature.fsample,
                                       feature.t_cue_off,
                                       feature.t_match_on,
                                       feature.fsample,
                                       feature.times.data,
                                       feature.sizes["trials"],
                                       early_delay=early_delay,
                                       align_to="cue",
                                       flatten=False)
        for stage in mask.keys():
            mask[stage] = xr.DataArray(mask[stage], dims=('trials', 'times'),
                                       coords={"trials": feature.trials.data,
                                               "times": feature.times.data
                                               })
        for stage in mask.keys():
            # Number of observation in the specific stage
            n_obs = xr.DataArray(mask[stage].sum("times"), dims="trials",
                                 coords={"trials": feature.trials.data})
            out += [(feature * mask[stage]).sum("times") / n_obs]

        out = xr.concat(out, "times")
        out = out.transpose("trials", "roi", "freqs", "times")
        out.attrs = feature.attrs
    else:
        out = feature.transpose("trials", "roi", "freqs", "times")
    return out


# Extract area names
def _extract_roi(roi, sep):
    # Code by Etiene
    x_s, x_t = [], []
    for r in roi:
        _x_s, _x_t = r.split(sep)
        x_s.append(_x_s), x_t.append(_x_t)
    roi_c = np.c_[x_s, x_t]
    idx = np.argsort(np.char.lower(roi_c.astype(str)), axis=1)
    roi_s, roi_t = np.c_[[r[i] for r, i in zip(roi_c, idx)]].T
    return roi_s, roi_t

def _create_roi_area_mapping(roi):
    """
    Create a mapping between pairs of rois and integer indexes

    Parameters
    ----------
    roi: array_like
       Array of size (n_edges) containing the name of the rois
       (i, j) that form the FC link separated by "-". 
       Ex: ["V1-a3", "V6a-LIP", ..., "a46d-a8"]

    Returns
    -------
    roi_s: array_like
        The name of the source areas
    roi_t: array_like
        The name of the target areas
    roi_is: array_like
        The index of the source areas
    roi_it: array_like
        The index of the target areas
    areas: array_like
        The name of the areas
    mapping: dict
        The mapping area-index created
    """

    # Get sources and target names
    roi_s, roi_t = _extract_roi(roi, "-")
    # Get unique area names
    areas = np.unique(np.stack((roi_s, roi_t)))
    # Get number of unique areas
    n_areas = len(areas)
    # Assign a index for each area
    mapping = dict(zip(areas, range(n_areas)))
    # Convert roi_s roi_t to integer indexes
    roi_is = np.array([mapping[s] for s in roi_s])
    roi_it = np.array([mapping[t] for t in roi_t])
    # return rois names from roi
    # the index for each edge roi
    # and the mapping
    return roi_s, roi_t, roi_is, roi_it, areas, mapping



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


def remove_same_roi(df):

    rois = df.roi.values
    roi_s, roi_t = _extract_roi(rois, "-")
    return df.iloc[~(roi_s == roi_t), :]

def xr_remove_same_roi(xar):
    
    roi_s, roi_t = _extract_roi(xar.roi.data, "-")
    return xar.isel(roi=~(roi_s == roi_t))

def xr_remove_same_roi_mc(xar):
    
    roi_s, roi_t = _extract_roi(xar.roi.data, "~")
    
    roi_s_1, roi_s_2 = _extract_roi(roi_s, "-")
    roi_t_1, roi_t_2 = _extract_roi(roi_t, "-")
    
    idx1 = (roi_s == roi_t)
    idx2 = (roi_s_1 == roi_t_1)
    idx3 = (roi_s_2 == roi_t_2)
    
    idx = np.logical_or(idx1, idx2)
    idx = np.logical_or(idx, idx3)
    
    return xar.isel(roi=~idx)

def remove_sca(df):
    
    sca = ["Caudate", "Claustrum", "Thal", "Putamen"]
    roi_s, roi_t = _extract_roi(df.roi.values, "-")
    idx = np.logical_or([s in sca for s in roi_s],
                         [t in sca for t in roi_t])
    return df.iloc[~idx, :]

def node_remove_sca(df):
    
    sca = ["Caudate", "Claustrum", "Thal", "Putamen"]
    idx = np.array([r in sca for r in df.roi.values])
    return df.iloc[~idx, :]

def node_xr_remove_sca(xar):
    
    sca = ["Caudate", "Claustrum", "Thal", "Putamen"]
    idx = np.array([r in sca for r in xar.roi.data])
    return xar.isel(roi=~idx)

def edge_xr_remove_sca(xar):
    
    sca = ["Caudate", "Claustrum", "Thal", "Putamen"]
    roi_s, roi_t = _extract_roi(xar.roi.data, "-")
    idx = np.logical_or([s in sca for s in roi_s],
                         [t in sca for t in roi_t])
    return xar.isel(roi=~idx)

def mc_edge_xr_remove_sca(xar):
    
    sca = ["Caudate", "Claustrum", "Thal", "Putamen"]
    roi_s, roi_t = _extract_roi(xar.roi.data, "~")
    
    roi_s_1, roi_s_2 = _extract_roi(roi_s, "-")
    roi_t_1, roi_t_2 = _extract_roi(roi_t, "-")
    
    idx_1 = np.logical_or([s in sca for s in roi_s_1],
                         [t in sca for t in roi_t_1])
    
    idx_2 = np.logical_or([s in sca for s in roi_s_2],
                          [t in sca for t in roi_t_2])
    
    
    return xar.isel(roi=~np.logical_or(idx_1, idx_2))

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)
