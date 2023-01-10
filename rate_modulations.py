import argparse
import os

import numpy as np
import xarray as xr
from tqdm import tqdm
from config import get_dates, return_delay_split
from GDa.util import average_stages, shuffle_along_axis

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX",
                    help="index of the session to load",
                    type=int)
parser.add_argument("METRIC",
                    help="which network metric to use",
                    type=str)
parser.add_argument("FREQ",
                    help="which freq. band to use",
                    type=int)
parser.add_argument("TT",   help="type of the trial",
                    type=int)
parser.add_argument("BR",   help="behavioral response",
                    type=int)
parser.add_argument("MONKEY", help="which monkey to use",
                    type=str)
parser.add_argument("ALIGNED", help="wheter power was align to cue or match",
                    type=str)
parser.add_argument("DELAY", help="which type of delay split to use",
                    type=int)

args = parser.parse_args()

sid = args.SIDX
metric = args.METRIC
freq = args.FREQ
at = args.ALIGNED
tt = args.TT
br = args.BR
ds = args.DELAY
monkey = args.MONKEY

early_cue, early_delay = return_delay_split(monkey=monkey, delay_type=ds)
sessions = get_dates(monkey)
s_id = sessions[sid]
##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser('~/funcog/gda')
_SAVE = os.path.join(_ROOT, "Results", monkey, "rate_modulations")

##############################################################################
# Utility functions
###############################################################################


def load_session_power(s_id: str, z_score: bool = False, avg: int = 0,
                       roi: Optional[str] = None):

    """
    This function loads and processes power data for a given session.
    It returns a DataArray containing the processed power data,
    as well as the trials and stim data.

    Parameters:
    ----------
    s_id (str): session id
    z_score (bool): whether or not to z-score the power data
    avg (int): Averages power for each period (baseline, cue, delay, match)
    if needed, if not set to 0 it should be set to int value of avg
    roi (str): region of interest to select,
     if not given all regions will be used

    Returns:
    --------
    out (xr.DataArray): the processed power data
    trials (np.ndarray):  the trials data
    stim (np.ndarray): the stim data
    """
    _FILE_NAME = f"power_tt_{tt}_br_{br}_at_{at}.nc"
    path_pow = os.path.join(
        _ROOT, f"Results/{monkey}/{s_id}/session01", _FILE_NAME)
    power = xr.load_dataarray(path_pow)
    if z_score:
        power.values = (power - power.mean("times")) / power.std("times")
    # Averages power for each period (baseline, cue, delay, match) if needed
    out = average_stages(power, avg)

    if isinstance(roi, str):
        out = out.sel(roi=roi)

    # out = out.sel(roi="V1")
    trials, stim = power.trials.data, power.stim

    return out, trials, stim


##############################################################################
# Time-resolved rate
###############################################################################


def compute_median_rate(
    data: xr.DataArray,
    roi: str = None,
    thr: float = 3.0,
    stim_label: int = None,
    time_slice: slice = None,
    freqs: float = None,
    n_boot: int = 100,
    verbose: bool = False,
):

    if isinstance(stim_label, int):
        stim_labels = data.stim
        idx_trials = stim_labels == stim_label
    else:
        idx_trials = [True] * data.sizes["trials"]

    # Apply threshold
    data = data >= thr

    # Get time-series
    ts = data.sel(freqs=freqs, times=time_slice,
                  roi=roi).isel(trials=idx_trials)
    times = ts.times.data
    # Stack rois
    if "roi" in ts.dims:
        ts_stacked = ts.stack(z=("trials", "roi")).data
    else:
        ts_stacked = ts.data.T
    n_rois = ts_stacked.shape[0]
    n_trials = ts_stacked.shape[1]
    # Compute bootstraps
    ci = []
    for i in tqdm(range(n_boot)) if verbose else range(n_boot):
        ci += [
            np.take_along_axis(
                ts_stacked,
                np.asarray(
                    [np.random.choice(range(n_trials), n_trials)
                     for _ in range(n_rois)]
                ),
                axis=-1,
            ).mean(-1)
        ]
    ci = np.stack(ci)

    surr = []
    for i in tqdm(range(n_boot)) if verbose else range(n_boot):
        surr += [shuffle_along_axis(ts_stacked, 0)]

    surr = np.stack(surr).mean(-1)

    ci = xr.DataArray(ci, dims=("boot", "times"), coords={"times": times})
    surr = xr.DataArray(surr, dims=("boot", "times"), coords={"times": times})

    return ci, surr


##############################################################################
# Power
###############################################################################
out, trials, stim = load_session_power(s_id, z_score=True, avg=0)

# Rate modulation
P_b = []
SP_b = []
rois = np.unique(out.roi.values)
time_slice = slice(-0.5, 2.0)
times = out.sel(times=time_slice).times.data

for roi in tqdm(rois):
    pb, spb = compute_median_rate(
        out.copy(),
        roi,
        stim_label=None,
        time_slice=time_slice,
        freqs=freq,
        n_boot=100,
        verbose=False,
    )

    P_b += [pb]
    SP_b += [spb]

# Stimulus dependent rate modulation
P_b_stim = []
SP_b_stim = []
for stim in tqdm(range(1, 6)):
    P_b_s = []
    SP_b_s = []
    rois = np.unique(out.roi.values)
    time_slice = slice(-0.5, 2.0)
    times = out.sel(times=time_slice).times.data

    for roi in rois:
        pb, spb = compute_median_rate(
            out.copy(),
            roi,
            stim_label=stim,
            time_slice=time_slice,
            freqs=freq,
            n_boot=100,
            verbose=False,
        )

        P_b_s += [pb]
        SP_b_s += [spb]

    P_b_stim += [xr.concat(P_b_s, "roi").assign_coords({"roi": rois})]
    SP_b_stim += [xr.concat(SP_b_s, "roi").assign_coords({"roi": rois})]

#
P_b = xr.concat(P_b, "roi")
P_b = P_b.assign_coords({"roi": rois})
SP_b = xr.concat(SP_b, "roi")
SP_b = SP_b.assign_coords({"roi": rois})

P_b_stim = xr.concat(P_b_stim, "stim")
SP_b_stim = xr.concat(SP_b_stim, "stim")

p = P_b_stim.quantile(0.5, "boot")
t_d = P_b.quantile(0.05, "boot")
t_u = P_b.quantile(0.95, "boot")

# Compute RMI
RMI = ((p > t_u) + (p < t_d)).mean("times").mean("stim")

# Save
P_b.to_netcdf(os.path.join(
    _SAVE, f"P_b_{s_id}_tt_{tt}_br_{br}_at_{at}_ds_{ds}_f_{freq}.nc"))
SP_b.to_netcdf(os.path.join(
    _SAVE, f"SP_b_{s_id}_tt_{tt}_br_{br}_at_{at}_ds_{ds}_f_{freq}.nc"))

P_b_stim.to_netcdf(os.path.join(
    _SAVE, f"P_b_stim_{s_id}_tt_{tt}_br_{br}_at_{at}_ds_{ds}_f_{freq}.nc"))
SP_b_stim.to_netcdf(os.path.join(
    _SAVE, f"SP_b_stim_{s_id}_tt_{tt}_br_{br}_at_{at}_ds_{ds}_f_{freq}.nc"))
RMI.to_netcdf(os.path.join(
    _SAVE, f"RMI_{s_id}_tt_{tt}_br_{br}_at_{at}_ds_{ds}_f_{freq}.nc"))
