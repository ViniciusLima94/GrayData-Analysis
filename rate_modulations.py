import os
import argparse

import numpy as np
import xarray as xr
from tqdm import tqdm
from config import get_dates
from frites.utils import parallel_func
from GDa.util import shuffle_along_axis
from GDa.loader import loader

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to load", type=int)
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("ALIGNED", help="wheter power was align to cue or match", type=str)
parser.add_argument("THR", help="which threshold value to use", type=float)
parser.add_argument("DECIM", help="decimation used on the power", type=int)

args = parser.parse_args()

sid = args.SIDX
at = args.ALIGNED
monkey = args.MONKEY
thr = args.THR
decim = args.DECIM

# early_cue, early_delay = return_delay_split(monkey=monkey, delay_type=ds)
sessions = get_dates(monkey)
s_id = sessions[sid]

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser("~/funcog/gda/")
_SAVE = os.path.join(_ROOT, "Results", monkey, "rate_modulations")

##############################################################################
# Utility functions
###############################################################################


def bootstrap(ts_stacked, n_trials, n_rois, n_boot):
    """
    Performs the bootstrap method on the given data.

    Parameters
    ----------
    n_boot : int
        The number of iterations for the bootstrap method
    ts_stacked : ndarray
        The time series data in stacked format
    n_trials : int
        The number of trials in the time series data
    n_rois : int
        The number of regions of interest in the time series data
    verbose : bool, optional
        Whether to print the iteration number, by default False

    Returns
    -------
    ndarray
        The bootstrapped confidence interval
    """
    trials = np.arange(n_trials, dtype=int)
    ci = []
    for i in range(n_boot):
        ci += [
            np.take_along_axis(
                ts_stacked,
                np.asarray([np.random.choice(trials, n_trials) for _ in range(n_rois)]),
                axis=-1,
            ).mean(-1)
        ]
    ci = np.stack(ci)
    return ci


def compute_median_rate(
    data: xr.DataArray,
    roi: str = None,
    thr: float = 0.95,
    stim_label: int = None,
    freqs: float = None,
    time_slice: slice = None,
    # freqs: float = None,
    n_boot: int = 100,
    n_jobs: int = 1,
    verbose: bool = False,
):
    """
    Calculates the median rate of the provided xarray DataArray, subject to optional parameters.

    Parameters:
    ----------
        data: xr.DataArray
            The input data for which the median rate is to be calculated.
        roi: str | None
            A string specifying the region of interest (ROI) to consider in the analysis.
            If None, all ROIs will be considered.
        thr: (float) | .95
            A float representing the quantile based threshold.
            Any values in 'data' less than this threshold will be set to zero.
        stim_label: int | None
            An int specifying a particular stimulus label to use in the analysis.
            If None, all stimuli will be considered.
        time_slice: slice | None
            A slice object specifying a range of times to consider in the analysis.
            If None, all times will be considered.
        freqs: float | None
            A float specifying a frequency range to use in the analysis.
            If None, all frequencies will be considered.
        n_boot: int | 100
            An int specifying the number of bootstrap resamples to use when calculating
            the median rate.
        verbose: bool | False
            When set to True, the tqdm library will be used to display a progress
            bar while the function is running.

    Returns:
    -------
        Tuple of two xarray.DataArray,
        where the first is the result of median rate calculation of the given data,
        and the second is the median rate calculated by shuffling the data along the first axis.

    """

    # Check coordinates of the DataArray
    np.testing.assert_array_equal(("roi", "freqs", "trials", "times"), data.dims)

    if isinstance(stim_label, int):
        # Get stimulus label from each trial
        stim_labels = data.stim
        # Select trials with the specific label
        idx_trials = stim_labels == stim_label
    else:
        # Otherwise get all trials
        idx_trials = [True] * data.sizes["trials"]

    # data = (data - data.mean("times")) / data.std("times")

    if thr > 0:
        # Compute quantile based threshold
        thr = data.quantile(thr, ("times"))
        # Apply threshold
        data = data >= thr
    else:
        data = (data - data.mean("times")) / data.std("times")
        data = data * (data >= 0)

    # Get time-series for specific trials, roi and time slice
    ts = data.sel(times=time_slice, roi=roi).isel(trials=idx_trials)

    if isinstance(freqs, (int, float)):
        freqs = [freqs]
    else:
        freqs = ts.freqs.data

    times = ts.times.data
    nfreqs = len(freqs)

    def _for_freq(f):
        """
        Compute the bootstrapped confidence interval and surrogate time series data
        for a specific frequency band.

        Parameters:
        -----------
        f: float or int
            The specific frequency band for which to compute the confidence
            interval and surrogate time series data.

        Returns:
        --------
        ci: xr.DataArray
            The bootstrapped confidence interval, in the form of a xr.DataArray,
            with dimensions ('boot', 'times').
        surr: xr.DataArray
            The surrogate time series data, in the form of a xr.DataArray,
            with dimensions ('boot', 'times').
        """

        # Stack rois
        if "roi" in ts.dims:
            ts_stacked = ts.sel(freqs=f).stack(z=("trials", "roi")).data
        else:
            ts_stacked = ts.sel(freqs=f).data.T

        n_rois = ts_stacked.shape[0]
        n_trials = ts_stacked.shape[1]

        ci = bootstrap(ts_stacked, n_trials, n_rois, n_boot)

        surr = []
        for i in tqdm(range(n_boot)) if verbose else range(n_boot):
            surr += [shuffle_along_axis(ts_stacked, 0)]
        surr = np.stack(surr).mean(-1)
        ci = xr.DataArray(ci, dims=("boot", "times"), coords={"times": times})
        surr = xr.DataArray(surr, dims=("boot", "times"), coords={"times": times})

        return ci, surr

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_freq, n_jobs=n_jobs, verbose=verbose, total=nfreqs
    )
    # Compute the single trial coherence
    out = parallel(p_fun(f) for f in freqs)

    ci = [out[i][0] for i in range(nfreqs)]
    surr = [out[i][1] for i in range(nfreqs)]

    ci = xr.concat(ci, "freqs").assign_coords({"freqs": freqs})
    surr = xr.concat(surr, "freqs").assign_coords({"freqs": freqs})

    return ci, surr


def return_burst_prob(power, conditional=False, thr=0.95, verbose=False):
    """
    Computes the burst probability and surrogate burst probability
    for each region of interest (ROI) in the given power dataset.

    Parameters
    ----------
    power : xarray Dataset
        A multi-dimensional dataset containing the power values of
        the time-frequency representation of the time series.
    conditional : bool, optional
        Whether to compute the burst probability and surrogate burst
        probability separately for each stimulus, by default False
    thr : float, optional
        The threshold for burst detection, by default 0.95

    Returns
    -------
    tuple
        A tuple containing the burst probability and surrogate burst
        probability for each ROI, in the format (P_b, SP_b)
    """
    kw_args = dict(
        thr=thr,
        freqs=None,
        stim_label=None,
        time_slice=slice(-0.5, 2.0),
        n_boot=100,
        verbose=False,
        n_jobs=10,
    )

    stim = power.attrs["stim"]
    rois = np.unique(power.roi.values)

    def _for_roi():

        # Rate modulation
        P_b = []
        SP_b = []

        for roi in tqdm(rois) if verbose else rois:
            ci, surr = compute_median_rate(power, roi=roi, **kw_args)
            P_b += [ci]
            SP_b += [surr]

        P_b = xr.concat(P_b, "roi")
        P_b = P_b.assign_coords({"roi": rois})
        SP_b = xr.concat(SP_b, "roi")
        SP_b = SP_b.assign_coords({"roi": rois})

        return P_b, SP_b

    if not conditional:
        return _for_roi()
    else:
        __iter = range(1, 6)
        # Stimulus dependent rate modulation
        P_b_stim = []
        SP_b_stim = []
        for stim in tqdm(__iter) if verbose else __iter:
            kw_args["stim_label"] = stim
            P_b, SP_b = _for_roi()

            P_b_stim += [P_b]
            SP_b_stim += [SP_b]
        P_b_stim = xr.concat(P_b_stim, "stim")
        SP_b_stim = xr.concat(SP_b_stim, "stim")

        return P_b_stim, SP_b_stim


##############################################################################
# Time-resolved rate
###############################################################################

data_loader = loader(_ROOT=_ROOT)

kw_loader = dict(
    session=s_id, aligned_at=at, channel_numbers=False, monkey=monkey, decim=decim
)

power_task = data_loader.load_power(**kw_loader, trial_type=1, behavioral_response=1)
power_fix = data_loader.load_power(**kw_loader, trial_type=2, behavioral_response=0)


# Computes burst probability for task and fixation
P_b_task, SP_b_task = return_burst_prob(power_task, thr=thr / 100)
P_b_fix, SP_b_fix = return_burst_prob(power_fix, thr=thr / 100)


# Computes burst probability for task and fixation
# P_b_task_stim, SP_b_task_stim = return_burst_prob(power_task, conditional=True, thr=thr / 100)

# percentile = int(thr * 100)
percentile = int(thr)

P_b_task.to_netcdf(os.path.join(_SAVE, f"P_b_task_{s_id}_at_{at}_q_{percentile}.nc"))
SP_b_task.to_netcdf(os.path.join(_SAVE, f"SP_b_task_{s_id}_at_{at}_q_{percentile}.nc"))

P_b_fix.to_netcdf(os.path.join(_SAVE, f"P_b_fix_{s_id}_at_{at}_q_{percentile}.nc"))
SP_b_fix.to_netcdf(os.path.join(_SAVE, f"SP_b_fix_{s_id}_at_{at}_q_{percentile}.nc"))

# P_b_task_stim.to_netcdf(
# os.path.join(_SAVE, f"P_b_task_stim_{s_id}_at_{at}_q_{percentile}.nc")
# )
# SP_b_task_stim.to_netcdf(
# os.path.join(_SAVE, f"SP_b_task_stim_{s_id}_at_{at}_q_{percentile}.nc")
# )
