import os
import argparse
import numpy as np
import xarray as xr

from frites.conn import conn_io
from GDa.session import session
from config import mode, bands, get_dates, return_evt_dt
import scipy
from mne.filter import filter_data

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to be run", type=int)
parser.add_argument("TT", help="type of the trial", type=int)
parser.add_argument("BR", help="behavioral response", type=int)
parser.add_argument("ALIGN", help="wheter to align data to cue or match", type=str)
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("DECIM", help="downsample factor", type=int)

args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX
tt = args.TT
br = args.BR
at = args.ALIGN
monkey = args.MONKEY
decim = args.DECIM

sessions = get_dates(monkey)

s_id = sessions[idx]

# Root directory
_ROOT = os.path.expanduser("~/funcog/gda")


###############################################################################
# Define Hilbert spectra method
###############################################################################
def hilbert_decomposition(
    data,
    sfreq=None,
    times=None,
    roi=None,
    bands=None,
    n_jobs=1,
    verbose=None,
    dtype=np.float32,
):
    """
    Perform Hilbert decomposition on time-series data to extract power, phase, and phase differences.

    Parameters
    ----------
    data : xarray.DataArray or) ndarray
        The input time-series data. Expected to be in the shape (n_trials, n_rois, n_times).
    sfreq : float, optional
        The sampling frequency of the data in Hz. Required for band-pass filtering.
    times : ndarray, optional
        The time points corresponding to the data samples.
    roi : list or ndarray, optional
        The regions of interest (ROIs) to analyze. Can be indices or names corresponding to the data.
    bands : list of tuple, optional
        The frequency bands to filter the data. Each tuple should contain the low and high frequency (in Hz) of the band.
    n_jobs : int, optional
        The number of parallel jobs to use for computations (default is 1).
    verbose : bool or int, optional
        Verbosity level for logging.
    dtype : data-type, optional
        The data type of the returned arrays (default is np.float32).
    **kw_links : dict
        Additional arguments for connection analysis.

    Returns
    -------
    power : xarray.DataArray
        Power time-series of the filtered signals across the specified frequency bands.
        Dimensions: (n_trials, n_rois, n_freqs, n_times).

    Notes
    -----
    The Hilbert decomposition is applied after band-pass filtering the data to extract the analytic signal,
    from which power and phase are derived. Pairwise phase differences are computed in parallel
    for specified pairs of ROIs.
    """
    # ________________________________ INPUTS _________________________________
    # inputs conversion
    data, cfg = conn_io(
        data,
        times=times,
        roi=roi,
        agg_ch=False,
        win_sample=None,
        sfreq=sfreq,
        verbose=verbose,
        name="Hilbert Decomposition",
        kw_links={},
    )

    # Extract variables
    x, trials, attrs = data.data, data["y"].data, cfg["attrs"]
    times, _ = data["times"].data, len(trials)
    x_s, x_t, roi_p, roi = cfg["x_s"], cfg["x_t"], cfg["roi_p"], data["roi"].data
    _, sfreq = cfg["blocks"], cfg["sfreq"]
    n_pairs, f_vec, n_freqs = len(x_s), np.mean(bands, axis=1), len(bands)
    # If no bands are passed use broadband signal

    _dims = ("trials", "roi", "freqs", "times")
    _coord_nodes = (trials, roi, f_vec, times)
    _coord_links = (trials, roi_p, f_vec, times)

    # Filter data in the specified bands
    x_filt = []
    print(x.shape)

    for f_low, f_high in bands:
        x_filt += [
            xr.DataArray(
                filter_data(
                    x,
                    sfreq,
                    f_low,
                    f_high,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    method="iir",
                ),
                dims=data.dims,
                coords=data.coords,
                attrs=attrs,
            )
        ]

    x_filt = xr.concat(x_filt, "freqs").transpose("trials", "roi", "freqs", "times")

    # Hilbert coefficients
    h = scipy.signal.hilbert(x_filt, axis=3)

    # Power and phase time-series
    power = (h * np.conj(h)).real

    # Wrapp to xrray
    power = xr.DataArray(
        power, dims=_dims, coords=_coord_nodes, attrs=attrs, name="power"
    )

    return power.astype(dtype)


###########################################################################
# Loading session
###########################################################################

# Window in which the data will be read
evt_dt = return_evt_dt(at, monkey=monkey)
# Path to LFP data
raw_path = os.path.expanduser("~/funcog/gda/GrayLab/")
# Instantiate class
ses = session(
    raw_path=raw_path,
    monkey=monkey,
    date=s_id,
    session=1,
    slvr_msmod=True,
    align_to=at,
    evt_dt=evt_dt,
)

# Read data from .mat files
ses.read_from_mat()

# Filtering by trials
if tt == 2 or tt == 3:
    data = ses.filter_trials(trial_type=[tt], behavioral_response=None)
else:
    data = ses.filter_trials(trial_type=[tt], behavioral_response=[br])

###########################################################################
# Compute power spectra
###########################################################################
# if mode in ["morlet", "multitaper"]:
#    sxx = _tf_decomp(
#        data,
#        data.attrs["fsample"],
#        freqs,
#        mode=mode,
#        n_cycles=n_cycles,
#        mt_bandwidth=None,
#        decim=decim,
#        kw_cwt={},
#        kw_mt={},
#        n_jobs=15,
#    )
#
#    sxx = xr.DataArray(
#        (sxx * np.conj(sxx)).real,
#        name="power",
#        dims=("trials", "roi", "freqs", "times"),
#        coords=(data.trials.values, data.roi.values, freqs, data.time.values[::decim]),
#    )
# elif mode == "hilbert":
#    sxx = hilbert_spectra(
#        data, data.attrs["fsample"], freqs, 4, n_jobs=20, verbose=False, kw_filter={}
#    )[..., ::decim]

sxx = hilbert_decomposition(
    data, data.attrs["fsample"], "time", "roi", bands["lucy"], 15, True
)

# sm_times = int(np.round(0.1 * data.attrs["fsample"]  / decim))
# kernel = _create_kernel(sm_times, 1)
# sxx.values = _smooth_spectra(sxx.values, kernel, scale=False, decim=1)

###########################################################################
# Saves file
###########################################################################

# Path in which to save coherence data
results_path = os.path.join(_ROOT, "Results", monkey, s_id, "session01")
# Create results path in case it does not exist
if not os.path.exists(results_path):
    os.makedirs(results_path)

file_name = f"power_tt_{tt}_br_{br}_at_{at}_decim_{decim}_{mode}.nc"
path_pow = os.path.join(results_path, file_name)

sxx.attrs = data.attrs
sxx.attrs["evt_dt"] = evt_dt
sxx.to_netcdf(path_pow)
