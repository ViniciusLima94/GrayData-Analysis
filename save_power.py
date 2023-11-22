import os
import argparse
import numpy as np
import xarray as xr

from GDa.session import session
from frites.conn import define_windows
from tqdm import tqdm
from GDa.session import session
from config import mode, freqs, n_cycles, get_dates, return_evt_dt
from frites.conn.conn_tf import _tf_decomp
import scipy

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
def hilbert_spectra(
    data, fsample, freqs, bandwidth, n_jobs=1, verbose=False, kw_filter={}
):
    """
    Compute the Hilbert spectra of a 3D data array.

    Parameters:
    - data (xr.DataArray): Input data array with dimensions ("trials", "roi", "time").
    - fsample (float): Sampling frequency of the data.
    - freqs (array-like): Center frequencies for the spectral analysis.
    - bandwidth (float): Half-width of the frequency bands. n_jobs (int, optional): Number of parallel jobs to run for filtering. Default is 1.
    - verbose (bool, optional): If True, print verbose messages during filtering. Default is False.
    - kw_filter (dict, optional): Additional keyword arguments for the `filter_data` function.

    Returns:
    - xr.DataArray: Hilbert spectra of the input data, with dimensions ("trials", "roi", "freqs", "times").

    Note:
    The input data is filtered into frequency bands centered at the specified frequencies with the given bandwidth.
    The Hilbert transform is then applied to obtain the analytic signal, and the squared magnitude of the analytic
    signal is computed to obtain the Hilbert spectra.

    The resulting DataArray has dimensions representing trials, regions of interest (ROIs), frequency bins, and time points.
    """

    from mne.filter import filter_data

    assert isinstance(data, xr.DataArray)

    dims = data.dims
    coords = data.coords
    attrs = data.attrs

    np.testing.assert_array_equal(dims, ("trials", "roi", "time"))

    lfreqs = np.clip(freqs - bandwidth, 0, np.inf)
    hfreqs = freqs + bandwidth

    bands = np.stack((lfreqs, hfreqs), axis=1)

    data_filtered = []
    for lf, hf in bands:
        data_filtered += [
            filter_data(
                data.values,
                fsample,
                lf,
                hf,
                n_jobs=n_jobs,
                **kw_filter,
                verbose=verbose,
            )
        ]
    data_filtered = np.stack(data_filtered, axis=1)
    hilbert = scipy.signal.hilbert(data_filtered, axis=-1)
    sxx = hilbert * np.conj(hilbert)

    sxx = xr.DataArray(
        sxx.real,
        dims=("trials", "freqs", "roi", "times"),
        coords=(coords["trials"], freqs, coords["roi"], coords["time"]),
    ).transpose("trials", "roi", "freqs", "times")

    return sxx


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
if mode in ["morlet", "multitaper"]:
    sxx = _tf_decomp(
        data,
        data.attrs["fsample"],
        freqs,
        mode=mode,
        n_cycles=n_cycles,
        mt_bandwidth=None,
        decim=decim,
        kw_cwt={},
        kw_mt={},
        n_jobs=5,
    )

    sxx = xr.DataArray(
        (sxx * np.conj(sxx)).real,
        name="power",
        dims=("trials", "roi", "freqs", "times"),
        coords=(data.trials.values, data.roi.values, freqs,
                data.time.values[::decim]),
    )
else:
    sxx = hilbert_spectra(
        data, data.attrs["fsample"], freqs, 4, n_jobs=20,
        verbose=False, kw_filter={}
    )[::decim]

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
