import os

import argparse
import numpy as np
import xarray as xr
import itertools
from frites.conn.conn_tf import _tf_decomp

from scipy.signal import find_peaks
from GDa.session import session
from GDa.util import create_stages_time_grid
from config import sessions

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to run",
                    type=int)
args = parser.parse_args()
# The index of the session to use
idx = args.SIDX

sidx = sessions[idx]

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser('~/funcog/gda')

##############################################################################
# Spectral analysis parameters
##############################################################################

sm_times = 0.3  # In seconds
sm_freqs = 1
sm_kernel = "square"

# Defining parameters
decim = 20  # Downsampling factor
mode = "multitaper"  # Wheter to use Morlet or Multitaper

n_freqs = 60  # How many frequencies to use
freqs = np.linspace(3, 75, n_freqs)  # Frequency array
n_cycles = freqs / 4  # Number of cycles
mt_bandwidth = None


def return_evt_dt(align_at):
    """Return the window in which the data will be loaded
    depending on the alignment"""
    assert align_at in ["cue", "match"]
    if align_at == "cue":
        return [-0.65, 3.00]
    else:
        return [-2.2, 0.65]


##############################################################################
# Loading data
##############################################################################
# Instantiate class
ses = session(
    raw_path=os.path.expanduser("~/funcog/gda/GrayLab/"),
    monkey="lucy",
    date=sidx,
    session=1,
    slvr_msmod=False,
    align_to="cue",
    evt_dt=[-0.65, 3.00],
)

# Read data from .mat files
ses.read_from_mat()

# Filtering by trials
data = ses.filter_trials(trial_type=[1], behavioral_response=[1])

band = slice(26, 43)

##############################################################################
# Channels with large beta power
##############################################################################

w = _tf_decomp(
    data,
    data.attrs["fsample"],
    freqs,
    mode=mode,
    n_cycles=n_cycles,
    mt_bandwidth=None,
    decim=decim,
    kw_cwt={},
    kw_mt={},
    n_jobs=20,
)

w = xr.DataArray(
    (w * np.conj(w)).real,
    name="power",
    dims=("trials", "roi", "freqs", "times"),
    coords=(data.trials.values, data.roi.values,
            freqs, data.time.values[::decim])
)

##############################################################################
# Compute average spectra per time-stage
##############################################################################
# Create masks to separate stages
mask = create_stages_time_grid(data.t_cue_on, data.t_cue_off, data.t_match_on,
                               data.fsample, data.time.values[::decim],
                               data.sizes['trials'], early_delay=0.3)

# Number of observations per stage per trials
n_obs = {}
for key in mask.keys():
    n_obs[key] = mask[key].sum(-1)

w_static = []
for key in mask.keys():
    temp = (w * mask[key][:, None, None, :]).sum('times')
    temp = temp / n_obs[key][:, None, None]
    w_static += [temp]
w_static = xr.concat(w_static, 'times')
w_static = w_static.assign_coords({'times': list(mask.keys())})

##############################################################################
# Compute peaks for each roi
##############################################################################
unique_rois = np.unique(w_static.roi.values)
freqs = w_static.freqs.values

peaks = np.zeros((len(unique_rois), w_static.sizes['freqs']))
peaks = xr.DataArray(peaks, dims=('roi', 'freqs'),
                     coords={'roi': unique_rois,
                             'freqs': freqs})


for i, roi in enumerate(unique_rois):
    counts = np.zeros_like(freqs)
    idx = w_static.roi.values == roi
    w_sel = w_static.isel(roi=idx).stack(T=("trials", "roi")).sel(times="baseline")
    p_idx = []
    for t in range(w_sel.sizes["T"]):
        temp, _ = find_peaks(w_sel[:, t], threshold=1e-8)
        p_idx += [temp]
    p_idx = list(itertools.chain(*p_idx))
    p_idx, c = np.unique(p_idx, return_counts=True)
    counts[p_idx.astype(int)] = c
    peaks[i, :] = counts

# Path to results folder
_RESULTS = os.path.join(_ROOT,
                        "Results/lucy/peaks",
                        f"{sidx}.nc")

peaks.to_netcdf(_RESULTS)
