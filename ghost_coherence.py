import os

import argparse
import numpy as np
import pandas as pd
import xarray as xr
from frites.conn.conn_spec import conn_spec
from frites.conn.conn_tf import _tf_decomp
from mne.time_frequency import psd_array_multitaper
from frites.utils import parallel_func
from scipy.signal import find_peaks
from tqdm import tqdm
from config import sessions

from fooof import FOOOFGroup
from fooof.analysis import get_band_peak_fg
from fooof.bands import Bands

from GDa.session import session
from GDa.signal.surrogates import trial_swap_surrogates
from GDa.util import _extract_roi


###############################################################################
# Argument parsing
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to run",
                    type=int)
parser.add_argument("METRIC", help="which dFC metric to use",
                    type=str)
args = parser.parse_args()
# The index of the session to use
sidx = args.SIDX
# Get name of the dFC metric
metric = args.METRIC


###############################################################################
# Define function to perform the analysis
###############################################################################
def detect_peaks(
    data, norm=None, kw_peaks={}, return_value=None, verbose=False, n_jobs=1
):

    assert isinstance(data, xr.DataArray)
    np.testing.assert_array_equal(data.dims, ["trials", "roi", "freqs"])

    # Names of properties in kw_peaks
    p_names = ["".join(list(key)) for key in kw_peaks.keys()]

    if norm:
        assert norm in ["max", "area"]
        if norm == "max":
            norm_values = data.max("freqs")
        else:
            norm_values = data.integrate("freqs")
        data = data / norm_values

    n_trials, n_rois = data.sizes["trials"], data.sizes["roi"]

    # Compute for each roi
    def _for_roi(i):
        peaks = np.zeros((n_trials, n_freqs))
        for t in range(n_trials):
            out, properties = find_peaks(data[t, i, :].data, **kw_peaks)
            if return_value is None:
                peaks[t, out] = 1
            else:
                peaks[t, out] = properties[return_value]
        return peaks

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_roi, n_jobs=n_jobs, verbose=verbose, total=n_rois
    )
    # Compute the single trial coherence
    peaks = parallel(p_fun(i) for i in range(n_rois))

    peaks = xr.DataArray(
        np.stack(peaks, 1), dims=data.dims, coords=data.coords,
        name="prominence"
    )

    return peaks


###############################################################################
# Spectral analysis parameters
###############################################################################
# Smoothing windows
sm_times = 0.3  # In seconds
sm_freqs = 1
sm_kernel = "square"

# Defining parameters
decim = 20  # Downsampling factor
mode = "multitaper"  # Wheter to use Morlet or Multitaper

n_freqs = 40  # How many frequencies to use
freqs = np.linspace(3, 75, n_freqs)  # Frequency array
n_cycles = freqs / 4  # Number of cycles
mt_bandwidth = None

###############################################################################
# Loading data
###############################################################################
# Instantiate class
ses = session(
    raw_path=os.path.expanduser("~/funcog/gda/GrayLab/"),
    monkey="lucy",
    date=sessions[sidx],
    session=1,
    slvr_msmod=False,
    align_to="cue",
    evt_dt=[-0.7, 0.00],
)

# Read data from .mat files
ses.read_from_mat()

# Filtering by trials
data = ses.filter_trials(trial_type=[1], behavioral_response=[1])
# ROIs with channels
rois = [
    f"{roi} ({channel})" for roi, channel in zip(data.roi.data, data.channels_labels)
]
data = data.assign_coords({"roi": rois})

data_surr = trial_swap_surrogates(data, seed=123456, verbose=False)

###############################################################################
# Computing power and coherence
###############################################################################

# w = _tf_decomp(
    # data,
    # data.attrs["fsample"],
    # freqs,
    # mode=mode,
    # n_cycles=n_cycles,
    # mt_bandwidth=None,
    # decim=decim,
    # kw_cwt={},
    # kw_mt={},
    # n_jobs=20,
# )

# # Compute power spectra and average over time
# w = (w * np.conj(w)).real.mean(-1)

# w = xr.DataArray(
    # w,
    # name="power",
    # dims=("trials", "roi", "freqs"),
    # coords=(data.trials.values, data.roi.values, freqs),
# )

sfreq = data.fsample

w, fr_psd = psd_array_multitaper(
    data, sfreq, fmin=6, fmax=80, bandwidth=8, n_jobs=20
)

w = xr.DataArray(
    w,
    dims=("trials", "roi", "freqs"),
    coords=(data.trials, data.roi, fr_psd),
    name="power",
)

kw = dict(
    freqs=freqs,
    times="time",
    roi="roi",
    foi=None,
    n_jobs=10,
    pairs=None,
    sfreq=ses.data.attrs["fsample"],
    mode=mode,
    n_cycles=n_cycles,
    decim=decim,
    metric="coh",
    sm_times=sm_times,
    sm_freqs=sm_freqs,
    sm_kernel=sm_kernel,
    block_size=4,
)

# compute the coherence
coh = conn_spec(data, **kw).astype(np.float32, keep_attrs=True).mean("times")
coh_surr = conn_spec(data_surr, **kw).astype(np.float32,
                                             keep_attrs=True).mean("times")

coh = np.clip(coh - coh_surr.quantile(0.95, "trials"), 0, np.inf)
###############################################################################
# Finding peaks in the spectra
###############################################################################

bands = Bands(
    {
        "alpha": [6, 14],
        "beta": [14, 26],
        "high_beta": [26, 43],
        "gamma": [43, 80],
    }
)

# Number of spectra per roi
n_spectra = w.sizes["trials"]
# Frequency axis
freqs = w.freqs.data
# Frequency range
freq_range = [freqs[0], freqs[-1]]
# ROI names
rois = w.roi.data
# Trial labels
trials = w.trials.data

peak_freqs = np.zeros((bands.n_bands, n_spectra, len(rois)))

i = 0
for roi in tqdm(rois):
    spectra = w.isel(roi=i).data
    fg = FOOOFGroup(verbose=False)
    fg.fit(freqs, spectra, n_jobs=-1)
    for b in range(bands.n_bands):
        label = bands.labels[b]
        peak_freqs[b, :, i] = get_band_peak_fg(fg, bands[label])[:, 1]
    i = i + 1

peak_freqs = xr.DataArray(
    peak_freqs, dims=("freqs", "trials", "roi"),
    coords={"trials": trials, "roi": rois}
)


###############################################################################
# Compute peak coherence prominence
###############################################################################

p_coh = detect_peaks(
    coh,
    kw_peaks=dict(height=0.1, prominence=0),
    return_value="prominences",
    norm=None,
)

max_pro_coh = p_coh.max().data

p_coh = detect_peaks(
    coh,
    kw_peaks=dict(height=0.1, prominence=0.1 * max_pro_coh),
    return_value=None,
    norm=None,
)

###############################################################################
# Detecting false positives
###############################################################################

p_band = (peak_freqs > 0).astype(int)

p_coh_band = []
for i, label in enumerate(bands.labels):
    flow, fhigh = bands[label][0], bands[label][1]
    p_coh_band += [(p_coh.sel(freqs=slice(flow, fhigh)
                              ).sum("freqs") > 0).astype(int)]

p_coh_band = xr.concat(p_coh_band, "freqs")

roi_s, roi_t = _extract_roi(p_coh_band.roi.data, "-")

Om = np.zeros((bands.n_bands, n_spectra, len(roi_s)))

for i in tqdm(range(p_coh_band.sizes["roi"])):
    s, t = roi_s[i], roi_t[i]

    G = (p_band.sel(roi=s) + p_band.sel(roi=t)) == 2

    Om[..., i] = p_coh_band.isel(roi=i) * (p_band.sel(roi=s) +
                                           p_band.sel(roi=t)) + p_coh_band.isel(roi=i)

    Om[..., i] = Om[..., i] - (1 - p_coh_band.isel(roi=i)) * G

Om = xr.DataArray(
    Om, dims=("freqs", "trials", "roi"),
    coords={"trials": trials, "roi": p_coh_band.roi.data}
)

###############################################################################
# Compute False Positive Ratio
###############################################################################


def compute_TPR(data):

    TP = (data == 3).sum("trials")
    FN = (data == -1).sum("trials")

    return TP / (TP + FN)


def compute_FPR(data):

    TN = (data == 0).sum("trials")
    FP = (np.logical_or(data == 2, data == 1)).sum("trials")

    return FP / (FP + TN)


def compute_FDR(data):

    TP = (data == 3).sum("trials")
    FP = (np.logical_or(data == 2, data == 1)).sum("trials")

    return FP / (FP + TP)

roi_s, roi_t = _extract_roi(Om.roi.values, "-")

df_list = []
FPR_list = []
TPR_list = []
FDR_list = []

for i, key in enumerate(bands.labels):
    # False Positves
    x = Om.sel(freqs=i)
    # Mean power in the band
    y = w.sel(freqs=slice(bands[key][0], bands[key][1])).mean("freqs")
    # Mean coherence in the band
    z = (
        coh.sel(roi=x.roi, freqs=slice(bands[key][0], bands[key][1]))
        .mean("freqs")
        .to_dataframe("coh")
        .reset_index()
    )

    df = pd.concat(
        [
            x.to_dataframe("Om").reset_index(),
            y.sel(roi=roi_s).to_dataframe("pow_s").reset_index(),
            y.sel(roi=roi_t).to_dataframe("pow_t").reset_index(),
            z,
        ],
        axis=1,
    )
    df = df.loc[:, ~df.columns.duplicated()]
    df.drop("quantile", axis=1, inplace=True)
    df["freqs"] = len(df) * [i]
    df_list += [df]

    FPR = compute_FPR(x)
    FPR = FPR.sel(roi=x.roi.values).to_dataframe("FPR").reset_index()
    FPR["freqs"] = len(FPR) * [i]
    FPR_list += [FPR]

    TPR = compute_TPR(x)
    TPR = TPR.sel(roi=x.roi.values).to_dataframe("TPR").reset_index()
    TPR["freqs"] = len(TPR) * [i]
    TPR_list += [TPR]

    FDR = compute_FDR(x)
    FDR = FDR.sel(roi=x.roi.values).to_dataframe("FDR").reset_index()
    FDR["freqs"] = len(FDR) * [i]
    FDR_list += [FDR]

df = pd.concat(df_list, axis=0)
FPR = pd.concat(FPR_list, axis=0).fillna(0)
FDR = pd.concat(FDR_list, axis=0).fillna(0)
TPR = pd.concat(TPR_list, axis=0).fillna(0)


save_path = os.path.expanduser("~/funcog/gda/Results/lucy/ghost_coherence")
# Om.to_netcdf(os.path.join(save_path, f"om_{metric}_{sessions[sidx]}.nc"))
df.to_csv(os.path.join(save_path, f"om_{metric}_{sessions[sidx]}.csv"))
FPR.to_csv(os.path.join(save_path, f"FPR_{metric}_{sessions[sidx]}.csv"))
TPR.to_csv(os.path.join(save_path, f"TPR_{metric}_{sessions[sidx]}.csv"))
FDR.to_csv(os.path.join(save_path, f"FDR_{metric}_{sessions[sidx]}.csv"))
