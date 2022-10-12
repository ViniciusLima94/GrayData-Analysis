import os
import argparse

from mne.filter import filter_data
import numpy as np
import scipy
import xarray as xr
from frites.conn.conn_tf import _tf_decomp
from mne.time_frequency import psd_array_multitaper
from scipy.signal import find_peaks
from scipy.stats import ks_2samp
from tqdm import tqdm

from config import get_dates
from GDa.util import _extract_roi
from GDa.session import session
from GDa.signal.surrogates import trial_swap_surrogates

parser = argparse.ArgumentParser()
parser.add_argument("BW",
                    help="which bandwidth to use in MT",
                    type=int)
parser.add_argument("IDX",
                    help="index of the session to use",
                    type=int)
parser.add_argument("LINEAR",
                    help="whether to use the linear or semi-log power",
                    type=int)
args = parser.parse_args()


bw = args.BW
idx = args.IDX
linear = bool(args.LINEAR)

sessions = get_dates("lucy")
sid = sessions[idx]
print(f"bw = {bw}, session = {sid}")


##############################################################################
# Utils
##############################################################################

def node_xr_remove_sca(xar):
    sca = ["Caudate", "Claustrum", "Thal", "Putamen"]
    _, rois = _extract_roi(xar.roi.data, " ")
    idx = np.array([r in sca for r in rois])
    return xar.isel(roi=~idx)

def flatten(xss):
    return [x for xs in xss for x in xs]


def load_session_data(sid):

    # Instantiate class
    ses = session(
        raw_path=os.path.expanduser("~/funcog/gda/GrayLab/"),
        monkey="lucy",
        date=sid,
        session=1,
        slvr_msmod=False,
        only_unique_recordings=True,
        align_to="cue",
        evt_dt=[-0.65, 1.50],
    )

    # Read data from .mat files
    ses.read_from_mat()

    # Filtering by trials
    data = ses.filter_trials(
        trial_type=[1], behavioral_response=[1])
    # ROIs with channels
    rois = [
        f"{roi} ({channel})"
        for roi, channel in zip(data.roi.data, data.channels_labels)
    ]
    data = data.assign_coords({"roi": rois})

    return node_xr_remove_sca(data)


def detect_peak_frequencies(power=None, prominence=0.01, verbose=False):

    assert power.ndim == 2
    assert isinstance(power, xr.DataArray)

    roi, freqs = power.roi.data, power.freqs.data
    n_roi = len(roi)

    rois = []
    peak_freqs = []
    peak_prominences = []

    __iter = range(n_roi)
    for i in tqdm(__iter) if verbose else __iter:
        peak_index, peak_info = find_peaks(power[i, :], prominence=prominence)
        peak_freqs += [freqs[peak_index]]
        peak_prominences += [peak_info["prominences"]]
        rois += [[roi[i]] * len(peak_index)]

    return peak_freqs, peak_prominences, rois


##############################################################################
# Define paths
##############################################################################
figs_path = f"figures/betagamma/{sid}"

if not os.path.exists(figs_path):
    os.makedirs(figs_path)

save_path = os.path.expanduser(
    f"~/funcog/gda/Results/lucy/harmonics/{sid}/bw{bw}")

if not os.path.exists(save_path):
    os.makedirs(save_path)

##############################################################################
# Setting spectral analysis parameters
##############################################################################

# Defining parameters
decim = 10  # Downsampling factor
mode = "multitaper"  # Wheter to use Morlet or Multitaper
n_freqs = 30  # How many frequencies to use
fc = np.linspace(4, 80, n_freqs)  # Frequency array
n_cycles = fc / 4  # Number of cycles
mt_bandwidth = 4

if linear:
    bands = {
        "theta": [0, 6],
        "alpha": [6, 14],
        "beta_1": [14, 26],
        "beta_2": [26, 43],
        "gamma": [43, 80],
    }
else:
    bands = {
        "theta": [0, 6],
        "alpha": [6, 14],
        "beta_1": [14, 26],
        "beta_2": [26, 43],
        "gamma": [62, 80],
    }

##############################################################################
# Loading data
##############################################################################
data = [load_session_data(sid) for sid in [sid]][0]
print(data.roi.data)
##############################################################################
# Static spectra
##############################################################################
sfreq = data.fsample


power_static, freqs = psd_array_multitaper(
    data, sfreq, fmin=0, fmax=80, bandwidth=bw, n_jobs=20
)

power_static = xr.DataArray(
    power_static,
    dims=("trials", "roi", "freqs"),
    coords=(data.trials, data.roi, freqs),
    name="power",
)

power_static = power_static.mean("trials")

power_static_norm = power_static / power_static.max("freqs")
if not linear:
    power_static_norm = np.log10(power_static)

##############################################################################
# Peaks and peak promincences
##############################################################################
prominence_thr = 0.01
if not linear:
    prominence_thr = 0.044

peak_freqs, peak_prominences, rois = detect_peak_frequencies(
    power_static_norm, prominence=prominence_thr, verbose=True)

has_peak = np.zeros((power_static_norm.sizes["roi"], len(bands)), dtype=bool)

for i in tqdm(range(power_static_norm.sizes["roi"])):
    for peak in peak_freqs[i]:
        for n_band, band in enumerate(bands.keys()):
            if not has_peak[i, n_band]:
                has_peak[i, n_band] = bands[band][0] <= peak <= bands[band][1]

has_peak = xr.DataArray(
    has_peak, dims=("roi", "bands"), coords=(data.roi, list(bands.keys()))
)

peak_freqs = xr.DataArray(
    np.hstack(peak_freqs), dims="roi", coords={"roi": np.hstack(rois)},
    name="peak_freq"
)

peak_prominences = xr.DataArray(
    np.hstack(peak_prominences),
    dims="roi",
    coords={"roi": np.hstack(rois)},
    name="peak_prom",
)

##############################################################################
# Time-frequency multitaper spectra
##############################################################################

# Get areas with both beta and gamma peak
indexes = np.logical_and(has_peak[:, 3], has_peak[:, 4])
power_static = power_static.isel(roi=indexes)
data_sel = data.isel(roi=indexes)

w = _tf_decomp(
    data_sel,
    data.attrs["fsample"],
    fc,
    mode=mode,
    n_cycles=n_cycles,
    mt_bandwidth=None,
    decim=decim,
    kw_cwt={},
    kw_mt={},
    n_jobs=20,
)

w = xr.DataArray(
    w,
    name="power",
    dims=("trials", "roi", "freqs", "times"),
    coords=(data_sel.trials.values, data_sel.roi.values,
            fc, data_sel.time.values[::decim]),
)

# Power time series for beta and gamma bands
beta = slice(28, 43)
gamma = slice(58, 74)

# Power
power = (w * np.conj(w)).real

power_beta = power.sel(freqs=beta).mean("freqs")
power_gamma = power.sel(freqs=gamma).mean("freqs")

# Phase between beta and gamma
phi = np.angle(scipy.signal.hilbert(power_beta)) - np.angle(
    scipy.signal.hilbert(power_gamma)
)
phi = xr.DataArray(phi, dims=power_beta.dims, coords=power_beta.coords)

##############################################################################
# Correlation and surrogate correlation between beta and gamma
##############################################################################

# Z-scored time series
z_beta = (power_beta - power_beta.mean("times")) / power_beta.std("times")
z_gamma = (power_gamma - power_gamma.mean("times")) / power_gamma.std("times")

# Correlation
cc = (z_beta * z_gamma).mean("times")

# Surrogates
CC = []
for i in tqdm(range(100)):

    x1 = trial_swap_surrogates(z_beta.copy(), seed=i + 1000)
    x2 = trial_swap_surrogates(z_gamma.copy(), seed=i + 2000)

    cc_surr = (x1 * x2).mean("times")

    CC += [cc_surr]

CC = xr.DataArray(
    np.vstack(CC),
    dims=("trials", "roi"),
    coords={"roi": z_gamma.roi},
)

p_values = []

for i in tqdm(range(cc.sizes["roi"])):

    p_values += [ks_2samp(cc.isel(roi=i), CC.isel(roi=i),
                          alternative="less", method="exact")[1]]

p_values = xr.DataArray(p_values, dims=("roi"), coords={"roi": cc.roi})

##############################################################################
# Phase between beta and gamma LFP
##############################################################################


def triggered_avg(
    data=None,
    low_pass=None,
    high_pass=None,
    win_size=None,
    height=None,
    find_troughs=None,
    verbose=False,
):

    n_trials, n_roi, n_times = data.shape
    roi = data.roi.data

    # High-pass filtered data
    data_hp = filter_data(
        data, data.fsample, l_freq=low_pass, h_freq=high_pass, verbose=verbose
    )

    data_hp = (data_hp - data_hp.mean()) / data_hp.std()

    # Converts back to DataArray
    data_hp = xr.DataArray(data_hp, dims=data.dims, coords=data.coords)

    win_size = int(win_size * data.fsample)

    data_hp = data_hp.data.swapaxes(0, 1).reshape(n_roi, n_trials * n_times)

    def _for_roi(i):
        if find_troughs:
            peaks, _ = find_peaks(-data_hp[i], height)
        else:
            peaks, _ = find_peaks(data_hp[i], height)
        snipets = np.zeros((len(peaks), 2 * win_size))
        for pidx, idx in enumerate(peaks):
            temp = data_hp[i, (idx - win_size): (idx + win_size)]
            if len(temp) == 2 * win_size:
                snipets[pidx, :] = temp

        return snipets.mean(0)

    snipets = np.stack([_for_roi(i) for i in range(n_roi)])

    times = np.linspace(-win_size, win_size, snipets.shape[1])
    snipets = xr.DataArray(snipets, dims=("roi", "times"),
                           coords=(roi, times))

    return snipets


def cycle_triggered_avg(
    data=None,
    band_b=None,
    band_g=None,
    win_size=None,
    height=None,
    find_troughs=None,
    verbose=False,
):

    n_trials, n_roi, n_times = data.shape
    roi = data.roi.data

    # Beta component
    data_beta = filter_data(
        data, data.fsample, band_b[0], band_b[1], verbose=verbose)
    # Gamma co,ponent
    data_gamma = filter_data(
        data, data.fsample, band_g[0], band_g[1], verbose=verbose)

    # Converts back to DataArray
    data_beta = xr.DataArray(data_beta, dims=data.dims, coords=data.coords)
    data_gamma = xr.DataArray(data_gamma, dims=data.dims, coords=data.coords)

    data_beta = (data_beta - data_beta.mean()) / data_beta.std()
    data_gamma = (data_gamma - data_gamma.mean()) / data_gamma.std()

    win_size = int(win_size * data.fsample)

    data_beta = data_beta.data.swapaxes(
        0, 1).reshape(n_roi, n_trials * n_times)
    data_gamma = data_gamma.data.swapaxes(
        0, 1).reshape(n_roi, n_trials * n_times)

    def _for_roi(i):
        if find_troughs:
            peaks, _ = find_peaks(-data_beta[i], height)
        else:
            peaks, _ = find_peaks(data_beta[i], height)
        snipets = np.zeros((len(peaks), 2 * win_size))
        for pidx, idx in enumerate(peaks):
            temp = data_gamma[i, (idx - win_size): (idx + win_size)]
            if len(temp) == 2 * win_size:
                snipets[pidx, :] = temp

        return snipets.mean(0)

    snipets = np.stack([_for_roi(i) for i in range(n_roi)])
    times = np.linspace(-win_size, win_size, snipets.shape[1])
    snipets = xr.DataArray(snipets, dims=("roi", "times"),
                           coords=(roi, times))

    return snipets


# Compute cycle average for band passed LFP
snipets = triggered_avg(
    data=data_sel,
    low_pass=15,
    high_pass=140,
    win_size=0.05,
    height=3,
    find_troughs=True,
    verbose=False,
)

snipets_gamma = cycle_triggered_avg(
    data=data_sel,
    band_b=(15, 140),
    band_g=(55, 75),
    win_size=0.05,
    height=3,
    find_troughs=True,
    verbose=False,
)

##############################################################################
# Save data
##############################################################################

# Save frequency of detected peaks
peak_freqs.to_netcdf(os.path.join(save_path, f"peak_freqs_linear_{linear}.nc"))
# Save prominence of detected peaks
peak_prominences.to_netcdf(os.path.join(
    save_path, f"peak_prominences_linear_{linear}.nc"))
# Save which channes has peaks in each band
has_peak.to_netcdf(os.path.join(save_path, f"has_peak_linear_{linear}.nc"))
# Save LFP data for channels with beta and gamma peaks
data_sel.to_netcdf(os.path.join(save_path, f"data_sel_linear_{linear}.nc"))
# Power time series for beta band
power_static.to_netcdf(os.path.join(
    save_path, f"power_static_linear_{linear}.nc"))
# Power time series for beta band
power_beta.to_netcdf(os.path.join(save_path, f"power_beta_linear_{linear}.nc"))
# Power time series for gamma band
power_gamma.to_netcdf(os.path.join(
    save_path, f"power_gamma_linear_{linear}.nc"))
# Correlation between beta and gamma power time-series
cc.to_netcdf(os.path.join(save_path, f"cc_linear_{linear}.nc"))
# Surrogate correlation between beta and gamma power time-series
CC.to_netcdf(os.path.join(save_path, f"cc_surr_linear_{linear}.nc"))
# P-values
p_values.to_netcdf(os.path.join(save_path, f"pvalues_linear_{linear}.nc"))
# Phase between beta and gamma correlation
phi.to_netcdf(os.path.join(save_path, f"phi_linear_{linear}.nc"))
# Beta band LFP triggered average
snipets.to_netcdf(os.path.join(save_path, f"snipets_linear_{linear}.nc"))
# Gamma band LFP triggered average
snipets_gamma.to_netcdf(os.path.join(
    save_path, f"snipets_gamma_linear_{linear}.nc"))
