from GDa.flatmap.flatmap import flatmap
from GDa.util import _extract_roi
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import xarray as xr
from frites.conn.conn_tf import _tf_decomp
from mne.time_frequency import psd_array_multitaper
from mne.filter import filter_data
from scipy.signal import find_peaks
from tqdm import tqdm

from config import sessions
from GDa.session import session


idx = int(sys.argv[-1])
sid = sessions[idx]


def flatten(xss):
    return [x for xs in xss for x in xs]


# Define path to sabe figures
figs_path = f"figures/betagamma/{sid}"

if not os.path.exists(figs_path):
        os.makedirs(figs_path)

##############################################################################
# Setting spectral analysis parameters
##############################################################################

decim = 20  # Downsampling factor
mode = "multitaper"  # Wheter to use Morlet or Multitaper
n_freqs = 100  # How many frequencies to use
fc = np.linspace(4, 80, n_freqs)  # Frequency array
n_cycles = fc / 4  # Number of cycles
mt_bandwidth = None

bands = {
    "theta": [0, 6],
    "alpha": [6, 14],
    "beta_1": [14, 26],
    "beta_2": [26, 43],
    "gamma": [43, 80],
}

##############################################################################
# Loanding data
##############################################################################
def load_session_data(sid):

    # Instantiate class
    ses = session(
        raw_path=os.path.expanduser("~/funcog/gda/GrayLab/"),
        monkey="lucy",
        date=sid,
        session=1,
        slvr_msmod=False,
        align_to="cue",
        evt_dt=[-0.65, 3.00],
    )

    # Read data from .mat files
    ses.read_from_mat()

    # Filtering by trials
    data = ses.filter_trials(trial_type=[1], behavioral_response=[1])
    # ROIs with channels
    rois = [
        f"{roi} ({channel})"
        for roi, channel in zip(data.roi.data, data.channels_labels)
    ]
    data = data.assign_coords({"roi": rois})

    return data


data = load_session_data(sid)

##############################################################################
# Computing trial averaged static spectra
##############################################################################

sfreq = data.fsample

power_static, freqs = psd_array_multitaper(
    data, sfreq, fmin=0, fmax=80, bandwidth=1, n_jobs=20
)

power_static = xr.DataArray(
    power_static,
    dims=("trials", "roi", "freqs"),
    coords=(data.trials, data.roi, freqs),
    name="power",
)

power_static = power_static.mean("trials")
power_static = power_static / power_static.max("freqs")

##############################################################################
# Distribution of peak promininces
##############################################################################


def detect_peak_frequencies(power=None, prominence=0.01, verbose=False):

    assert power.ndim == 2
    assert isinstance(power, xr.DataArray)

    roi, freqs = power.roi.data, power.freqs.data
    n_roi = len(roi)

    peak_freqs = []
    peak_prominences = []

    __iter = range(n_roi)
    for i in tqdm(__iter) if verbose else __iter:
        peak_index, peak_info = find_peaks(power[i, :], prominence=prominence)
        peak_freqs += [freqs[peak_index]]
        peak_prominences += [peak_info["prominences"]]

    return peak_freqs, peak_prominences


peak_freqs, peak_prominences = detect_peak_frequencies(
    power_static, verbose=True)

has_peak = np.zeros((power_static.sizes["roi"], len(bands)), dtype=bool)

for i in tqdm(range(power_static.sizes["roi"])):
    for peak in peak_freqs[i]:
        for n_band, band in enumerate(bands.keys()):
            if not has_peak[i, n_band]:
                has_peak[i, n_band] = bands[band][0] <= peak <= bands[band][1]

has_peak = xr.DataArray(
    has_peak, dims=("roi", "bands"), coords=(data.roi, list(bands.keys()))
)

##############################################################################
# Time-frequency multitaper spectra
##############################################################################

w = _tf_decomp(
    data,
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
    coords=(data.trials.values, data.roi.values,
            fc, data.time.values[::decim]),
)

beta = slice(28, 38)
gamma = slice(60, 70)

# Power
power = (w * np.conj(w)).real

# Power time series in hgamma and beta band
power_beta = power.sel(freqs=beta).mean("freqs")
power_gamma = power.sel(freqs=gamma).mean("freqs")

# Select channels with peaks in both beta and gamma bands 
indexes = np.logical_and(has_peak[:, 3], has_peak[:, 4])
power_beta = power_beta.isel(roi=indexes)
power_gamma = power_gamma.isel(roi=indexes)

# Phase beta and gamma
phi = np.angle(scipy.signal.hilbert(power_beta)) - np.angle(
    scipy.signal.hilbert(power_gamma)
)
phi = xr.DataArray(phi, dims=power_beta.dims, coords=power_beta.coords)

# z-scored and max norm time-series
z_beta = (power_beta - power_beta.mean("times")) / power_beta.std("times")
z_gamma = (power_gamma - power_gamma.mean("times")) / power_gamma.std("times")
max_beta = power_beta / power_beta.max(("times", "trials"))
max_gamma = power_gamma / power_gamma.max(("times", "trials"))

cc = (z_beta * z_gamma).mean("times")

trials_shuffle = z_gamma.trials.data

CC = []

### pseudorandom
for i in tqdm(range(1000)):

    trials_shuffle = z_gamma.trials.data.copy()

    np.random.seed(i + 1000)

    np.random.shuffle(trials_shuffle)

    x1 = z_beta.sel(trials=trials_shuffle).values

    trials_shuffle = z_gamma.trials.data.copy()

    x2 = z_gamma.sel(trials=np.roll(trials_shuffle, 30)).values

    cc_surr = (x1 * x2).mean(-1)

    CC += [cc_surr]

CC = xr.DataArray(
    np.vstack(CC),
    dims=("trials", "roi"),
    coords={"roi": z_gamma.roi},
)

# df = pd.concat(
    # [
        # z_beta.to_dataframe("power_beta").reset_index(),
        # z_gamma.to_dataframe("power_gamma").reset_index(),
        # phi.to_dataframe("phi").reset_index(),
    # ],
    # axis=1,
# )

# df = df.loc[:, ~df.columns.duplicated()].copy()

cc_time = z_beta * z_gamma

has_beta_gamma = has_peak.sel(bands=['beta_2', 'gamma']).sum('bands') == 2

roi_idx = has_beta_gamma.roi[has_beta_gamma].data

try:
    roi_idx = np.random.choice(roi_idx, 3, replace=False)
except ValueError:
    roi_idx = np.random.choice(roi_idx, 3)

power_static_plot = power_static.sel(
    roi=roi_idx
)

_, areas = _extract_roi(has_beta_gamma.roi[has_beta_gamma].roi.data, " ")
areas = np.unique(areas)
print(areas)

fig = plt.figure(figsize=(10.0, 5.0), dpi=150)

gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.05,
                       right=0.35, bottom=0.55, top=0.92)
gs2 = fig.add_gridspec(nrows=1, ncols=1, left=0.35,
                       right=0.60, bottom=0.55, top=0.92)
gs3 = fig.add_gridspec(nrows=3, ncols=1, left=0.67,
                       right=0.95, bottom=0.55, top=0.92)
gs4 = fig.add_gridspec(nrows=1, ncols=1, left=0.05,
                       right=0.6, bottom=0.1, top=0.4)
gs5 = fig.add_gridspec(nrows=1, ncols=1, left=0.65,
                       right=0.95, bottom=0.1, top=0.4)

# Panel A
ax1 = plt.subplot(gs1[0])

# Panels C-E
ax2 = plt.subplot(gs2[0])

# Panel B
ax3 = plt.subplot(gs3[0])
ax4 = plt.subplot(gs3[1])
ax5 = plt.subplot(gs3[2])

ax6 = plt.subplot(gs4[0])

ax7 = plt.subplot(gs5[0])


plt.sca(ax1)
plt.plot(np.hstack(peak_freqs), np.hstack(peak_prominences), ".", ms=5)
plt.hlines(0.01, 0, 80, "r", ls="--", lw=0.5)
plt.ylim([-0.01, 1.01])
plt.xlim([4, 80.01])
plt.ylabel("Prominence")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


fmap = flatmap([1] * len(areas), [a.lower() for a in areas])
fmap.plot(ax=ax2, colormap="binary_r")

plt.sca(ax3)
plt.plot(power_static_plot.freqs, power_static_plot.isel(roi=0), "b")
plt.xticks([])
plt.yticks([])
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["bottom"].set_visible(False)
ax3.spines["left"].set_visible(False)
plt.ylim(0, 1)
ax3.legend([power_static_plot.isel(roi=0).roi.data],
           fontsize=10, frameon=False)
plt.sca(ax4)
plt.plot(power_static_plot.freqs, power_static_plot.isel(roi=1), "b")
plt.xticks([])
plt.yticks([])
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.spines["bottom"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax4.legend([power_static_plot.isel(roi=1).roi.data],
           fontsize=10, frameon=False)
plt.ylim(0, 1)
plt.sca(ax5)
plt.plot(power_static_plot.freqs, power_static_plot.isel(roi=2), "b")
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax5.legend([power_static_plot.isel(roi=2).roi.data],
           fontsize=10, frameon=False)
plt.ylabel("power", fontsize=12)
plt.ylim(-0.1, 1)
plt.yticks([0, 1])
plt.xlabel("Frequency [Hz]")

##############################################################################
plt.sca(ax6)
df = cc.to_dataframe(name="cc").reset_index()
df = df.sort_values("cc", ascending=False)

ax6 = sns.boxplot(
    x=df["roi"],
    y=df["cc"],
    color="cyan",
    medianprops=dict(color="b", label="median"),
)

rois = [t.get_text() for t in ax6.get_xticklabels()]

CC = CC.sel(roi=rois)

mean = CC.median("trials")
std1 = CC.quantile(0.05, "trials")
std2 = CC.quantile(0.95, "trials")

plt.plot(range(cc.sizes["roi"]), mean, "k-")

plt.fill_between(
    range(cc.sizes["roi"]),
    std1,
    std2,
    alpha=0.2,
    color="gray",
)

plt.xticks(rotation=45)
plt.xlabel("")
ax6.spines["right"].set_visible(False)
ax6.spines["top"].set_visible(False)
plt.xlim(-1, cc.sizes["roi"])
plt.title(r"Power-correlation between $\beta$ and $\gamma$ band distribution")
ax6.spines["top"].set_visible(False)
ax6.spines["right"].set_visible(False)

plt.suptitle(f"session {sid}", x=0.5, y=0.97, fontsize=12)

plt.sca(ax7)
sns.scatterplot(
    x=cc_time.data.flatten(),
    y=phi.data.flatten(),
    s=np.abs(cc_time.data.flatten()),
    c=phi.data.flatten(),
    palette="flare",
)
plt.xlabel("Strength of cofluctuations")
plt.ylabel("Phase")
ax7.spines["top"].set_visible(False)
ax7.spines["right"].set_visible(False)

plt.savefig(os.path.join(figs_path, f"beta_gamma_{sid}.png"))
plt.close()


##############################################################################
# Triggered averaged gamma cycles plots
##############################################################################

data = data.sel(roi=indexes)


def triggered_avg(
    data=None,
    high_pass=None,
    win_size=None,
    height=None,
    find_troughs=None,
    verbose=False,
):

    n_trials, n_roi, n_times = data.shape

    # High-pass filtered data
    data_hp = filter_data(data, data.fsample,
                          l_freq=high_pass, h_freq=None, verbose=verbose)

    # Converts back to DataArray
    data_hp = xr.DataArray(data_hp, dims=data.dims, coords=data.coords)

    win_size = int(win_size * data.fsample)

    data_hp = data_hp.data.swapaxes(0, 1).reshape(n_roi, n_trials * n_times)
    if find_troughs:
        data_hp = -data_hp

    def _for_roi(i):
        peaks, _ = find_peaks(data_hp[i], height)
        snipets = np.zeros((len(peaks), 2 * win_size))
        for pidx, idx in enumerate(peaks):
            temp = data_hp[i, (idx - win_size): (idx + win_size)]
            if len(temp) == 2 * win_size:
                snipets[pidx, :] = temp

        return snipets

    snipets = [_for_roi(i) for i in range(n_roi)]

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

    # Beta component
    data_beta = filter_data(
        data, data.fsample, band_b[0], band_b[1], verbose=verbose)
    # Gamma co,ponent
    data_gamma = filter_data(
        data, data.fsample, band_g[0], band_g[1], verbose=verbose)

    # Converts back to DataArray
    data_beta = xr.DataArray(data_beta, dims=data.dims, coords=data.coords)
    data_gamma = xr.DataArray(data_gamma, dims=data.dims, coords=data.coords)

    win_size = int(win_size * data.fsample)

    data_beta = data_beta.data.swapaxes(
        0, 1).reshape(n_roi, n_trials * n_times)
    if find_troughs:
        data_beta = -data_beta

    data_gamma = data_gamma.data.swapaxes(
        0, 1).reshape(n_roi, n_trials * n_times)

    def _for_roi(i):
        peaks, _ = find_peaks(data_beta[i], height)
        snipets = np.zeros((len(peaks), 2 * win_size))
        for pidx, idx in enumerate(peaks):
            temp = data_gamma[i, (idx - win_size): (idx + win_size)]
            if len(temp) == 2 * win_size:
                snipets[pidx, :] = temp

        return snipets

    snipets = [_for_roi(i) for i in range(n_roi)]

    return snipets


# High-passed signal
snipets_hp = triggered_avg(
    data=data,
    high_pass=15,
    win_size=0.1,
    height=0.00015,
    find_troughs=True
)

# Gamma averaged cycles
snipets = cycle_triggered_avg(
    data=data,
    band_b=(28, 42),
    band_g=(58, 72),
    win_size=0.015,
    height=0.00015,
    find_troughs=True,
)

n_channel = len(snipets)

plt.figure(figsize=(15, 6))
for r in range(n_channel):
    plt.subplot(2, n_channel, r + 1)
    for i in range(len(snipets_hp[r])):
        plt.plot(np.arange(-100, 100), snipets_hp[r][i], "b", lw=0.05)
    plt.plot(np.arange(-100, 100), snipets_hp[r].mean(0), "k", lw=4)
    plt.title(f"{data.roi.data[r]}")


for r in range(n_channel):
    plt.subplot(2, n_channel, n_channel + r + 1)
    for i in range(len(snipets[r])):
        plt.plot(np.arange(-15, 15), snipets[r][i], "b", lw=0.05)
    plt.plot(np.arange(-15, 15), snipets[r].mean(0), "k", lw=4)
    plt.title(f"{data.roi.data[r]}")
plt.tight_layout()

plt.savefig(os.path.join(figs_path, f"snipets_{sid}.png"))
plt.close()

