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
from scipy.signal import find_peaks
from tqdm import tqdm

from config import sessions
from GDa.session import session


idx = int(sys.argv[-1])
sid = sessions[idx]


def flatten(xss):
    return [x for xs in xss for x in xs]

#### Setting spectral analysis parameters


# Defining parameters
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

#### Loanding data


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

#### Computing trial averaged static spectra

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

#### Distribution of peak promininces


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

#### Time-frequency multitaper spectra

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

power_beta = power.sel(freqs=beta).mean("freqs")
power_gamma = power.sel(freqs=gamma).mean("freqs")

indexes = np.logical_and(has_peak[:, 3], has_peak[:, 4])
power_beta = power_beta.isel(roi=indexes)
power_gamma = power_gamma.isel(roi=indexes)

# Phase beta and gamma
phi = np.angle(scipy.signal.hilbert(power_beta)) - np.angle(
    scipy.signal.hilbert(power_gamma)
)
phi = xr.DataArray(phi, dims=power_beta.dims, coords=power_beta.coords)

z_beta = (power_beta - power_beta.mean("times")) / power_beta.std("times")
z_gamma = (power_gamma - power_gamma.mean("times")) / power_gamma.std("times")

cc = (z_beta * z_gamma).mean("times")

trials_shuffle = z_gamma.trials.data

CC = []

for i in tqdm(range(1000)):

    np.random.shuffle(trials_shuffle)

    x1 = z_beta.sel(trials=trials_shuffle)

    np.random.shuffle(trials_shuffle)

    x2 = z_gamma.sel(trials=trials_shuffle)

    cc_surr = (x1 * x2).mean("times")

    CC += [cc_surr]


df = pd.concat(
    [
        z_beta.to_dataframe("power_beta").reset_index(),
        z_gamma.to_dataframe("power_gamma").reset_index(),
        phi.to_dataframe("phi").reset_index(),
    ],
    axis=1,
)

df = df.loc[:, ~df.columns.duplicated()].copy()

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
thr = xr.concat(CC, dim="surr").quantile(0.95, "surr")
thr = thr.to_dataframe("thr").reset_index()

mean = thr.groupby(["roi"]).mean("thr")
std = thr.groupby(["roi"])["thr"].std()

df = cc.to_dataframe(name="cc").reset_index()
df = df.sort_values("cc", ascending=False)

sns.boxplot(x=df["roi"], y=df["cc"], color="lightgray",
            medianprops=dict(color="k", label='median'))

rois = [t.get_text() for t in ax6.get_xticklabels()]

mean = mean.reindex(index=rois).thr.values
std = std.reindex(index=rois).reset_index().thr.values

plt.fill_between(
    range(cc.sizes["roi"]),
    mean - std,
    mean + std,
    alpha=0.5,
    color="red",
)
plt.xticks(rotation=30, fontsize=8)
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

plt.savefig(f"figures/betagamma/beta_gamma_{sid}")
plt.close()
