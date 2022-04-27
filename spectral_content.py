import os

import argparse
import numpy as np
import xarray as xr
from frites.conn.conn_tf import _tf_decomp

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

channels = np.array([36, 63, 66, 69, 73, 83, 84, 92, 95, 105, 106, 117, 236])
idx = [(ch in channels) for ch in data.channels_labels]

w_sel = w.isel(roi=idx)

##############################################################################
# Compute average spectra per time-stage 
##############################################################################
rois = [
    f"{area} ({channel})" for area, channel in zip(w.roi.values, data.channels_labels)
]

w = w.assign_coords({"roi": rois})

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
# Compute peaks in each band 
##############################################################################
bands = np.array([[0, 6],
                  [6, 14],
                  [14, 26],
                  [26, 43],
                  [43, 80]])

peaks = np.zeros((w_static.sizes['times'], w_static.sizes['trials'],
                 w_static.sizes['roi'], bands.shape[0]))
peaks = xr.DataArray(peaks, dims=w_static.dims,
                     coords = {'times': w_static.times.data,
                               'trials': w_static.trials.data,
                               'roi': w_static.roi.data})

for i, (flow, fhigh) in enumerate(bands):
    peaks[..., i] = w_static.sel(freqs=slice(flow, fhigh)).max(['freqs'])

# Path to results folder
_RESULTS = os.path.join(_ROOT,
                        "Results/lucy/peaks",
                        f"{sidx}.nc")

peaks.to_netcdf(_RESULTS)

# S_tilda = w_static / w_static.integrate(coord="freqs")  # Normalize spectra
# S_tilda = S_tilda.assign_coords({"roi": rois})
# S_tilda = w_static.assign_coords({"roi": rois})

# SC = S_tilda.sel(freqs=band).integrate(coord="freqs")  # Spectral content in beta band

# SC = S_tilda.sel(freqs=band).max('freqs')

# df = SC.to_dataframe(name="SC").reset_index()
# df = df.sort_values("SC", ascending=False)

# _, rois = _extract_roi(df.roi.values, ' ')

# unique_rois = np.unique(rois)
# custom_palette = sns.color_palette("pastel", len(unique_rois))

# colors = [(0, 0, 0)] * len(rois)
# for i_r_u, r_u in enumerate(unique_rois):
    # for i_r, r in enumerate(rois):
        # if r == r_u:
            # colors[i_r] = custom_palette[i_r_u]
# colors = dict(zip(df.roi.values, colors))

# plt.figure(figsize=(15, 6))
# ax = plt.subplot(111)
# sns.boxplot(x=df["roi"], y=df["SC"], palette=colors, showfliers = False)
# plt.xticks(rotation=90)
# plt.xlabel("")
# ax.spines["right"].set_visible(False)
# ax.spines["top"].set_visible(False)
# # plt.hlines(0.1, -1, SC.sizes["roi"], color="r", ls="--", lw=0.6)
# plt.xlim(-1, SC.sizes["roi"])
# plt.title(f"Ranked spectral content (26-43 Hz) - ({sidx})")

# plt.savefig(f'figures/spectral_content/sc_ranked_{sidx}.png', bbox_inches='tight')
# plt.close()

##############################################################################
# Overlap of beta and gamma bands
##############################################################################
# beta = slice(26, 43)
# gamma = slice(43, 80)

# rois = [
    # f"{area} ({channel})" for area, channel in zip(w.roi.values, data.channels_labels)
# ]

# w_beta = w.sel(freqs=beta).mean('freqs')
# w_gamma = w.sel(freqs=gamma).mean('freqs')


# z_beta = (w_beta - w_beta.mean('times')) / w_beta.std('times')
# z_gamma = (w_gamma - w_gamma.mean('times')) / w_gamma.std('times')

# cc = (z_beta * z_gamma).mean('times')
# cc = cc.assign_coords({"roi": rois})

# df = cc.to_dataframe(name='cc').reset_index()
# df = df.sort_values("cc", ascending=False)

# plt.figure(figsize=(15, 6))
# ax = plt.subplot(111)
# sns.boxplot(x=df["roi"], y=df["cc"], palette=colors)
# plt.xticks(rotation=90)
# plt.xlabel("")
# ax.spines["right"].set_visible(False)
# ax.spines["top"].set_visible(False)
# plt.hlines(0.1, -1, SC.sizes["roi"], color="r", ls="--", lw=0.6)
# plt.xlim(-1, SC.sizes["roi"])
# plt.title(f"Power-correlation between beta and gamma band ({sidx})")
# plt.savefig(f'figures/spectral_content/correlation_beta_gamma_{sidx}.png', bbox_inches='tight')
# plt.close()
