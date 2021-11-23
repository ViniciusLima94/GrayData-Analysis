import os
import numpy as np
from GDa.session import session
from frites.dataset import SubjectEphy, DatasetEphy
from frites.workflow import WfMi
from GDa.util import create_stages_time_grid
from config import (sm_times, sm_kernel, sm_freqs, delta,
                    mode, freqs, n_cycles)
from xfrites.conn.conn_tf import wavelet_spec
import matplotlib.pyplot as plt
import argparse

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX",   help="index of the session to run",
                    type=int)

args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')
# Get session number
sessions = np.loadtxt("GrayLab/lucy/sessions.txt", dtype=str)
s_id = sessions[idx]

###############################################################################
# Loading session
###############################################################################

# Instantiate class
ses = session(raw_path='GrayLab/', monkey='lucy', date=s_id, session=1,
              slvr_msmod=True, align_to='cue', evt_dt=[-0.65, 3.00])

# Read data from .mat files
ses.read_from_mat()

# Filtering by trials
data = ses.filter_trials(trial_type=[1], behavioral_response=[1])

###############################################################################
# Compute power spectra
###############################################################################
sxx = wavelet_spec(data, freqs=freqs, roi=data.roi, times="time",
                   sfreq=data.attrs["fsample"], foi=None, sm_times=sm_times,
                   sm_freqs=sm_freqs, sm_kernel=sm_kernel, mode=mode,
                   n_cycles=n_cycles, mt_bandwidth=None,
                   decim=delta, kw_cwt={}, kw_mt={}, block_size=1,
                   n_jobs=20, verbose=None)

# Get stage mask for this session
s_mask = create_stages_time_grid(sxx.attrs["t_cue_on"], sxx.attrs["t_cue_off"],
                                 sxx.attrs["t_match_on"], sxx.attrs["fsample"],
                                 sxx.times.values, sxx.sizes["trials"],
                                 flatten=False)

###############################################################################
# MI Workflow
###############################################################################

# Convert to DatasetEphy
dt = DatasetEphy([sxx], y=[sxx.attrs["stim"].astype(int)], times="times",
                 roi=None, agg_ch=False)

mi_type = 'cd'
inference = 'ffx'
kernel = np.hanning(1)
wf = WfMi(mi_type, inference, verbose=True, kernel=kernel)

kw = dict(n_jobs=20, n_perm=200)
"""
The `cluster_th` input parameter specifies how the threshold is defined.
Use either :
* a float for a manual threshold
* None and it will be infered using the distribution of permutations
* 'tfce' for a TFCE threshold
"""
cluster_th = None  # {float, None, 'tfce'}

mi, pvalues = wf.fit(dt, mcp="cluster", cluster_th=cluster_th, **kw)

# Setting roi names for mi and pvalues
mi = mi.assign_coords({"roi": data.roi.values})
pvalues = pvalues.assign_coords({"roi": data.roi.values})

###############################################################################
# Saving data
###############################################################################

path_st = os.path.join(_ROOT, f"Results/lucy/{s_id}/session01")
path_mi = os.path.join(path_st, "mi_values.nc")
path_pval = os.path.join(path_st, "p_values.nc")

# Not storing for storage issues
# mi.to_netcdf(path_mi)
# pvalues.to_netcdf(path_pval)

###############################################################################
# Plotting values with p<=0.05
###############################################################################
out = mi * (pvalues <= 0.05)

plt.figure(figsize=(20, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    idx = out.isel(freqs=i).sum("times") > 0
    out.isel(freqs=i, roi=idx).plot(x="times", hue="roi")
plt.tight_layout()

plt.savefig(f"figures/mi_power/mi_power_{s_id}.png", dpi=150)
