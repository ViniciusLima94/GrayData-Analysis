import os
import numpy as np
from GDa.session import session
from config import (sm_times, sm_kernel, sm_freqs, delta,
                    mode, freqs, n_cycles)
from xfrites.conn.conn_tf import wavelet_spec
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

# Convert to format that should be used in xfrites
# sxx = [sxx.isel(roi=[r]) for r in range(len(sxx['roi']))]

path_pow = os.path.join(_ROOT, f"Results/lucy/{s_id}/session01/power.nc")

del sxx.attrs["mt_bandwidth"]
sxx.to_netcdf(path_pow)
