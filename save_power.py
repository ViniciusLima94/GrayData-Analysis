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
parser.add_argument("TT",   help="type of the trial",
                    type=int)
parser.add_argument("BR",   help="behavioral response",
                    type=int)

args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX
tt = args.TT
br = args.BR

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
data = ses.filter_trials(trial_type=[tt],
                         behavioral_response=[br])

###############################################################################
# Compute power spectra
###############################################################################

sxx = wavelet_spec(data, freqs=freqs, roi=data.roi, times="time",
                   sfreq=data.attrs["fsample"], foi=None, sm_times=sm_times,
                   sm_freqs=sm_freqs, sm_kernel=sm_kernel, mode=mode,
                   n_cycles=n_cycles, mt_bandwidth=None,
                   decim=delta, kw_cwt={}, kw_mt={}, block_size=1,
                   n_jobs=20, verbose=None)

###############################################################################
# Saves file
###############################################################################

results_path = f"Results/lucy/{s_id}/session01"

# Create results path in case it does not exist
if not os.path.exists(results_path):
    os.makedirs(results_path)

file_name = f"power_tt_{tt}_br_{br}.nc"
path_pow = os.path.join(_ROOT, results_path,
                        file_name)

del sxx.attrs["mt_bandwidth"]
sxx.to_netcdf(path_pow)
