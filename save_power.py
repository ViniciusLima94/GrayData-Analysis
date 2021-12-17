import os
import numpy as np
from GDa.session import session
from config import (decim, mode, freqs, n_cycles,
                    sessions, return_evt_dt)
from xfrites.conn.conn_tf import wavelet_spec
import argparse

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to run",
                    type=int)
parser.add_argument("TT", help="type of the trial",
                    type=int)
parser.add_argument("BR", help="behavioral response",
                    type=int)
parser.add_argument("ALIGN", help="wheter to align data to cue or match",
                    type=str)

args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX
tt = args.TT
br = args.BR
at = args.ALIGN

# Root directory
_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')
# Get session number
s_id = sessions[idx]

###############################################################################
# Loading session
###############################################################################

# Window in which the data will be read
evt_dt = return_evt_dt(at)

# Instantiate class
ses = session(raw_path='GrayLab/', monkey='lucy', date=s_id, session=1,
              slvr_msmod=True, align_to=at, evt_dt=evt_dt)

# Read data from .mat files
ses.read_from_mat()

# Filtering by trials
data = ses.filter_trials(trial_type=[tt],
                         behavioral_response=[br])

###############################################################################
# Compute power spectra
###############################################################################

sxx = wavelet_spec(data, freqs=freqs, roi=data.roi, times="time",
                   sfreq=data.attrs["fsample"], foi=None, sm_times=0,
                   sm_freqs=0, sm_kernel="square", mode=mode,
                   n_cycles=n_cycles, mt_bandwidth=None,
                   decim=decim, kw_cwt={}, kw_mt={}, block_size=1,
                   n_jobs=20, verbose=None)


###############################################################################
# Saves file
###############################################################################

results_path = f"Results/lucy/{s_id}/session01"

# Create results path in case it does not exist
if not os.path.exists(results_path):
    os.makedirs(results_path)

file_name = f"power_tt_{tt}_br_{br}_at_{at}.nc"
path_pow = os.path.join(_ROOT, results_path,
                        file_name)

sxx.attrs["evt_dt"] = evt_dt
del sxx.attrs["mt_bandwidth"]
sxx.to_netcdf(path_pow)
