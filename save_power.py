import os
import argparse

import xarray as xr

from tqdm import tqdm
from GDa.session import session
from GDa.util import _extract_roi
from config import (mode, bandwidth, sessions, t_win,
                    fmin, fmax, return_evt_dt, n_fft)
from xfrites.conn.conn_csd import conn_csd
###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
# parser.add_argument("SIDX", help="index of the session to run",
                    # type=int)
parser.add_argument("TT", help="type of the trial",
                    type=int)
parser.add_argument("BR", help="behavioral response",
                    type=int)
parser.add_argument("ALIGN", help="wheter to align data to cue or match",
                    type=str)

args = parser.parse_args()

# Index of the session to be load
# idx = args.SIDX
tt = args.TT
br = args.BR
at = args.ALIGN

# Root directory
_ROOT = os.path.expanduser('~/funcog/gda')
# Get session number
# s_id = sessions[idx]

for s_id in tqdm(sessions):
    ###########################################################################
    # Loading session
    ###########################################################################

    # Window in which the data will be read
    evt_dt = return_evt_dt(at)

    # Instantiate class
    ses = session(raw_path=os.path.join(_ROOT, 'GrayLab/'),
                  monkey='lucy', date=s_id, session=1,
                  slvr_msmod=True, align_to=at, evt_dt=evt_dt)

    # Read data from .mat files
    ses.read_from_mat()

    # Filtering by trials
    data = ses.filter_trials(trial_type=[tt],
                             behavioral_response=[br])

    ###########################################################################
    # Compute power spectra
    ###########################################################################

    kw = dict(fmin=fmin, fmax=fmax,bandwidth=bandwidth, n_fft=n_fft)
    csd = []
    for t0, t1 in t_win:
        csd += [conn_csd(ses.data.sel(time=slice(t0,t1)), times='time', roi='roi',
                         sfreq=ses.data.attrs["fsample"], mode=mode, metric="csd",
                         freqs=None, n_jobs=20, verbose=None,  csd_kwargs=kw)]
    csd = xr.concat(csd, "times")
    x_s, x_t = csd.attrs['x_s'], csd.attrs['x_t']
    psd = csd.isel(roi=(x_s==x_t)).real

    ###########################################################################
    # Saves file
    ###########################################################################

    results_path = f"Results/lucy/{s_id}/session01"

    # Create results path in case it does not exist
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    file_name = f"power_csd_tt_{tt}_br_{br}_at_{at}.nc"
    path_pow = os.path.join(_ROOT, results_path,
                            file_name)

    psd.attrs["evt_dt"] = evt_dt
    del psd.attrs['attrs'], psd.attrs['freqs'], psd.attrs['sm_times'], psd.attrs['sm_freqs']
    del psd.attrs['roi_idx'], psd.attrs['win_sample'], psd.attrs['win_times'], psd.attrs['blocks']
    psd.to_netcdf(path_pow)
