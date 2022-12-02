import os
import argparse
import numpy as np
import xarray as xr

from tqdm import tqdm
from GDa.session import session
from config import (decim, mode, freqs, n_cycles,
                    get_dates, return_evt_dt)
from frites.conn.conn_tf import _tf_decomp

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("TT", help="type of the trial",
                    type=int)
parser.add_argument("BR", help="behavioral response",
                    type=int)
parser.add_argument("ALIGN", help="wheter to align data to cue or match",
                    type=str)
parser.add_argument("MONKEY", help="which monkey to use",
                    type=str)

args = parser.parse_args()

# Index of the session to be load
tt = args.TT
br = args.BR
at = args.ALIGN
monkey = args.MONKEY

if tt == 2 or tt == 3:
    br = None

sessions = get_dates(monkey)

# Root directory
_ROOT = os.path.expanduser('~/funcog/gda')

for s_id in tqdm(sessions):
    ###########################################################################
    # Loading session
    ###########################################################################

    # Window in which the data will be read
    evt_dt = return_evt_dt(at, monkey=monkey)
    # Path to LFP data
    raw_path = os.path.expanduser("~/funcog/gda/GrayLab/")
    # Instantiate class
    ses = session(raw_path=raw_path, monkey=monkey, date=s_id, session=1,
                  slvr_msmod=True, align_to=at, evt_dt=evt_dt)

    # Read data from .mat files
    ses.read_from_mat()

    # Filtering by trials
    data = ses.filter_trials(trial_type=[tt],
                             behavioral_response=[br])

    ###########################################################################
    # Compute power spectra
    ###########################################################################
    sxx = _tf_decomp(
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

    sxx = xr.DataArray(
        (sxx * np.conj(sxx)).real,
        name="power",
        dims=("trials", "roi", "freqs", "times"),
        coords=(data.trials.values, data.roi.values,
                freqs, data.time.values[::decim]),
    )

    ###########################################################################
    # Saves file
    ###########################################################################

    # Path in which to save coherence data
    results_path = os.path.join(_ROOT, 'Results',
                           monkey, s_id, 'session01')
    # Create results path in case it does not exist
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    file_name = f"power_tt_{tt}_br_{br}_at_{at}.nc"
    path_pow = os.path.join(results_path,
                            file_name)
    # print(path_pow)

    sxx.attrs = data.attrs
    sxx.attrs["evt_dt"] = evt_dt
    sxx.to_netcdf(path_pow)
