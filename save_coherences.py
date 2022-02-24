import os
import time
import numpy as np
import xarray as xr
import argparse

from config import (mode, bandwidth, sessions, t_win,
                    fmin, fmax, return_evt_dt, n_fft)
from GDa.session import session
from xfrites.conn.conn_csd import conn_csd
from GDa.signal.surrogates import trial_swap_surrogates

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("ALIGNED", help="wheter to align data to cue or match",
                    type=str)
parser.add_argument("SIDX", help="index of the session to run",
                    type=int)
parser.add_argument("SURR", help="wheter to compute for surrogate data or not",
                    type=int)
parser.add_argument("SEED", help="seed for create surrogates",
                    type=int)

args = parser.parse_args()
# Wheter to align data to cue or match
at = args.ALIGNED
# The index of the session to use
idx = args.SIDX
# Wheter to use surrogate or not
surr = bool(args.SURR)
# Wheter to use surrogate or not
seed = args.SEED

# Window in which the data will be read
evt_dt = return_evt_dt(at)

###############################################################################
# Method to compute the bias accordingly to Lachaux et. al. (2002)
###############################################################################

if __name__ == '__main__':

    # Path in which to save coherence data
    path_st = os.path.join('/home/vinicius/funcog/gda/Results',
                           'lucy', sessions[idx], 'session01')
    # Check if path existis, if not it will be created
    if not os.path.exists(path_st):
        os.makedirs(path_st)

    # Add name of the coherence file
    path_st_coh = os.path.join(path_st,
                               f'coh_csd_{mode}_at_{at}.nc')

    # Remove file if it was already created
    if os.path.isfile(path_st_coh):
        os.system(f'rm {path_st_coh}')

    #  Instantiating session
    ses = session(raw_path="GrayLab/",
                  monkey="lucy",
                  date=sessions[idx],
                  session=1, slvr_msmod=True,
                  align_to=at, evt_dt=evt_dt)
    # Load data
    ses.read_from_mat()

    start = time.time()

    # kw = dict(t0=evt_dt[0], fmin=fmin, fmax=fmax,bandwidth=bandwidth)
    kw = dict(fmin=fmin, fmax=fmax,bandwidth=bandwidth, n_fft=n_fft)
    coh = []
    for t0, t1 in t_win:
        coh += [conn_csd(ses.data.sel(time=slice(t0,t1)), times='time', roi='roi',
                         sfreq=ses.data.attrs["fsample"], mode=mode, metric="coh",
                         freqs=None, n_jobs=20, verbose=None,  csd_kwargs=kw)]
    coh = xr.concat(coh, "times")

    # reordering dimensions
    coh = coh.transpose("roi", "freqs", "trials", "times")
    # Add areas as attribute
    coh.attrs['areas'] = ses.data.roi.values.astype('str')
    del coh.attrs['attrs'], coh.attrs['freqs'], coh.attrs['sm_times'], coh.attrs['sm_freqs']
    del coh.attrs['roi_idx'], coh.attrs['win_sample'], coh.attrs['win_times'], coh.attrs['blocks']
    # Save to file
    coh.to_netcdf(path_st_coh)
    # To release memory
    del coh

    # Create data surrogate
    if surr:
        data_surr = trial_swap_surrogates(
            ses.data.astype(np.float32), seed=seed, verbose=False)
        coh_surr = []
        for t0, t1 in t_win:
            coh_surr += [conn_csd(data_surr.sel(time=slice(t0,t1)), times='time', roi='roi',
                             sfreq=ses.data.attrs["fsample"], mode=mode, metric="coh",
                             freqs=None, n_jobs=20, verbose=None,  csd_kwargs=kw)]
        coh_surr = xr.concat(coh_surr, "times")
        #  Estimate significance level from 95% percentile over trials
        coh_surr = coh_surr.quantile(0.95, dim="trials")
        # Add name of the coherence file
        path_st_surr = os.path.join(
            path_st, f'coh_csd_{mode}_at_{at}_surr.nc')
        coh_surr.to_netcdf(path_st_surr)
        del coh_surr

    end = time.time()
    print(
        f'Elapsed time to compute coherences: {str((end - start)/60.0)} min.')
