import os
import time
import numpy as np
import argparse

from config import (sm_times, sm_kernel, sm_freqs, decim,
                    mode, freqs, n_cycles, sessions,
                    return_evt_dt)
from GDa.session import session
from frites.conn.conn_spec import conn_spec
from GDa.signal.surrogates import trial_swap_surrogates

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("METRIC", help="which connectivity metric to use",
                    type=str)
parser.add_argument("ALIGNED", help="wheter to align data to cue or match",
                    type=str)
parser.add_argument("SIDX", help="index of the session to run",
                    type=int)
parser.add_argument("SURR", help="wheter to compute for surrogate data or not",
                    type=int)
parser.add_argument("SEED", help="seed for create surrogates",
                    type=int)

args = parser.parse_args()
# The connectivity metric that should be used
metric = args.METRIC
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


def _bias_lachaux(sm_times, freqs, n_cycles):
    return (1+2*sm_times*freqs/n_cycles)**-1


if __name__ == '__main__':

    # Path in which to save coherence data
    path_st = os.path.join('/home/vinicius/funcog/gda/Results',
                           'lucy', sessions[idx], 'session01')
    # Check if path existis, if not it will be created
    if not os.path.exists(path_st):
        os.makedirs(path_st)

    # Add name of the coherence file
    path_st_coh = os.path.join(path_st,
                               f'{metric}_at_{at}.nc')

    # Remove file if it was already created
    if os.path.isfile(path_st_coh):
        os.system(f'rm {path_st_coh}')

    #  Instantiating session
    raw_path = os.path.expanduser("~/funcog/gda/GrayLab")
    ses = session(raw_path=raw_path,
                  monkey="lucy",
                  date=sessions[idx],
                  session=1, slvr_msmod=True,
                  align_to=at, evt_dt=evt_dt)
    # Load data
    ses.read_from_mat()

    start = time.time()

    kw = dict(
        freqs=freqs, times="time", roi="roi", foi=None,
        n_jobs=10, pairs=None, sfreq=ses.data.attrs['fsample'],
        mode=mode, n_cycles=n_cycles, decim=decim, metric=metric,
        sm_times=sm_times, sm_freqs=sm_freqs, sm_kernel=sm_kernel, block_size=2
    )

    # compute the coherence
    coh = conn_spec(ses.data, **kw).astype(np.float32, keep_attrs=True)
    # reordering dimensions
    coh = coh.transpose("roi", "freqs", "trials", "times")
    # Add areas as attribute
    coh.attrs['areas'] = ses.data.roi.values.astype('str')
    coh.attrs['bias'] = _bias_lachaux(sm_times, freqs, n_cycles).tolist()
    coh.attrs['k'] = sm_times
    coh.attrs['mode'] = mode
    # Save to file
    coh.to_netcdf(path_st_coh)
    # Store attributes
    attrs = coh.attrs
    # To release memory
    del coh

    # Create data surrogate
    if surr:
        data_surr = trial_swap_surrogates(
            ses.data.astype(np.float32), seed=seed, verbose=False)
        coh_surr = conn_spec(data_surr, **kw)
        coh_surr.attrs = attrs
        # Add name of the coherence file
        path_st_surr = os.path.join(
            path_st, f'{metric}_at_{at}_surr.nc')
        coh_surr.to_netcdf(path_st_surr)
        #  Estimate significance level from 95% percentile over trials
        # coh_surr = coh_surr.quantile(0.95, dim="trials")
        # path_st_thr = os.path.join(
            # path_st, f'thr_{metric}_at_{at}.nc')
        # coh_surr.to_netcdf(path_st_surr)
        del coh_surr

    end = time.time()
    print(
        f'Elapsed time to compute coherences: {str((end - start)/60.0)} min.')
