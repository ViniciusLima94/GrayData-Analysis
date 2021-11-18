import sys
import os
import time
import numpy                           as     np
import h5py
from   config                          import *
from   GDa.session                     import session
from   GDa.io                          import set_paths
from   xfrites.conn.conn_spec          import conn_spec
from   GDa.signal.surrogates           import trial_swap_surrogates
from   joblib                          import Parallel, delayed
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("METRIC", help="which connectivity metric to use", 
                    type=str)
parser.add_argument("SIDX",   help="index of the session to run", 
                    type=int)
parser.add_argument("SURR",   help="wheter to compute for surrogate data or not",
                    type=int)
parser.add_argument("SEED",   help="seed for create surrogates",
                    type=int)

args   = parser.parse_args()
# The connectivity metric that should be used
metric = args.METRIC
# The index of the session to use
idx    = args.SIDX
# Wheter to use surrogate or not
surr   = bool( args.SURR )
# Wheter to use surrogate or not
seed   = args.SEED

nmonkey = 0
nses    = 1
ntype   = 0

#################################################################################################
# Which trial type, alignment and behav. response to use
#################################################################################################
trial_type          = 3
align_to            = 'cue'
behavioral_response = None 

#################################################################################################
# Method to compute the bias accordingly to Lachaux et. al. (2002)
#################################################################################################
def _bias_lachaux(sm_times, freqs, n_cycles):
    return (1+2*sm_times*freqs/n_cycles)**-1

if  __name__ == '__main__':

    # Path in which to save coherence data
    path_st = os.path.join('Results', str(dirs['monkey'][nmonkey]), str(dirs['date'][nmonkey][idx]), f'session0{nses}')
    # Check if path existis, if not it will be created
    if not os.path.exists(path_st):
        os.makedirs(path_st)

    # Add name of the coherence file
    path_st_coh = os.path.join(path_st, f'{metric}_k_{sm_times}_{mode}.nc')

    # Remove file if it was already created
    if os.path.isfile(path_st_coh):
        os.system(f'rm {path_st_coh}')

    #  Instantiating session
    ses   = session(raw_path = dirs['rawdata'], monkey = dirs['monkey'][nmonkey], date = dirs['date'][nmonkey][idx],
                    session = nses, slvr_msmod = True, align_to = align_to, evt_dt = [-0.65, 3.00])
    # Load data
    ses.read_from_mat()

    start = time.time()

    kw = dict(
        freqs=freqs, times="time", roi=ses.data.roi, foi=None, n_jobs=10, pairs=None,
        sfreq=ses.data.attrs['fsample'], mode=mode, n_cycles=n_cycles, decim=delta, metric=metric,
        sm_times=sm_times, sm_freqs=sm_freqs, sm_kernel=sm_kernel, block_size=1
    )

    # compute the coherence
    coh = conn_spec(ses.data, **kw).astype(np.float32, keep_attrs=True)
    # reordering dimensions
    coh = coh.transpose("roi","freqs","trials","times")
    # Add areas as attribute
    coh.attrs['areas'] = ses.data.roi.values.astype('str')
    coh.attrs['bias']  = _bias_lachaux(sm_times, freqs, n_cycles).tolist()
    # Save to file
    coh.to_netcdf(path_st_coh)
    # To release memory
    del coh
    
    # Create data surrogate
    if surr:
        data_surr = trial_swap_surrogates(ses.data.astype(np.float32), seed=seed, verbose=False)
        coh_surr  = conn_spec(data_surr, **kw)
        #  Estimate significance level from 95% percentile over trials
        coh_surr  = coh_surr.quantile(0.95, dim="trials")
        # Apply threshold
        #  coh       = (coh>=coh_surr)*coh

        # Add name of the coherence file
        path_st_surr = os.path.join(path_st, f'{metric}_k_{sm_times}_{mode}_surr.nc')
        coh_surr.to_netcdf(path_st_surr)
        del coh_surr

    end = time.time()
    print(f'Elapsed time to compute coherences: {str((end - start)/60.0)} min.' )
