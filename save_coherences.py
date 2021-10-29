import sys
import os
import time
import numpy                           as     np
import h5py
from   config                          import *
from   GDa.session                     import session
from   GDa.io                          import set_paths
from   xfrites.conn.conn_spec          import conn_spec
#  from   xfrites.conn.conn_coh          import conn_coherence_wav
from   GDa.signal.surrogates           import trial_swap_surrogates
from   joblib                          import Parallel, delayed
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("METRIC", help="which connectivity metric to use", 
                    type=str)
parser.add_argument("SIDX", help="index of the session to run", 
                    type=int)
parser.add_argument("SURR", help="wheter to compute for surrogate data or not",
                    type=int)
parser.add_argument("SEED", help="seed for create surrogates",
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
                    session = nses, slvr_msmod = False, align_to = align_to, evt_dt = [-0.65, 3.00])
    # Load data
    ses.read_from_mat()

    start = time.time()

    kw = dict(
        freqs=freqs, times="time", roi=ses.data.roi, foi=None, n_jobs=20, pairs=None,
        sfreq=ses.data.attrs['fsample'], mode=mode, n_cycles=n_cycles, decim=delta, metric=metric,
        sm_times=sm_times, sm_freqs=sm_freqs, sm_kernel=sm_kernel, block_size=2
    )

    # compute the coherence
    coh = conn_spec(ses.data.astype(np.float32), **kw).astype(np.float32)
    # reordering dimensions
    coh = coh.transpose("roi","freqs","trials","times")
    # replace trial axis for the actual values
    coh = coh.assign_coords({"trials":ses.data.trials.values}) 
    #  copying data attributes
    for key in ses.data.attrs.keys():
        coh.attrs[key] = ses.data.attrs[key]
    coh.attrs['decim']   = delta
    coh.attrs['areas']   = ses.data.roi.values.astype('str')

    # Saving the data
    coh.to_netcdf(path_st_coh)
    # To release memory
    del coh
    
    # Create data surrogate
    if surr:
        n_surr    = 10
        data_surr = []
        coh_surr  = []
        for i in range(n_surr):
            data_surr = trial_swap_surrogates(ses.data.astype(np.float32), seed=i+seed, verbose=False)
            coh_surr += [conn_spec(data_surr, **kw)]
        coh_surr = xr.concat(coh_surr, dim="seeds").astype(np.float32)

    # Compute thresholds based on the coherence surrogates
    thr = coh_surr.quantile(0.95, "seeds")
    del coh_surr

    # Add name of the surrogate threshold file
    path_st_thr = os.path.join(path_st, f'thr_{metric}_k_{sm_times}_{mode}.nc')

    # Saving the data
    thr.to_netcdf(path_st_thr)

    end = time.time()
    print(f'Elapsed time to compute coherences: {str((end - start)/60.0)} min.' )
