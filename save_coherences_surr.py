import sys
import os
import time
import numpy                           as     np
import h5py
from   config                          import *
from   GDa.session                     import session
from   GDa.io                          import set_paths
from   GDa.signal.surrogates           import phase_rand_surrogates, trial_swap_surrogates
from   xfrites.conn.conn_coh           import conn_coherence_wav
from   joblib                          import Parallel, delayed

idx     = int(sys.argv[-1])
seed    = int(sys.argv[-2]) * 1000

nmonkey = 0
nses    = 1
ntype   = 0


#################################################################################################
# Which trial type, alignment and behav. response to use
#################################################################################################
trial_type = 3
align_to  = 'cue'
behavioral_response = None 

if  __name__ == '__main__':

    # Path in which to save coherence data
    path_st = os.path.join('Results', str(dirs['monkey'][nmonkey]), str(dirs['date'][nmonkey][idx]), f'session0{nses}')
    # Check if path existis, if not it will be created
    if not os.path.exists(path_st):
        os.makedirs(path_st)
    # Add name of the file
    path_st = os.path.join(path_st, f'super_tensor_{seed}.nc')

    #  Instantiating session
    ses   = session(raw_path = dirs['rawdata'], monkey = dirs['monkey'][nmonkey], date = dirs['date'][nmonkey][idx],
                    session = nses, slvr_msmod = False, align_to = align_to, evt_dt = [-0.65, 3.00])
    # Load data
    ses.read_from_mat()

    start = time.time()

    # Get the channel pairs
    x_s, x_t  = np.triu_indices(ses.data.sizes['roi'], k=1)
    pairs     = np.array([x_s,x_t]).T

    kw = dict(
        freqs=freqs, times=ses.data.time, roi=ses.data.roi, foi=foi, n_jobs=20, pairs=pairs,
        sfreq=ses.data.attrs['fsample'], mode=mode, decim_at=decim_at, n_cycles=n_cycles, decim=delta,
        sm_times=sm_times, sm_freqs=sm_freqs, sm_kernel=sm_kernel, block_size=1
    )

    # Create data surrogate
    #  ses.data.values = phase_rand_surrogates(ses.data, val=0, seed=seed,verbose=False,n_jobs=-1)
    ses.data.values = trial_swap_surrogates(ses.data, seed=seed, verbose=False)

    # compute the coherence
    coh = conn_coherence_wav(ses.data.values, **kw).astype(np.float32)
    # reordering dimensions
    coh = coh.transpose("roi","freqs","trials","times")
    # replace trial axis for the actual values
    coh = coh.assign_coords({"trials":ses.data.trials.values}) 
    # deleting attributes assigned by the method
    coh.attrs = {}
    # copying data attributes
    for key in ses.data.attrs.keys():
        coh.attrs[key] = ses.data.attrs[key]
    coh.attrs['sources'] = x_s
    coh.attrs['targets'] = x_t
    #  coh.attrs['areas']   = ses.data.roi.values.astype('str')

    if os.path.isfile(path_st):
        os.system(f'rm {path_st}')

    # Saving the data
    coh.to_netcdf(path_st)

    end = time.time()
    print(f'Elapsed time to compute coherences: {str((end - start)/60.0)} min.' )