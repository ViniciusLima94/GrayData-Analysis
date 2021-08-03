import sys
import os
import time
import numpy                           as     np
import xarray                          as     xr
import h5py
from   config                          import *
from   GDa.session                     import session
from   GDa.io                          import set_paths
from   GDa.fc.dFC                      import conn_correlation
from   joblib                          import Parallel, delayed

idx     = int(sys.argv[-1])

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
    path_st = os.path.join(path_st, f'corr.nc')

    #  Instantiating session
    ses   = session(raw_path = dirs['rawdata'], monkey = dirs['monkey'][nmonkey], date = dirs['date'][nmonkey][idx],
                    session = nses, slvr_msmod = False, align_to = align_to, evt_dt = [-0.65, 3.00])
    # Load data
    ses.read_from_mat()

    start = time.time()

    # Get the channel pairs
    x_s, x_t  = np.triu_indices(ses.data.sizes['roi'], k=1)
    pairs     = np.array([x_s,x_t]).T

    # (data, times=None, roi=None, sfreq=None, f_low=None, f_high=None, pairs=None, win_sample=None, decim=None, block_size=None, verbose=False, n_jobs=1)
    kw = dict(
        times=ses.data.time, roi=ses.data.roi, n_jobs=20, pairs=pairs,
        sfreq=ses.data.attrs['fsample'], decim=15,
    )

    # compute the correlation for each frequency band
    corr = []
    for f_low, f_high in foi:
        corr += [conn_correlation(ses.data, f_low=f_low, f_high=f_high, **kw)]
    corr = xr.concat(corr, dim="freqs")
    # reordering dimensions
    corr = corr.transpose("roi","freqs","trials","times")
    # replace trial axis for the actual values
    corr = corr.assign_coords({"trials":ses.data.trials.values}) 
    # deleting attributes assigned by the method
    corr.attrs = {}
    # copying data attributes
    for key in ses.data.attrs.keys():
        corr.attrs[key]   = ses.data.attrs[key]
    corr.attrs['sources'] = x_s
    corr.attrs['targets'] = x_t

    if os.path.isfile(path_st):
        os.system(f'rm {path_st}')

    # Saving the data
    corr.to_netcdf(path_st)

    end = time.time()
    print(f'Elapsed time to compute coherences: {str((end - start)/60.0)} min.' )
