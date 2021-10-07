import sys
import os
import time
import numpy                           as     np
import xarray                          as     xr
import h5py
from   config                          import *
from   GDa.session                     import session
from   GDa.io                          import set_paths

from mne.filter                        import filter_data
from frites.estimator                  import GCMIEstimator, CorrEstimator, DcorrEstimator
from frites.conn                       import conn_dfc, define_windows
from tqdm                              import tqdm

idx     = int(sys.argv[-1])

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
    # Add name of the file
    path_st = os.path.join(path_st, f'MI_k{sm_times}.nc')

    #  Instantiating session
    ses   = session(raw_path = dirs['rawdata'], monkey = dirs['monkey'][nmonkey], date = dirs['date'][nmonkey][idx],
                    session = nses, slvr_msmod = False, align_to = align_to, evt_dt = [-0.65, 3.00])
    # Load data
    ses.read_from_mat()

    # Filter data
    #  Xf = []
    #  for fb in foi:
    #      Xf += [filter_data(ses.data, ses.data.attrs["fsample"], fb[0], fb[1], method='iir', n_jobs=20)]
    #  Xf = np.stack(Xf,axis=0)

    #  Xf = xr.DataArray(Xf, dims=("freqs", 'trials', 'roi', 'times'),
    #               coords=(foi.mean(axis=1), ses.data.trials, ses.data.roi, ses.data.time))

    start = time.time()

    slwin_len  = 0.50    # windows of length 50ms
    slwin_step = 0.01    # 20ms step between each window (or 480ms overlap)
    # define the sliding windows
    sl_win, twin = define_windows(ses.data.time.values, slwin_len=slwin_len, slwin_step=slwin_step)

    est = GCMIEstimator('cc', copnorm=None, biascorrect=True, demeaned=False)

    n_pairs = int( ses.data.sizes["roi"]*(ses.data.sizes["roi"]-1)/2 )
    #  MI = np.zeros((len(foi),Xf.sizes["trials"],n_pairs, len(twin)))
    MI = np.zeros((ses.data.sizes["trials"],n_pairs, len(twin)))
#  for i in tqdm( range( foi.shape[0] ) ):
    MI = conn_dfc(ses.data, times='time', roi='roi', win_sample=sl_win, estimator=est, n_jobs=40)
    # reordering dimensions
    MI = MI.transpose("roi","freqs","trials","times")
    #  deleting attributes assigned by the method
    #  MI.attrs = {}
    #  copying data attributes
    #  for key in ses.data.attrs.keys():
    #      coh.attrs[key] = ses.data.attrs[key]
    #  coh.attrs['sources'] = x_s
    #  coh.attrs['targets'] = x_t
    #  coh.attrs['decim']   = delta
    #  coh.attrs['areas']   = ses.data.roi.values.astype('str')

    if os.path.isfile(path_st):
        os.system(f'rm {path_st}')

    # Saving the data
    coh.to_netcdf(path_st)

    end = time.time()
    print(f'Elapsed time to compute coherences: {str((end - start)/60.0)} min.' )
