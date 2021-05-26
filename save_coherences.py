import sys
import os
import time
import numpy                           as     np
import h5py
from   config                          import *
from   GDa.session                     import session
from   GDa.io                          import set_paths
from   xfrites.conn.conn_coh           import conn_coherence_wav
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
    #path_st = os.path.join('Results', str(dirs['monkey'][nmonkey])+'_session01_'+str(dirs['date'][nmonkey][idx])+'.h5')
    path_st = os.path.join('Results', str(dirs['monkey'][nmonkey]), str(dirs['date'][nmonkey][idx]), f'session0{nses}')
    # Check if path existis, if not it will be created
    if not os.path.exists(path_st):
        os.mkdir(path_st)
    # Add name of the file
    path_st = os.path.join(path_st, 'super_tensor.h5')

    #  Instantiating session
    ses   = session(raw_path = dirs['rawdata'], monkey = dirs['monkey'][nmonkey], date = dirs['date'][nmonkey][idx],
                    session = nses, slvr_msmod = False, align_to = align_to, evt_dt = [-0.65, 3.00])
    # Load data
    ses.read_from_mat()

    start = time.time()

    kw = dict(
        freqs=freqs, times=ses.data.time, roi=ses.data.roi, foi=foi, n_jobs=-1,
        sfreq=ses.data.attrs['fsample'], mode=mode, decim_at=decim_at, n_cycles=n_cycles, decim=delta,
        sm_times=sm_times, sm_freqs=sm_freqs, block_size=2
    )

    # compute the coherence
    coh = conn_coherence_wav(ses.data.values.astype(np.float32), **kw)

    try:
        hf = h5py.File(path_st, 'r+')
    except:
        hf = h5py.File(path_st, 'w')

    #hf = h5py.File(path_st, 'w')
    hf.create_dataset('coherence', data=coh.transpose("roi", "trials", "freqs", "times"))
    hf.create_dataset('freqs',     data=freqs)
    hf.create_dataset('tarray',    data=coh.times.values)
    hf.create_dataset('bands',     data=foi)
    [hf.create_dataset('info/'+k,  data=ses.data.attrs[k]) for k in ses.data.attrs.keys()]
    hf.close()

    end = time.time()
    print(f'Elapsed time to compute coherences: {str((end - start)/60.0)} min.' )
