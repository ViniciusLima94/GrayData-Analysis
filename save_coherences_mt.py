import sys
import os
import time
import numpy                           as     np
import h5py
from   config                          import *
from   GDa.session                     import session
from   GDa.spectral_analysis           import time_frequency    as tf
from   GDa.io                          import set_paths
from   joblib                          import Parallel, delayed

idx = 3 #int(sys.argv[-1])

nmonkey = 0
nses    = 1
ntype   = 0

#################################################################################################
# Which trial type, alignment and behav. response to use
#################################################################################################
trial_type = 3
align_to  = 'cue'
behavioral_response = None 
#################################################################################################

#  Set the paths
paths = set_paths(raw_path = dirs['rawdata'], monkey = dirs['monkey'][nmonkey], 
                         date = dirs['date'][nmonkey][idx], session = nses)
#  Instantiating session
ses   = session(raw_path = dirs['rawdata'], monkey = dirs['monkey'][nmonkey], date = dirs['date'][nmonkey][idx], 
                session = nses, slvr_msmod = False, align_to = align_to, evt_dt = [-0.65, 3.00])
ses.read_from_mat()
#  Downsampled time array
tarray = ses.time[::delta]

if  __name__ == '__main__':

    start = time.time()

    tf.wavelet_coherence(data = ses.data, pairs = ses.data.attrs['pairs'], fs = ses.data.attrs['fsample'], 
                         freqs = freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, delta = delta, 
                         method = method, win_time = win_time, win_freq = win_freq, dir_out = paths.dir_out, n_jobs = -1)

    # Load all the files generated and save in a single file
    super_tensor = np.zeros([ses.data.attrs['nP'], len(ses.data['trials']), freqs.shape[0], tarray.shape[0]])

    for j in range(ses.data.attrs['nP']):
        path = os.path.join(paths.dir_out, 
                            'ch1_'+str(ses.data.attrs['pairs'][j,0])+'_ch2_'+str(ses.data.attrs['pairs'][j,1])+'.h5' )
        with h5py.File(path, 'r') as hf:
                super_tensor[j,:,:,:] = hf['coherence'][:]

    path_st = os.path.join('super_tensors', dirs['monkey'][nmonkey] + '_session01_' + dirs['date'][nmonkey][idx]+ '.h5')

    try:
        hf = h5py.File(path_st, 'r+')
    except:
        hf = h5py.File(path_st, 'w')

    hf.create_dataset('coherence', data=super_tensor)
    hf.create_dataset('freqs',     data=freqs)
    hf.create_dataset('tarray',    data=tarray)
    hf.create_dataset('bands',     data=bands[dirs['monkey'][nmonkey]])
    [hf.create_dataset('info/'+k, data=ses.data.attrs[k]) for k in ses.data.attrs.keys()]
    hf.close()

    end = time.time()
    print('Elapsed time to compute coherences: ' +str((end - start)/60.0) + ' min.' )
