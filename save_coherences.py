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

idx = 1#int(sys.argv[-1])

nmonkey = 0
nses    = 1
ntype   = 0

#  Set the paths
paths        = set_paths(raw_path = dirs['rawdata'], monkey = dirs['monkey'][nmonkey], 
                         date = dirs['date'][nmonkey][idx], session = nses)
#  Instantiating session
ses = session(raw_path = dirs['rawdata'], monkey = dirs['monkey'][nmonkey], date = dirs['date'][nmonkey][idx], 
              session = nses, slvr_msmod = False, align_to = 'cue', trial_type = 1,
              behavioral_response = 1, evt_dt = [-0.65, 3.00], save_to_h5=False)

if  __name__ == '__main__':

    start = time.time()

    tf.wavelet_coherence(data = ses.data, pairs = ses.readinfo['pairs'], fs = ses.readinfo['fsample'], 
                         freqs = freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, delta = delta, 
                         method = method, win_time = win_time, win_freq = win_freq, dir_out = paths.dir_out, n_jobs = -1)
    

    # Load all the files generated and save in a single file
    super_tensor = np.zeros([nP, nT, freqs.shape[0], tarray.shape[0]])
    for j in range(nP):
        path = os.path.join(dir_out, 
                            'ch1_'+str(pairs[j,0])+'_ch2_'+str(pairs[j,1])+'.h5' )
        with h5py.File(path, 'r') as hf:
                super_tensor[j,:,:,:] = hf['coherence'][:]

    #  # Averaging bands of interest
    temp = np.zeros([nP, nT, len(bands), tarray.shape[0]])

    for i in range( len(bands) ):
        fidx = (freqs>=bands[i][0])*(freqs<bands[i][1])
        temp[:,:,i,:] = super_tensor[:,:,fidx,:].mean(axis=2)

    super_tensor = temp.copy()
    del temp

    path_st = os.path.join('super_tensors', dirs['monkey'][nmonkey] + '_session01_' + dirs['date'][nmonkey][idx]+ '.h5')
    with h5py.File(path_st, 'w') as hf:
        hf.create_dataset('supertensor', data=super_tensor)
        hf.create_dataset('freqs', data=freqs)
        hf.create_dataset('tarray', data=ses.tarray[::delta])
        hf.create_dataset('bands', data=bands)

    end = time.time()
    print('Elapsed time to compute coherences: ' +str((end - start)/60.0) + ' min.' )
