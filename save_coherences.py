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

idx = 3#int(sys.argv[-1])

nmonkey = 0
nses    = 1
ntype   = 0

#################################################################################################
# Which trial type, alignment and behav. response to use
#################################################################################################
trial_type = 1
align_to  = 'cue'
behavioral_response = 1
#################################################################################################

#  Set the paths
paths        = set_paths(raw_path = dirs['rawdata'], monkey = dirs['monkey'][nmonkey], 
                         date = dirs['date'][nmonkey][idx], session = nses)
#  Instantiating session
ses = session(raw_path = dirs['rawdata'], monkey = dirs['monkey'][nmonkey], date = dirs['date'][nmonkey][idx], 
              session = nses, slvr_msmod = False, align_to = align_to, trial_type = trial_type,
              behavioral_response = behavioral_response , evt_dt = [-0.65, 3.00])
ses.read_from_mat()
#  Downsampled time array
tarray = ses.time[::delta]

if  __name__ == '__main__':

    start = time.time()

    tf.wavelet_coherence(data = ses.data, pairs = ses.readinfo['pairs'], fs = ses.readinfo['fsample'], 
                         freqs = freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, delta = delta, 
                         method = method, win_time = win_time, win_freq = win_freq, dir_out = paths.dir_out, n_jobs = -1)
    

    # Load all the files generated and save in a single file
    super_tensor = np.zeros([ses.readinfo['nP'], ses.readinfo['nT'], freqs.shape[0], tarray.shape[0]])
    for j in range(ses.readinfo['nP']):
        path = os.path.join(paths.dir_out, 
                            'ch1_'+str(ses.readinfo['pairs'][j,0])+'_ch2_'+str(ses.readinfo['pairs'][j,1])+'.h5' )
        with h5py.File(path, 'r') as hf:
                super_tensor[j,:,:,:] = hf['coherence'][:]

    #  # Averaging bands of interest
        temp = np.zeros([ses.readinfo['nP'], ses.readinfo['nT'], len(bands), tarray.shape[0]])

    for i in range( len(bands) ):
        fidx = (freqs>=bands[i][0])*(freqs<bands[i][1])
        temp[:,:,i,:] = super_tensor[:,:,fidx,:].mean(axis=2)

    super_tensor = temp.copy()
    del temp

    path_st = os.path.join('super_tensors', dirs['monkey'][nmonkey] + '_session01_' + dirs['date'][nmonkey][idx]+ '.h5')

    try:
        hf = h5py.File(path_st, 'r+')
    except:
        hf = h5py.File(path_st, 'w')

    # Create group
    group = os.path.join('trial_type_'+str(trial_type), 
                         'aligned_to_' + str(align_to),
                         'behavioral_response_'+str(behavioral_response)) 
    g1 = hf.create_group(group)
    g1.create_dataset('coherence', data=super_tensor)
    g1.create_dataset('freqs',     data=freqs)
    g1.create_dataset('tarray',    data=tarray)
    g1.create_dataset('bands',     data=bands)
    [g1.create_dataset('info/'+k, data=ses.readinfo[k]) for k in ses.readinfo.keys()]
    hf.close()

    end = time.time()
    print('Elapsed time to compute coherences: ' +str((end - start)/60.0) + ' min.' )
