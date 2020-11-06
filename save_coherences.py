import sys
import os
import time
import numpy                           as     np
import h5py
from   GDa.spectral_analysis           import spectral_analysis
from   joblib                          import Parallel, delayed

idx = 3#int(sys.argv[-1])

nmonkey = 0
nses    = 6
ntype   = 0
#####################################################################################################
# Directories
#####################################################################################################
dirs = {'rawdata':'GrayLab/',
        'results':'Results/',
        'monkey' :['lucy', 'ethyl'],
        'session':'session01',
        'date'   :[['141014', '141015', '141205', '150128', '150211', '150304'], []]
        }
     
# Raw LFP path   
path = 'raw_lfp/'+dirs['monkey'][nmonkey]+'_'+'session01'+'_'+dirs['date'][nmonkey][idx]+'.npy'
# Range of frequencies to be analyzed
freqs = np.arange(4,60,2)
# Delta for downsampling
delta = 15
# Number of cycles for the wavelet
n_cycles = 5
# Instantiating spectral analysis class
spec = spectral_analysis()
# Loading data info
session_data = np.load(path, allow_pickle = True).item()
# LFP data
LFP          = session_data['data']
# Index of all pair of channels
pairs        = session_data['info']['pairs']
# Sample frequency
fsample      = int(session_data['info']['fsample'])
# Number of pairs
nP           = session_data['info']['nP']
# Number of trials
nT           = session_data['info']['nT']
# Time array
tarray       = session_data['info']['tarray'][::delta]
# Directory were to save the coherence data
dir_out      = session_data['path']['dir_out']
# Bands of interest
bands = np.array([[4,8],[8,15],[15,30],[30,60]])

if  __name__ == '__main__':

    start = time.time()

    spec.wavelet_coherence(data = LFP, pairs = pairs, fs = fsample, freqs = freqs, 
                           n_cycles = n_cycles, time_bandwidth = None, delta = delta, method = 'morlet', 
                           win_time = 34, win_freq = 1, dir_out = dir_out, n_jobs = -1)

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
        hf.create_dataset('tarray', data=tarray)
        hf.create_dataset('bands', data=bands)

    end = time.time()
    print('Elapsed time to compute coherences: ' +str((end - start)/60.0) + ' min.' )
