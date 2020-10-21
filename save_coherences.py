import sys
import time
import numpy                           as     np
from   GDa.spectral_analysis           import spectral_analysis
from   joblib import Parallel, delayed

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
# Directory were to save the coherence data
dir_out      = session_data['path']['dir_out']

if  __name__ == '__main__':

    start = time.time()

    spec.wavelet_coherence(data = LFP, pairs = pairs, fs = fsample, freqs = freqs, 
                           n_cycles = n_cycles, time_bandwidth = None, delta = delta, method = 'morlet', 
                           win_time = 34, win_freq = 1, dir_out = dir_out, n_jobs = -1)

    end = time.time()
    print('Elapsed time to compute coherences: ' +str((end - start)/60.0) + ' min.' )