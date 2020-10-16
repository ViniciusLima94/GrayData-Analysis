import sys
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
freqs = np.arange(6,100,1)
# Delta for downsampling
delta = 15
# Instantiating spectral analysis class
spec = spectral_analysis(session = None, path = path, freqs = freqs, delta=delta)
'''
def save_coherences(trial_number, index_pair):
    ch1, ch2 = pairs[index_pair,0], pairs[index_pair,1]
    lfp1 = data[trial_number, ch1,:]
    lfp2 = data[trial_number, ch2,:]
    coh =  gabor_coherence(signal1 = lfp1, signal2 = lfp2, fs = fsample, freqs = freqs, win_time = 500, win_freq = 1, n_cycles = 6.0)
    file_name = os.path.join( dir_out, 'trial_' +str(trial) + '_pair_' + str(int(index_pair)) + '.npy')
    np.save(file_name, {'coherence' : coh, 'freqs': freqs, 'time': tarray})
'''
# Computing in parallel for each pair
#for trial in range(nT):
#    print('Trial = ' + str(trial))
#    Parallel(n_jobs=-1, backend='loky', max_nbytes=1e6)(delayed(save_coherences)(trial, index_pair) for index_pair in range(nP) )

if  __name__ == '__main__':
	spec.parallel_wavelet_coherence(n_cycles = freqs/2.0, win_time = 500, win_freq = 1, 
                                        time_bandwidth = 8.0, method = 'multitaper', backend = 'loky', n_jobs=-1)
