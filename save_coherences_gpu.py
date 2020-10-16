import sys
import numpy                           as     np
from   GDa.spectral_analysis           import spectral_analysis
from   numba import vectorize

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
# Instantiating spectral analysis class
spec = spectral_analysis(session = None, path = path, freqs = freqs, delta=delta)

@vectorize(['void(int32, int32)'], target='cuda')
def save_coherences(trial_number, index_pair):
    spec._gabor_coherence(trial = trial_number, index_pair = index_pair, 
		                  win_time = 500, win_freq = 1, n_cycles = 6.0, save_to_file=True) 
    #file_name = os.path.join( dir_out, 'trial_' +str(trial) + '_pair_' + str(int(index_pair)) + '.npy')
    #np.save(file_name, {'coherence' : coh, 'freqs': freqs, 'time': tarray})

if  __name__ == '__main__':

	trial_array = np.arange(0,spec.nT)
	pairs_array = np.arange(0, spec.nP)

	save_coherences(trial_array, pairs_array)