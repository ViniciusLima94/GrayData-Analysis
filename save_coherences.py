import sys
import numpy                         as     np
from   GDa.spectral_analysis         import spectral_analysis
from   joblib import Parallel, delayed
import multiprocessing

idx = int(sys.argv[-1])

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
        
path = 'raw_lfp/'+dirs['monkey'][nmonkey]+'_'+'session01'+'_'+dirs['date'][nmonkey][idx]+'.npy'

spectral = spectral_analysis(path = path, step = 25, dt = 250, fc = np.arange(6, 62, 2), df = 4, 
							 save_filtered = False, save_morlet = False, save_coh = True)

spectral.compute_coherences(n_jobs = -1)
