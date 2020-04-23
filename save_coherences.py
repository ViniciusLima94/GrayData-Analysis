import sys
import numpy                         as     np
from   GDa.spectral_analysis         import spectral_analysis 
#from   GDa.spectral                  import *
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
'''
step = 25 
dt   = 250
fc   = np.arange(6, 62, 2)
df   = 4


LFP  = np.load(path, allow_pickle=True).item()

nP      = LFP['info']['nP']
nT      = LFP['info']['nT']
pairs   = LFP['info']['pairs']
tarray  = LFP['info']['tarray']
tidx    = np.arange(dt, LFP['data'].shape[2]-dt, step)
taxs    = LFP['info']['tarray'][tidx]
fsample = LFP['info']['fsample']
data    = LFP['data']
dir_out = LFP['path']['dir_out']

for trial in range(nT):
	Parallel(n_jobs=40, backend='loky', max_nbytes=1e6)( delayed(pairwise_coherence)
						        (data, trial, index_pair, taxs, tidx, fc, df, dt, fsample, pairs, dir_out, n_jobs = 1, save_to_file = True) 
						        for index_pair in range(nP) 
						)
'''
spectral = spectral_analysis(path = path, step = 25, dt = 250, fc = np.arange(6, 62, 2), df = 4, 
							 save_filtered = False, save_morlet = False, save_coh = True)

spectral.compute_coherences(n_jobs = -1)
#for trial in range(spectral.nT):
#	Parallel(n_jobs=40, backend='threading')( delayed(spectral.pairwise_coherence)(trial, ip, n_jobs = 1, save_to_file = True) for ip in range(spectral.nP) )
