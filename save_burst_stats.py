import GDa.stats.bursting                as     bst
from   GDa.session                       import session
from   GDa.temporal_network              import temporal_network
from   GDa.util                          import smooth

import seaborn                           as       sns
import numpy                             as       np
import xarray                            as       xr
import matplotlib.pyplot                 as       plt
import scipy.signal
import time
import os
import h5py

from   tqdm                              import tqdm
from   sklearn.manifold                  import TSNE
from   config                            import *
from   scipy                             import stats

# Bands names
band_names  = [r'$\theta$', r'$\alpha$', r'$\beta$', r'h-$\beta$', r'gamma']
# Stage names
stages      = ['baseline', 'cue', 'delay', 'match']
# Burst stats. names
stats_names = [r"$\mu$","std$_{\mu}$",r"$\mu_{tot}$","CV"]

##################################################################################
# Parameters to read the data (temporary)
##################################################################################
idx      = 3
nses     = 1
nmonkey  = 0
align_to = 'cue'

dirs = { 'rawdata':'/home/vinicius/storage1/projects/GrayData-Analysis/GrayLab',
         'results':'Results/',
         'monkey' :['lucy', 'ethyl'],
         'session':'session01',
         'date'   :[['141014', '141015', '141205', '150128', '150211', '150304'], []] }

##################################################################################
# Config params to specify which coherence file to read
##################################################################################
_REL  = False # If threshold is relative or absolute
_SURR = False # If it is surrogate data or not
_KS   = 500   # 0.5s kernel size

if _SURR:
    _COH_FILE = f'super_tensor_s{12000}_k{_KS}.nc'
else:
    _COH_FILE = f'super_tensor_k{_KS}.nc'


##################################################################################
# Instantiate a dummy temp net
##################################################################################

# Instantiating a temporal network object without thresholding the data
net =  temporal_network(coh_file=_COH_FILE, monkey=dirs['monkey'][nmonkey], 
                        session=1, date='150128', trial_type=[1],
                        behavioral_response=[1], wt=(20,20), 
                        verbose=True, q=None)

##################################################################################
# Compute burstness statistics for different thresholds
##################################################################################

# Define list of thresholds
q_list  = np.arange(0.2, 1.0, 0.1)

# Store burst stats. for each link in each stage and frequency band
bs_stats = np.zeros([len(q_list), net.super_tensor.shape[0], len(stages), 4])

for j in tqdm( range(len(q_list)) ):
    ## Default threshold
    kw = dict(q=q_list[j], keep_weights=False, relative=_REL)

    # Instantiating a temporal network object without thresholding the data
    net =  temporal_network(coh_file=_COH_FILE, monkey=dirs['monkey'][nmonkey],
                            session=1, date='150128', trial_type=[1],
                            behavioral_response=[1], wt=(20,20), drop_trials_after=True,
                            verbose=False, **kw)
    # Creating mask for stages
    net.create_stage_masks(flatten=False)

    n_samp = []
    for stage in stages:
        n_samp += [net.get_number_of_samples(stage=stage, total=True)]

    bs_stats[j] = bst.tensor_burstness_stats(net.super_tensor.isel(freqs=1), net.s_mask,
                                             drop_edges=True, samples=n_samp,
                                             dt=delta/net.super_tensor.attrs['fsample'],
                                             n_jobs=20, verbose=False)

bs_stats = xr.DataArray(bs_stats, dims=("thr","roi","stages","stats"),
                        coords={"thr":q_list,
                                "roi":net.super_tensor.roi,
                                "stages":stages,
                                "stats":stats_names})

bs_stats.to_netcdf(f"bs_stats_k_{_KS}_surr_{_SURR}_rel_{_REL}.nc")
