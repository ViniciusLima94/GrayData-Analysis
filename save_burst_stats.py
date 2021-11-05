import sys
import argparse

import GDa.stats.bursting                as     bst
from   GDa.session                       import session
from   GDa.temporal_network              import temporal_network

import numpy                             as       np
import xarray                            as       xr
import os

from   tqdm                              import tqdm
from   config                            import *

# Bands names
band_names  = [r'$\theta$', r'$\alpha$', r'$\beta$', r'h-$\beta$', r'gamma']
# Stage names
stages      = ['baseline', 'cue', 'delay', 'match']
# Burst stats. names
stats_names = [r"$\mu$","std$_{\mu}$",r"$\mu_{tot}$","CV"]

##################################################################################
# Config params to specify which coherence file to read
##################################################################################
#  mode    = sys.argv[-2]
#  idx     = int(sys.argv[-1])

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("METRIC", help="which connectivity metric to use", 
                    type=str)
parser.add_argument("MODE", help="wheter to load coherence computed with morlet or multitaper", choices=["morlet","multitaper"],
                    type=str)
parser.add_argument("IDX", help="which condition to run",
                    type=int)
args   = parser.parse_args()
# The connectivity metric that should be used
metric = args.METRIC
mode   = args.MODE
idx    = args.IDX

#                 _REL   _SURR
#  pars  = np.array([[False,False],
#                    [False,True],
#                    [True,False],
#                    [True,True]])
pars = np.array([False, True])

# If threshold is relative or absolute
#  _REL  = pars[idx,0] 
_REL  = pars[idx] 
# If it is surrogate data or not
#  _SURR = pars[idx,1] 
_KS   = 0.3   # 0.3s kernel size

_COH_FILE     = f'{metric}_k_{_KS}_{mode}.nc'
_COH_FILE_SIG = f'{metric}_k_{_KS}_{mode}_surr.nc'

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
         'date'   :[['141014', '141015', '141205', '141017', '150128', '150211', '150304'], []] }

# Path in which to save burst stats data
path_st = os.path.join('Results', str(dirs['monkey'][nmonkey]), str(dirs['date'][nmonkey][idx]), f'session0{nses}')
path_st = os.path.join(path_st, f"bs_stats_k_{_KS}_rel_{_REL}_numba_{mode}.nc")

##################################################################################
# Instantiate a dummy temp net
##################################################################################

# Instantiating a temporal network object without thresholding the data
net =  temporal_network(coh_file=_COH_FILE, coh_sig_file=_COH_FILE_SIG, 
                        date='141017', trial_type=[1], behavioral_response=[1])


##################################################################################
# Compute burstness statistics for different thresholds
##################################################################################

# Define list of thresholds
q_list  = np.array([0])#np.arange(0.2, 1.0, 0.1)

# Store burst stats. for each link in each stage and frequency band
bs_stats = np.zeros((len(q_list), net.super_tensor.sizes["freqs"], net.super_tensor.shape[0], len(stages), 4))

for j in tqdm( range(len(q_list)) ):
    ## Default threshold
    #  kw  = dict(q=q_list[j], relative=_REL)

    #  coh = net.get_thresholded_coherence(q_list[j], _REL, True) 
    coh = (net.super_tensor>0)
 
    #  net =  temporal_network(coh_file=_COH_FILE, coh_sig_file=_COH_FILE_SIG, **kw,
    #                          date='141017', trial_type=[1], behavioral_response=[1])

    #  net =  temporal_network(coh_file=_COH_FILE, monkey=dirs['monkey'][nmonkey],
    #                          session=1, date='150128', trial_type=[1],
    #                          behavioral_response=[1], wt=(20,20), drop_trials_after=True,
    #                          verbose=False, **kw)
    # Creating mask for stages
    net.create_stage_masks(flatten=False)

    n_samp = []
    for stage in stages:
        n_samp += [net.get_number_of_samples(stage=stage, total=True)]

    np_mask = {}
    for key in net.s_mask.keys(): np_mask[key] = net.s_mask[key].values

    for f in range(net.super_tensor.sizes["freqs"]):
        bs_stats[j,f] = bst.tensor_burstness_stats(coh.isel(freqs=f).values, np_mask,
                                                   drop_edges=True, samples=n_samp, find_zeros=True,
                                                   dt=delta/net.super_tensor.attrs['fsample'],
                                                   n_jobs=1)

bs_stats = xr.DataArray(bs_stats, dims=("thr","freqs","roi","stages","stats"),
                        coords={"thr":q_list,
                                "freqs":net.super_tensor.freqs,
                                "roi":net.super_tensor.roi,
                                "stages":stages,
                                "stats":stats_names})

bs_stats.to_netcdf(path_st)
