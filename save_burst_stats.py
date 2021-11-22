import argparse

import GDa.stats.bursting as bst
from GDa.temporal_network import temporal_network

import numpy as np
import xarray as xr
import os

from config import delta

# Bands names
band_names = [r'$\theta$', r'$\alpha$', r'$\beta$', r'h-$\beta$', r'gamma']
# Stage names
stages = ['baseline', 'cue', 'delay', 'match']
# Burst stats. names
stats_names = [r"$\mu$", r"std$_{\mu}$", r"$\mu_{tot}$", "CV"]

###############################################################################
# Config params to specify which coherence file to read
###############################################################################

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("METRIC", help="which connectivity metric to use",
                    type=str)
parser.add_argument("MODE",
                    help="wheter to load coherence computed\
                          with morlet or multitaper",
                    choices=["morlet", "multitaper"],
                    type=str)
parser.add_argument("IDX", help="which session to run",
                    type=int)
args = parser.parse_args()
# The connectivity metric that should be used
metric = args.METRIC
# Wheter it is morlet or multitaper coherence
mode = args.MODE
# Index of the session to be load
idx = args.IDX

###############################################################################
# Parameters to read the data
###############################################################################

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')
# Get session number
sessions = np.loadtxt("GrayLab/lucy/sessions.txt", dtype=str)
session = sessions[idx]
_KS = 0.3   # 0.3s kernel size

_COH_FILE = f'{metric}_k_{_KS}_{mode}.nc'
_COH_FILE_SIG = f'{metric}_k_{_KS}_{mode}_surr.nc'

###############################################################################
# Path in which to save burst stats data
###############################################################################

path_st = os.path.join(_ROOT, f"Results/lucy/{session}/session01")
path_st = os.path.join(path_st, f"bs_stats_k_{_KS}_numba_{mode}.nc")

###############################################################################
# Instantiate temp net
###############################################################################

# Instantiating a temporal network object without thresholding the data
net = temporal_network(coh_file=_COH_FILE, coh_sig_file=_COH_FILE_SIG,
                       date=session, trial_type=[1], behavioral_response=[1])

###############################################################################
# Compute burstness statistics for different thresholds
###############################################################################

bs_stats = np.zeros((2, net.super_tensor.sizes["freqs"],
                     net.super_tensor.shape[0], len(stages), 4))


# Set to one all values about siginificance level
coh = (net.super_tensor > 0)
# Wheter to compute the burst stats for sequences of siliences
# or activations
find_zeros = False

for j in range(2):

    # Creating mask for stages
    net.create_stage_masks(flatten=False)

    n_samp = []
    for stage in stages:
        n_samp += [net.get_number_of_samples(stage=stage, total=True)]

    np_mask = {}
    for key in net.s_mask.keys():
        np_mask[key] = net.s_mask[key].values

    if j == 1:
        find_zeros = True

    for f in range(net.super_tensor.sizes["freqs"]):
        bs_stats[j, f] = bst.tensor_burstness_stats(
            coh.isel(freqs=f).values, np_mask,
            drop_edges=True, samples=n_samp, find_zeros=find_zeros,
            dt=delta/net.super_tensor.attrs['fsample'],
            n_jobs=1)

bs_stats = xr.DataArray(bs_stats,
                        dims=("zeros", "freqs", "roi", "stages", "stats"),
                        coords={"zeros":  [0, 1],
                                "freqs":  net.super_tensor.freqs,
                                "roi":    net.super_tensor.roi,
                                "stages": stages,
                                "stats":  stats_names},
                        attrs=net.super_tensor.attrs)

bs_stats.to_netcdf(path_st)
