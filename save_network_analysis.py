###############################################################################
# Perform network measurements on the super tensor (coherence data)
###############################################################################
import xarray as xr
import argparse
import os

from GDa.temporal_network import temporal_network
from GDa.net.layerwise import (compute_nodes_degree,
                               compute_nodes_coreness,
                               compute_nodes_coreness_bc)
from config import mode, sessions
from tqdm import tqdm

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("METRIC",
                    help="which metric to load",
                    type=str)
parser.add_argument("SIDX",
                    help="index of the session to load",
                    type=int)
parser.add_argument("THRESHOLD",
                    help="wheter to threshold the coherence or not",
                    type=int)

args = parser.parse_args()

metric = args.METRIC
s_id = args.SIDX
thr = args.THRESHOLD


##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')

# Path in which to save coherence data
_RESULTS = os.path.join('Results',
                        'lucy',
                        sessions[s_id],
                        'session01')

##############################################################################
# Get root path
###############################################################################

if bool(thr):
    coh_sig_file = f'coh_k_0.3_{mode}_surr.nc'
    wt = None
else:
    coh_sig_file = None
    wt = (-20, 20)

##############################################################################
# Load the supertensor and convert to adjacency matrix
###############################################################################

net = temporal_network(coh_file=f'{metric}_k_0.3_{mode}_at_cue.nc',
                       coh_sig_file=coh_sig_file, wt=wt,
                       date=sessions[s_id], trial_type=[1],
                       behavioral_response=[1])

net.convert_to_adjacency()


###############################################################################
# 1. Strength
###############################################################################

degree = []
for f in tqdm(range(net.A.sizes["freqs"])):
    degree += [compute_nodes_degree(net.A.isel(freqs=f))]
degree = xr.concat(degree, "freqs")
# Assign coords
degree = degree.assign_coords({"trials": net.A.trials.data,
                               "roi": net.A.sources.data,
                               "freqs": net.A.freqs.data,
                               "times": net.A.times.data})

degree = degree.transpose("trials", "roi", "freqs", "times")
degree.attrs = net.super_tensor.attrs

path_degree = os.path.join(_ROOT,
                           _RESULTS,
                           f"degree_thr_{thr}.nc")

degree.to_netcdf(path_degree)

###############################################################################
# 2. Coreness
###############################################################################

coreness = []
for f in tqdm(range(net.A.sizes["freqs"])):
    coreness += [compute_nodes_coreness_bc(net.A.isel(freqs=f),
                                           return_degree=False, delta=0.5,
                                           verbose=False, n_jobs=20)]
coreness = xr.concat(coreness, "freqs")
# Assign coords
coreness = coreness.assign_coords({"trials": net.A.trials.data,
                                   "roi": net.A.sources.data,
                                   "freqs": net.A.freqs.data,
                                   "times": net.A.times.data})

coreness = coreness.transpose("trials", "roi", "freqs", "times")
coreness.attrs = net.super_tensor.attrs

path_coreness = os.path.join(_ROOT,
                             _RESULTS,
                             f"coreness_thr_{thr}.nc")

coreness.to_netcdf(path_coreness)
