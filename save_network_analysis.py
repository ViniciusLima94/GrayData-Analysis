###############################################################################
# Perform network measurements on the super tensor (coherence data)
###############################################################################
import numpy as np
import xarray as xr
import argparse
import os

from GDa.temporal_network import temporal_network
from GDa.net.layerwise import (compute_nodes_degree,
                               compute_nodes_efficiency,
                               compute_nodes_coreness,
                               compute_network_partition)
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
parser.add_argument("ALIGNED", help="wheter to align data to cue or match",
                    type=str)

args = parser.parse_args()

# The connectivity metric that should be used
metric = args.METRIC
# The index of the session to use
s_id = args.SIDX
# Wheter to threshold coherence or not
thr = args.THRESHOLD
# Wheter to align data to cue or match
at = args.ALIGNED


##############################################################################
# Get root path
###############################################################################

# Path in which to save coherence data
_RESULTS = os.path.join('/home/vinicius/funcog/gda',
                        'Results',
                        'lucy',
                        sessions[s_id],
                        'session01')

##############################################################################
# Get root path
###############################################################################

if bool(thr):
    coh_sig_file = f'{metric}_csd_{mode}_at_{at}_surr.nc'
    wt = None
else:
    coh_sig_file = None
    wt = (-20, 20)

##############################################################################
# Load the supertensor and convert to adjacency matrix
###############################################################################

net = temporal_network(coh_file=f'{metric}_csd_{mode}_at_{at}.nc',
                       coh_sig_file=coh_sig_file, wt=wt,
                       date=sessions[s_id], trial_type=[1],
                       behavioral_response=[1])

net.convert_to_adjacency()

# If the metric is pec take the absolute value of weigths only
if metric == "pec":
    net.super_tensor.values = np.abs(net.super_tensor.values)

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

path_degree = os.path.join(_RESULTS,
                           f"{metric}_csd_degree_thr_{thr}_at_{at}.nc")

degree.to_netcdf(path_degree)
