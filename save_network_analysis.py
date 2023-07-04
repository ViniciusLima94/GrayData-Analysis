##############################################################################
# Perform network measurements on the super tensor (coherence data)
##############################################################################
import numpy as np
import xarray as xr
import argparse
import os

from GDa.temporal_network import temporal_network
from GDa.net.layerwise import (compute_nodes_degree,
                               compute_nodes_efficiency,
                               compute_nodes_coreness,
                               compute_network_partition)
from config import get_dates, return_delay_split

from tqdm import tqdm

##############################################################################
# Argument parsing
##############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("METRIC",
                    help="which metric to load",
                    type=str)
parser.add_argument("SIDX",
                    help="index of the session to load",
                    type=int)
parser.add_argument("ALIGNED", help="wheter to align data to cue or match",
                    type=str)
parser.add_argument("MONKEY", help="which monkey to use",
                    type=str)
parser.add_argument("SURR", help="which monkey to use",
                    type=int)

args = parser.parse_args()

# The connectivity metric that should be used
metric = args.METRIC
# The index of the session to use
s_id = args.SIDX
# Wheter to align data to cue or match
at = args.ALIGNED
# Wheter to use Lucy or Ethyl's data 
monkey = args.MONKEY
# whether to use surrogate data or not
surr = args.SURR

early_cue, early_delay = return_delay_split(monkey=monkey, delay_type=0)

sessions = get_dates(monkey)


##############################################################################
# Get root path
##############################################################################

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')

# Path in which to save coherence data
_RESULTS = os.path.join('/home/vinicius/funcog/gda',
                        'Results',
                        monkey,
                        sessions[s_id],
                        'session01/network')

if not os.path.isdir(_RESULTS):
    os.mkdir(_RESULTS)

##############################################################################
# Get root path
##############################################################################
if not bool(surr):
    coh_file = f'{metric}_at_{at}.nc'
else:
    coh_file = f'{metric}_at_{at}_surr.nc'

coh_sig_file = None
if metric == "coh":
    coh_sig_file = f'thr_{metric}_at_{at}_surr.nc'
wt = None

##############################################################################
# Load the supertensor and convert to adjacency matrix
##############################################################################

net = temporal_network(coh_file=coh_file,
                       coh_sig_file=coh_sig_file, wt=wt, align_to=at,
                       early_cue=early_cue, early_delay=early_delay,
                       date=sessions[s_id], trial_type=[1],
                       behavioral_response=[1], monkey=monkey)

net.convert_to_adjacency()

# If the metric is pec take the absolute value of weigths only
# if metric == "pec":
    # net.super_tensor.values = np.abs(net.super_tensor.values)

##############################################################################
# 1. Strength
##############################################################################

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

if not bool(surr):
    fname = f"{metric}_degree_at_{at}.nc"
else:
    fname = f"{metric}_degree_at_{at}_surr.nc"

path_degree = os.path.join(_ROOT,
                           _RESULTS,
                           fname)

degree.to_netcdf(path_degree)

##############################################################################
# 2. Coreness
##############################################################################

# coreness = []
# for f in tqdm(range(net.A.sizes["freqs"])):
    # coreness += [compute_nodes_coreness(net.A.isel(freqs=f),
                                        # backend="brainconn",
                                        # kw_bc=dict(delta=0.5),
                                        # verbose=False, n_jobs=30)]
# coreness = xr.concat(coreness, "freqs")
# # Assign coords
# coreness = coreness.assign_coords({"trials": net.A.trials.data,
                                   # "roi": net.A.sources.data,
                                   # "freqs": net.A.freqs.data,
                                   # "times": net.A.times.data})

# coreness = coreness.transpose("trials", "roi", "freqs", "times")
# coreness.attrs = net.super_tensor.attrs

# if not bool(surr):
    # fname = f"{metric}_coreness_at_{at}.nc"
# else:
    # fname = f"{metric}_coreness_at_{at}_surr.nc"

# path_coreness = os.path.join(_ROOT,
                             # _RESULTS,
                             # fname)

# coreness.to_netcdf(path_coreness)

##############################################################################
# 3. Efficiency
##############################################################################

# efficiency = []
# for f in tqdm(range(net.A.sizes["freqs"])):
    # efficiency += [compute_nodes_efficiency(net.A.isel(freqs=f),
                                            # backend="igraph",
                                            # verbose=False, n_jobs=30)]
# efficiency = xr.concat(efficiency, "freqs")
# # Assign coords
# efficiency = efficiency.assign_coords({"trials": net.A.trials.data,
                                       # "roi": net.A.sources.data,
                                       # "freqs": net.A.freqs.data,
                                       # "times": net.A.times.data})

# efficiency = efficiency.transpose("trials", "roi", "freqs", "times")
# efficiency.attrs = net.super_tensor.attrs

# path_efficiency = os.path.join(_ROOT,
                               # _RESULTS,
                               # f"{metric}_efficiency_at_{at}.nc")

# efficiency.to_netcdf(path_efficiency)

##############################################################################
# 4. Modularity
##############################################################################

# partition, modularity = [], []
# for f in tqdm(range(net.A.sizes["freqs"])):
    # p, m = compute_network_partition(net.A.isel(freqs=f),
                                     # backend="igraph",
                                     # verbose=False, n_jobs=30)
    # partition += [p]
    # modularity += [m]

# partition = xr.concat(partition, "freqs")
# modularity = xr.concat(modularity, "freqs")
# # Assign coords
# partition = partition.assign_coords({"trials": net.A.trials.data,
                                     # "roi": net.A.sources.data,
                                     # "freqs": net.A.freqs.data,
                                     # "times": net.A.times.data})

# partition = partition.transpose("trials", "roi", "freqs", "times")
# partition.attrs = net.super_tensor.attrs

# modularity = modularity.assign_coords({"trials": net.A.trials.data,
                                       # "freqs": net.A.freqs.data,
                                       # "times": net.A.times.data})

# modularity = modularity.transpose("trials", "freqs", "times")
# modularity.attrs = net.super_tensor.attrs

# # Saving
# path_partition = os.path.join(_ROOT,
                              # _RESULTS,
                              # f"{metric}_partition_at_{at}.nc")

# partition.to_netcdf(path_partition)

# path_modularity = os.path.join(_ROOT,
                               # _RESULTS,
                               # f"{metric}_modularity_at_{at}.nc")

# modularity.to_netcdf(path_modularity)
