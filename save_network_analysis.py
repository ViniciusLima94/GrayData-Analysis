###############################################################################
# Perform network measurements on the super tensor (coherence data)
###############################################################################
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

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')

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
    coh_sig_file = f'{metric}_k_0.3_{mode}_at_{at}_surr.nc'
    wt = None
else:
    coh_sig_file = None
    wt = (-20, 20)

##############################################################################
# Load the supertensor and convert to adjacency matrix
###############################################################################

net = temporal_network(coh_file=f'{metric}_k_0.3_{mode}_at_{at}.nc',
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
                           f"{metric}_degree_thr_{thr}_at_{at}.nc")

degree.to_netcdf(path_degree)

###############################################################################
# 2. Coreness
###############################################################################

coreness = []
for f in tqdm(range(net.A.sizes["freqs"])):
    coreness += [compute_nodes_coreness(net.A.isel(freqs=f),
                                        backend="brainconn",
                                        kw_bc=dict(delta=0.5),
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
                             f"{metric}_coreness_thr_{thr}_at_{at}.nc")

coreness.to_netcdf(path_coreness)

###############################################################################
# 3. Efficiency
###############################################################################

efficiency = []
for f in tqdm(range(net.A.sizes["freqs"])):
    efficiency += [compute_nodes_efficiency(net.A.isel(freqs=f),
                                            backend="igraph",
                                            verbose=False, n_jobs=20)]
efficiency = xr.concat(efficiency, "freqs")
# Assign coords
efficiency = efficiency.assign_coords({"trials": net.A.trials.data,
                                       "roi": net.A.sources.data,
                                       "freqs": net.A.freqs.data,
                                       "times": net.A.times.data})

efficiency = efficiency.transpose("trials", "roi", "freqs", "times")
efficiency.attrs = net.super_tensor.attrs

path_efficiency = os.path.join(_ROOT,
                               _RESULTS,
                               f"{metric}_efficiency_thr_{thr}_at_{at}.nc")

efficiency.to_netcdf(path_efficiency)

###############################################################################
# 4. Modularity
###############################################################################

partition, modularity = [], []
for f in tqdm(range(net.A.sizes["freqs"])):
    p, m = compute_network_partition(net.A.isel(freqs=f),
                                     backend="igraph",
                                     verbose=False, n_jobs=20)
    partition += [p]
    modularity += [m]

partition = xr.concat(partition, "freqs")
modularity = xr.concat(modularity, "freqs")
# Assign coords
partition = partition.assign_coords({"trials": net.A.trials.data,
                                     "roi": net.A.sources.data,
                                     "freqs": net.A.freqs.data,
                                     "times": net.A.times.data})

partition = partition.transpose("trials", "roi", "freqs", "times")
partition.attrs = net.super_tensor.attrs

modularity = modularity.assign_coords({"trials": net.A.trials.data,
                                       "freqs": net.A.freqs.data,
                                       "times": net.A.times.data})

modularity = modularity.transpose("trials", "freqs", "times")
modularity.attrs = net.super_tensor.attrs

# Saving
path_partition = os.path.join(_ROOT,
                              _RESULTS,
                              f"{metric}_partition_thr_{thr}_at_{at}.nc")

partition.to_netcdf(path_partition)

path_modularity = os.path.join(_ROOT,
                               _RESULTS,
                               f"{metric}_modularity_thr_{thr}_at_{at}.nc")

modularity.to_netcdf(path_modularity)
