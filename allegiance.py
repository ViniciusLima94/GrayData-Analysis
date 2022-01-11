##################################################################
# Compute modular structure across sessions
##################################################################
import os
import numpy as np
import xarray as xr

from config import sessions
from tqdm import tqdm
from GDa.flatmap.flatmap import flatmap
from GDa.net.layerwise import compute_network_partition
from GDa.net.temporal import compute_allegiance_matrix
from GDa.net.util import convert_to_adjacency
from GDa.temporal_network import temporal_network

##################################################################
# Loading temporal network
##################################################################

_ROOT = os.path.expanduser("~/storage1/projects/GrayData-Analysis")

# Path in which to save coherence data
_RESULTS = os.path.join("Results", "lucy", "141017", "session01")

coh_sig_file = "coh_k_0.3_multitaper_at_cue_surr.nc"
wt = None

net = temporal_network(
    coh_file="coh_k_0.3_multitaper_at_cue.nc",
    coh_sig_file=coh_sig_file,
    wt=wt,
    date="141017",
    trial_type=[1],
    behavioral_response=[1],
)

##################################################################
# Compute static coherence networks
##################################################################

# Stage masks
net.create_stage_masks(flatten=True)

coh = []
# Save attributes
attrs = net.super_tensor.attrs
# Stack trials and times
net.super_tensor = net.super_tensor.stack(samples=("trials", "times"))
# Average networks in each task stage
for stage in tqdm(net.s_mask.keys()):
    coh += [net.super_tensor
            .isel(samples=net.s_mask[stage].data)
            .mean("samples")]

# Reorganize dims and add dummy trials dim
net.super_tensor = xr.concat(coh, "times").transpose("roi", "freqs", "times")
del coh
net.super_tensor.attrs = attrs
net.super_tensor = net.super_tensor.expand_dims("trials", 2)
# Convert to adjacency
net.convert_to_adjacency()
A = net.A.copy()
A.attrs = attrs
del net

partition, modularity = [], []
for f in range(A.sizes["freqs"]):
    p, m = compute_network_partition(
        A.isel(freqs=f)
    )
    partition += [p]
    modularity += [m]
partition = xr.concat(partition, "freqs")
modularity = xr.concat(modularity, "freqs")
