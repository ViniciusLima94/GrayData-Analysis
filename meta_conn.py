"""
Computes meta-connectivity at roi level.
"""
import os
import xarray as xr
import numpy as np
import argparse

from frites.utils import parallel_func
from GDa.util import _extract_roi
from GDa.temporal_network import temporal_network
from GDa.fc.mc import meta_conn
from config import sessions
from tqdm import tqdm

###############################################################################
# Argument parsing
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to run",
                    type=int)
parser.add_argument("METRIC",
                    help="which FC metric to use",
                    type=str)
args = parser.parse_args()
metric = args.METRIC
idx = args.SIDX
session = sessions[idx]

###############################################################################
# Loading temporal network
###############################################################################
_ROOT = os.path.expanduser("~/funcog/gda")

# Path in which to save coherence data
_RESULTS = os.path.join("Results", "lucy", session, "session01")

coh_file = f'{metric}_k_0.3_multitaper_at_cue.nc'
coh_sig_file = f'{metric}_k_0.3_multitaper_at_cue_surr.nc'
wt = None

net = temporal_network(
    coh_file=coh_file,
    coh_sig_file=coh_sig_file,
    wt=wt,
    date=session,
    trial_type=[1],
    behavioral_response=[1],
)

# Masks for each stage
net.create_stage_masks(flatten=True)
# Stack trials
FC = net.super_tensor.stack(obs=("trials", "times")).data

###############################################################################
# Compute meta-connectivity
###############################################################################
n_edges = net.super_tensor.sizes["roi"]
n_freqs = net.super_tensor.sizes["freqs"]
stages = net.s_mask.keys()
n_stages = len(stages)

MC = np.zeros((n_edges, n_edges, n_freqs, n_stages))
for f in tqdm(range(n_freqs)):
    for s, stage in enumerate(net.s_mask.keys()):
        # Get index for the stages of interest
        idx = net.s_mask[stage].values
        MC[..., f, s] = np.corrcoef(FC[:, f, idx])

MC = xr.DataArray(MC,
                  dims = ("sources", "targets", "freqs", "times"),
                  coords = {"sources": net.super_tensor.roi.values,
                            "targets": net.super_tensor.roi.values,
                            "freqs": net.super_tensor.freqs.values,})

MC.attrs = net.super_tensor.attrs

###############################################################################
# Saving data
###############################################################################
_PATH = os.path.expanduser(os.path.join(_ROOT,
                                        "Results/lucy/meta_conn"))

# Save MC
MC.to_netcdf(os.path.join(_PATH, f"MC_{metric}_{session}.nc"))
