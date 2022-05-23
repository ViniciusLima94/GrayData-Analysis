"""
Computes meta-connectivity at roi level.
"""
import os
import argparse
import xarray as xr
import numpy as np

from tqdm import tqdm
from GDa.temporal_network import temporal_network
from config import sessions

##############################################################################
# Argument parsing
##############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to run",
                    type=int)
parser.add_argument("METRIC",
                    help="which FC metric to use",
                    type=str)
parser.add_argument("GLOBAL",
                    help="wheter to measure stage based or global MC",
                    type=int)
parser.add_argument("SURR",
                    help="wheter to use original or surrogate MC",
                    type=int)
args = parser.parse_args()
metric = args.METRIC
_global = args.GLOBAL
surr = args.SURR
idx = args.SIDX
session = sessions[idx]

##############################################################################
# Loading temporal network
##############################################################################
_ROOT = os.path.expanduser("~/funcog/gda")

# Path in which to save coherence data
_RESULTS = os.path.join("Results", "lucy", session, "session01")

if not bool(surr):
    coh_file = f'{metric}_at_cue.nc'
    coh_sig_file = f'thr_{metric}_at_cue_surr.nc'
else:
    coh_file = f'{metric}_at_cue_surr.nc'
    coh_sig_file = None 
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

##############################################################################
# Compute meta-connectivity
##############################################################################
n_edges = net.super_tensor.sizes["roi"]
n_freqs = net.super_tensor.sizes["freqs"]

if not _global:
    stages = net.s_mask.keys()
    n_stages = len(stages)

    MC = np.zeros((n_edges, n_edges, n_freqs, n_stages))
    for f in tqdm(range(n_freqs)):
        for s, stage in enumerate(net.s_mask.keys()):
            # Get index for the stages of interest
            idx = net.s_mask[stage].values
            MC[..., f, s] = np.corrcoef(FC[:, f, idx])

    MC = xr.DataArray(MC,
                      dims=("sources", "targets", "freqs", "times"),
                      coords={"sources": net.super_tensor.roi.values,
                              "targets": net.super_tensor.roi.values,
                              "freqs": net.super_tensor.freqs.values, })

    MC.attrs = net.super_tensor.attrs
    # Saving data
    _PATH = os.path.expanduser(os.path.join(_ROOT,
                                            "Results/lucy/meta_conn"))
    # Save MC
    if bool(surr):
        MC.to_netcdf(os.path.join(_PATH, f"MC_{metric}_{session}_surr.nc"))
    else:
        MC.to_netcdf(os.path.join(_PATH, f"MC_{metric}_{session}.nc"))
else:
    MC = np.zeros((n_edges, n_edges, n_freqs))
    for f in tqdm(range(n_freqs)):
        MC[..., f] = np.corrcoef(FC[:, f])

    MC = xr.DataArray(MC,
                      dims=("sources", "targets", "freqs"),
                      coords={"sources": net.super_tensor.roi.values,
                              "targets": net.super_tensor.roi.values,
                              "freqs": net.super_tensor.freqs.values, })
    MC.attrs = net.super_tensor.attrs
    # Saving data
    _PATH = os.path.expanduser(os.path.join(_ROOT,
                                            "Results/lucy/meta_conn"))
    # Save MC
    if bool(surr):
        MC.to_netcdf(os.path.join(
            _PATH, f"MC_{metric}_{session}_global_surr.nc"))
    else:
        MC.to_netcdf(os.path.join(
            _PATH, f"MC_{metric}_{session}_global.nc"))
