"""
Computes meta-connectivity at roi level.
"""
import os
import argparse
import xarray as xr
import numpy as np

from tqdm import tqdm
from GDa.temporal_network import temporal_network
from GDa.signal.surrogates import trial_swap_surrogates
from config import get_dates

##############################################################################
# Argument parsing
##############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to run",
                    type=int)
parser.add_argument("METRIC",
                    help="which FC metric to use",
                    type=str)
parser.add_argument("SURR",
                    help="wheter to use original or surrogate MC",
                    type=int)
parser.add_argument("THR",
                    help="wheter to threshold or not the coherence",
                    type=int)
parser.add_argument("MONKEY", help="which monkey to use",
                    type=str)

args = parser.parse_args()

metric = args.METRIC
surr = args.SURR
idx = args.SIDX
thr = args.THR
monkey = args.MONKEY

if monkey == "lucy":
    early_cue=0.2
    early_delay=0.3
elif monkey == "ethyl":
    early_cue=0.2
    early_delay=0.24

sessions = get_dates(monkey)
session = sessions[idx]

print(surr)
print(thr)

##############################################################################
# Loading temporal network
##############################################################################
_ROOT = os.path.expanduser("~/funcog/gda")

# Path in which to save coherence data
_RESULTS = os.path.join("Results", monkey, session, "session01")

if surr == 1:
    coh_file = f'{metric}_at_cue_surr.nc'
    coh_sig_file = None
else:
    coh_file = f'{metric}_at_cue.nc'
    coh_sig_file = None
    if bool(thr):
        coh_sig_file = f'thr_{metric}_at_cue_surr.nc'

wt = None

net = temporal_network(
    coh_file=coh_file,
    coh_sig_file=coh_sig_file,
    early_cue=early_cue, early_delay=early_delay,
    wt=wt, monkey=monkey,
    date=session,
    trial_type=[1],
    behavioral_response=[1],
)

# Masks for each stage
net.create_stage_masks(flatten=True)

if surr == 2:
    FC = net.super_tensor.transpose("trials", "roi", "times", "freqs")
    FC_surr = [trial_swap_surrogates(FC[..., f], verbose=True) for f in range(10)]
    FC = xr.concat(FC_surr, "freqs").assign_coords({"freqs": FC.freqs})
    FC = FC.transpose("roi", "freqs", "trials", "times").stack(obs=("trials", "times")).data
else:
    # Stack trials
    FC = net.super_tensor.stack(obs=("trials", "times")).data

##############################################################################
# Compute meta-connectivity
##############################################################################
n_edges = net.super_tensor.sizes["roi"]
n_freqs = net.super_tensor.sizes["freqs"]

stages = net.s_mask.keys()
n_stages = len(stages)

print(FC.shape)

MC = np.zeros((n_edges, n_edges, n_freqs, n_stages))
for f in tqdm(range(n_freqs)):
    for s, stage in enumerate(net.s_mask.keys()):
        # get index for the stages of interest
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
                                        f"Results/{monkey}/meta_conn"))
# Save MC
if bool(surr):
    MC.to_netcdf(os.path.join(_PATH, f"MC_{metric}_{session}_surr_{surr}.nc"))
else:
    MC.to_netcdf(os.path.join(_PATH, f"MC_{metric}_{session}_thr_{thr}.nc"))
