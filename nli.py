"""
Nodes-link Inter-relation coefficient

Does coherence is fully explained by local power? This can be adressed via
the nodes-link inter-realtion coefficient (NLI):

$NIL_{ij} = <H[Z(POW_i)]H[Z(POW_j)]H[Z(COH_{ij})]> +
<\tilde{H}[Z(POW_i)]\tilde{H}[Z(POW_j)]\tilde{H}[Z(COH_{ij})]>$

where,

$H[x] = x$, if $x >= 0$ and $\tilde{H}[x] = -x$, if $x <= 0$
"""
# Adding GDa to path
import os
import numba as nb
import numpy as np
import xarray as xr
# import pandas as pd
import argparse

from GDa.temporal_network import temporal_network
from config import get_dates
from tqdm import tqdm


###############################################################################
# Argument parsing
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to run",
                    type=int)
parser.add_argument("METRIC", help="which dFC metric to use",
                    type=str)
parser.add_argument("MONKEY", help="which monkey to use",
                    type=str)
args = parser.parse_args()
# The index of the session to use
idx = args.SIDX
# Get name of the dFC metric
metric = args.METRIC
# Wheter to use Lucy or Ethyl's data 
monkey = args.MONKEY

if monkey == "lucy":
    early_cue=0.2
    early_delay=0.3
elif monkey == "ethyl":
    early_cue=0.2
    early_delay=0.24

sessions = get_dates(monkey)

###############################################################################
# Loading power and temporal network
###############################################################################

_ROOT = os.path.expanduser("~/funcog/gda")
_RESULTS = os.path.join("Results", monkey, sessions[idx], "session01")

# dFC files
power_file = "power_tt_1_br_1_at_cue.nc"
coh_file = f"{metric}_at_cue.nc"
coh_sig_file = f"thr_{metric}_at_cue_surr.nc"
wt = None

power = xr.load_dataarray(os.path.join(_ROOT, _RESULTS, power_file))
power = power.transpose("roi", "freqs", "trials", "times")

net = temporal_network(
    coh_file=coh_file,
    coh_sig_file=coh_sig_file,
    early_cue=early_cue, early_delay=early_delay,
    wt=wt, monkey=monkey,
    date=sessions[idx],
    trial_type=[1],
    behavioral_response=[1],
)

# Concatenate trials and times
# power = power.stack(samples=("trials", "times"))
coh = net.super_tensor  # .stack(samples=("trials", "times"))

rois = coh.roi.values
channels = coh.attrs["channels_labels"]

# z-score
Zpower = (power - power.mean("times")) / power.std("times")
Zcoh = (coh - coh.mean("times")) / coh.std("times")

roi = Zcoh.roi.data
freqs = Zcoh.freqs.data
n_rois = len(roi)
n_freqs = len(freqs)

###############################################################################
# Compute NLI
###############################################################################


@nb.vectorize([nb.float64(nb.float64)])
def H(x):
    if x > 0:
        return 1
    return 0


@nb.vectorize([nb.float64(nb.float64)])
def H_tilde(x):
    if x < 0:
        return 1
    return 0


@nb.vectorize([nb.int64(nb.float64, nb.float64)])
def heaviside(x, thr=0.0):
    if x > thr:
        return 1
    return 0


sources = net.super_tensor.attrs["sources"].astype(int)
targets = net.super_tensor.attrs["targets"].astype(int)
areas = np.asarray(net.super_tensor.attrs["areas"])


def compute_NLI(x, y, z):
    return np.mean(H(x) * H(y) * H(z), axis=-1) + np.mean(
        H_tilde(x) * H_tilde(y) * H_tilde(z), axis=-1
    )


def compute_thr_NLI(x, y, z, thr):
    return np.mean(heaviside(x, thr) * heaviside(y, thr) * heaviside(z, 0.0), axis=-1)


# NLI
nli = np.zeros((Zcoh.shape[0], Zcoh.shape[1], Zcoh.shape[2]))

dpower = power.diff("times")
dcoh = coh.diff("times")

p = 0
for s, t in tqdm(zip(sources, targets)):
    nli[p] = compute_NLI(dpower[s], dpower[t], dcoh[p])
    p = p + 1

nli = xr.DataArray(
    nli,
    dims=("roi", "freqs", "trials"),
    name="nli",
    coords={
        "roi": net.super_tensor.roi.data,
        "freqs": net.super_tensor.freqs.data,
        "trials": net.super_tensor.trials.data,
    },
)

# tNLI
thresholds = power.quantile(0.70, ("roi", "trials", "times"))

nli_thr = np.zeros((Zcoh.shape[0], Zcoh.shape[1], Zcoh.shape[2]))

p = 0
for s, t in tqdm(zip(sources, targets)):
    nli_thr[p] = compute_thr_NLI(power[s], power[t], coh[p], thresholds)
    p = p + 1

nli_thr = xr.DataArray(
    nli_thr,
    dims=("roi", "freqs", "trials"),
    name="nli",
    coords={
        "roi": net.super_tensor.roi.data,
        "freqs": net.super_tensor.freqs.data,
        "trials": net.super_tensor.trials.data,
    },
)

# Create data frame with the data
mean_power = power.mean(("trials", "times"))
mean_coh = coh.mean(("trials", "times"))

nli.to_netcdf(os.path.join(_ROOT, "Results", monkey,
                           f"nli/nli_{metric}_{sessions[idx]}.nc")
              )
nli_thr.to_netcdf(os.path.join(_ROOT, "Results", monkey,
                               f"nli/nli_thr_{metric}_{sessions[idx]}.nc")
                  )
mean_power.to_netcdf(os.path.join(_ROOT, "Results", monkey,
                                  f"nli/mean_power_{metric}_{sessions[idx]}.nc")
                     )
mean_coh.to_netcdf(os.path.join(_ROOT, "Results", monkey,
                                f"nli/mean_coh_{metric}_{sessions[idx]}.nc")
                   )
