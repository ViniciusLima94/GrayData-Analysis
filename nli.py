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
import pandas as pd
import argparse

from GDa.temporal_network import temporal_network
from config import sessions

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to run",
                    type=int)
args = parser.parse_args()
# The index of the session to use
idx = args.SIDX

###############################################################################
# Loading power and temporal network
###############################################################################
_ROOT = os.path.expanduser("~/storage1/projects/GrayData-Analysis")
_RESULTS = os.path.join("Results", "lucy", sessions[idx], "session01")

power_file = "power_tt_1_br_1_at_cue.nc"
coh_sig_file = "coh_k_0.3_multitaper_at_cue_surr.nc"
wt = None

power = xr.load_dataarray(os.path.join(_ROOT, _RESULTS, power_file))
power = power.transpose("roi", "freqs", "trials", "times")

net = temporal_network(
    coh_file="coh_k_0.3_multitaper_at_cue.nc",
    coh_sig_file=coh_sig_file,
    wt=wt,
    date=sessions[idx],
    trial_type=[1],
    behavioral_response=[1],
)

# Concatenate trials and times
power = power.stack(samples=("trials", "times"))
coh = net.super_tensor.stack(samples=("trials", "times"))

# z-score
Zpower = (power - power.mean("samples")) / power.std("samples")
Zcoh = (coh - coh.mean("samples")) / coh.std("samples")

roi = Zcoh.roi.data
freqs = Zcoh.freqs.data
n_rois = len(roi)
n_freqs = len(freqs)

###############################################################################
# Compute NLI
###############################################################################


@nb.vectorize([nb.float64(nb.float64)])
def H(x):
    if x >= 0:
        return x
    else:
        return 0


@nb.vectorize([nb.float64(nb.float64)])
def H_tilde(x):
    if x <= 0:
        return -x
    else:
        return 0


sources = net.super_tensor.attrs["sources"].astype(int)
targets = net.super_tensor.attrs["targets"].astype(int)
areas = np.asarray(net.super_tensor.attrs["areas"])

nli = np.zeros((Zcoh.shape[0], Zcoh.shape[1]))
for p, (s, t) in enumerate(zip(sources, targets)):
    nli[p] = np.mean(H(Zpower[s]) * H(Zpower[t]) * H(Zcoh[p]), -1) + np.mean(
        H_tilde(Zpower[s]) * H_tilde(Zpower[t]) * H_tilde(Zcoh[p]), -1
    )
nli = xr.DataArray(
    nli,
    dims=("roi", "freqs"),
    name="nli",
    coords={"roi": net.super_tensor.roi.data,
            "freqs": net.super_tensor.freqs.data}
)

# Create data frame with the data
mean_power = power.mean("samples")
mean_coh = coh.mean("samples")

###############################################################################
# DataFrame for NLI
###############################################################################
out = []
for f in range(n_freqs):
    data = np.array([areas[sources],
                     areas[targets],
                     sources,
                     targets,
                     [freqs[f]]*n_rois,
                     nli.isel(freqs=f),
                     mean_coh.isel(freqs=f)])
    out += [pd.DataFrame(
        data=data.T,
        columns=["roi_s", "roi_t", "s", "t", "f", "nli", "coh_st"],
    )]

out = pd.concat(out, axis=0)
out.to_csv(f'Results/lucy/nli/nli_{sessions[idx]}.csv')

###############################################################################
# DataFrame for mean power
###############################################################################
mean_power.to_dataframe(name="power").reset_index().to_csv(
    f'Results/lucy/nli/mean_power_{sessions[idx]}.csv')