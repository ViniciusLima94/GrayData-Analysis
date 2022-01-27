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

###############################################################################
# Argument parsing
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to run",
                    type=int)
args = parser.parse_args()
# The index of the session to use
idx = args.SIDX
session = sessions[idx]

###############################################################################
# Loading temporal network
###############################################################################
_ROOT = os.path.expanduser("~/storage1/projects/GrayData-Analysis")

# Path in which to save coherence data
_RESULTS = os.path.join("Results", "lucy", session, "session01")

coh_sig_file = "coh_k_0.3_multitaper_at_cue_surr.nc"
wt = None

net = temporal_network(
    coh_file="coh_k_0.3_multitaper_at_cue.nc",
    coh_sig_file=coh_sig_file,
    wt=wt,
    date=session,
    trial_type=[1],
    behavioral_response=[1],
)

###############################################################################
# Compute meta-connectivity
###############################################################################
# Masks for each stage
net.create_stage_masks()
# Average same rois
coh = net.super_tensor.groupby("roi").mean("roi")
coh.attrs = net.super_tensor.attrs
mask = net.s_mask
del net

MC = []
for f in range(coh.sizes["freqs"]):
    MC += [meta_conn(coh.isel(freqs=f), mask=mask, n_jobs=10)]

MC = xr.concat(MC, "freqs")
MC = MC.transpose("sources", "targets", "freqs", "trials", "times")
MC = MC.assign_coords({"freqs": coh.freqs.data})

# Numerical roi
# n_roi = np.core.defchararray.add(
    # np.core.defchararray.add(coh.attrs["sources"].astype(str),
                             # ["-"] * 4371),
    # coh.attrs["targets"].astype(str)
# )
# MC = MC.assign_coords({"sources": n_roi})
# MC = MC.assign_coords({"targets": n_roi})

###############################################################################
# Compute trimmer strengths
###############################################################################


def compute_trimer_st(MC, x_s, x_t):
    # Get number of rois based on source/targets arrays
    n_rois = int(np.hstack((x_s, x_t)).max() + 1)
    ts = np.zeros(n_rois)
    for i in range(n_rois):
        idx = np.logical_or(x_s == i, x_t == i)
        ts[i] = MC[np.ix_(idx, idx)].sum()
    return ts


def trimmer_strength(MC, n_jobs=1, verbose=False):
    """
    Compute trimmer strengths for meta connectivity
    tensor (roi, roi, trials, times).
    """

    # Get rois names
    roi_s, roi_t = _extract_roi(MC.sources.data, "-")
    # Create a mapping to track rois and indexes
    mapping = creat_roi_mapping(roi_s, roi_t)
    x_s, x_t = area2idx(roi_s, mapping), area2idx(roi_t, mapping)
    # Get number of rois based on source/targets arrays
    n_rois = int(np.hstack((x_s, x_t)).max() + 1)
    # Get number of times and trials
    n_times, n_trials = MC.sizes["times"], MC.sizes["trials"]
    nt = n_times * n_trials

    # Stack trials and times
    A = MC.stack(z=("trials", "times")).data

    # Compute for a single observation
    def _for_frame(t):
        # Call core function
        Tst = compute_trimer_st(A[..., t], x_s, x_t)
        return Tst

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_frame, n_jobs=n_jobs, verbose=verbose,
        total=nt)
    # Compute the single trial coherence
    Tst = parallel(p_fun(t) for t in range(nt))
    # Convert to numpy array
    Tst = np.asarray(Tst)

    # Unstack trials and time
    Tst = Tst.reshape((n_rois, n_trials, n_times))

    Tst = xr.DataArray(Tst,
                       dims=("roi", "trials", "times"),
                       coords={
                           "roi": list(mapping.keys()),
                           "trials": MC.trials.data,
                           "times": MC.times.data,
                       })
    return Tst


def creat_roi_mapping(x_s, x_t):
    """
    Create hash table to access rois names
    x_s: array_like
        Source areas names
    x_t: array_like
        Target areas names
    """
    # Get unique names
    areas = np.sort(np.unique(np.hstack((x_s, x_t))))
    # Number of areas
    n_areas = len(areas)
    # Numerical index of each area
    idx = np.arange(0, n_areas, 1, dtype=np.int32)
    # Create mapping
    mapping = dict(zip(areas, idx))
    return mapping


def area2idx(areas, mapping):
    """
    Given a list with names of areas and a mapping
    return the indexes correponding to the areas.
    """
    return np.asarray([mapping[a] for a in areas])


Tst = []
for f in range(MC.sizes["freqs"]):
    Tst += [trimmer_strength(MC.isel(freqs=f), n_jobs=20)]
Tst = xr.concat(Tst, "freqs").transpose("roi", "freqs", "trials", "times")
Tst.attrs = coh.attrs


###############################################################################
# Saving data
###############################################################################
_PATH = os.path.expanduser(
    "~/storage1/projects/GrayData-Analysis/Results/lucy/meta_conn")

# Save MC
# MC.to_netcdf(os.path.join(_PATH, f"MC_{session}.nc"))
# MC.mean("trials").to_netcdf(os.path.join(_PATH, f"MC_avg_{session}.nc"))
# Save trimmer strengths
Tst.to_netcdf(os.path.join(_PATH, f"tst_{session}.nc"))
