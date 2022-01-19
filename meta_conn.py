"""
Computes meta-connectivity at roi level.
"""
import os
import brainconn as bc
import numpy as np
import xarray as xr
import argparse

from tqdm import tqdm
from GDa.net.layerwise import compute_network_partition
from GDa.temporal_network import temporal_network
from GDa.fc.mc import meta_conn
from GDa.util import _extract_roi
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
mask = net.s_mask
del net

MC = []
for f in range(coh.sizes["freqs"]):
    MC += [meta_conn(coh.isel(freqs=f), mask=mask, n_jobs=20)]

MC = xr.concat(MC, "freqs")
MC = MC.transpose("sources", "targets", "freqs", "trials", "times")
# Average over trials
MC_avg = MC.mean("trials")

###############################################################################
# Detecting communities in the MC matrix
###############################################################################


def MC_modularity(MC, n_jobs=1):
    """
    Modularity for the MC matrix with shape (roi, roi, trials, times)
    """
    partition, modularity = compute_network_partition(
        MC.data, backend="brainconn", n_jobs=n_jobs
    )
    # Set up dims and coords
    partition = partition.rename({"trials": "freqs"})
    modularity = modularity.rename({"trials": "freqs"})
    partition = partition.assign_coords(
        {"roi": MC.sources.data, "freqs": MC.freqs.data}
    )
    modularity = modularity.assign_coords({"freqs": MC.freqs.data})
    return partition, modularity


partition, modularity = [], []
for i in tqdm(range(10)):
    p, m = MC_modularity(MC.mean("trials"), n_jobs=20)
    partition += [p]
    modularity += [m]

partition = xr.concat(partition, "seeds")
modularity = xr.concat(modularity, "seeds")

###############################################################################
# Robust detection of communities
###############################################################################


def _allegiance(
    aff,
):
    """
    Compute the allegiance matrix based on a series of affiliation vectors.

    Parameters
    ----------
    aff: array_like
        Affiliation vector with shape (roi,observations).
        Observations can either be the affiliation vector for different time
        steps for a temporal network or for different realiaztions of
        stochastic comm. detection algorithms (e.g., Louvain or Leiden).

    Returns
    -------
    T: array_like
        The allegiance matrix with shape (roi,roi)
    """

    assert isinstance(aff, np.ndarray)

    # Number of nodes
    nC = aff.shape[0]
    # Number of observations
    nt = aff.shape[1]

    # Function to be applied to a single observation of the affiliation vector
    def _for_frame(t):
        # Allegiance for a frame
        T = np.zeros((nC, nC))
        # Affiliation vector
        av = aff[:, t]
        # For now convert affiliation vector to igraph format
        n_comm = int(av.max() + 1)
        for j in range(n_comm):
            p_lst = np.arange(nC, dtype=int)[av == j]
            grid = np.meshgrid(p_lst, p_lst)
            grid = np.reshape(grid, (2, len(p_lst) ** 2)).T
            T[grid[:, 0], grid[:, 1]] = 1
        np.fill_diagonal(T, 1)
        return T

    # Compute the single trial coherence
    T = [_for_frame(t) for t in range(nt)]
    T = np.nanmean(T, 0)

    return T


rois, freqs, times = (partition.roi.data,
                      partition.freqs.data, partition.times.data)
n_rois, n_freqs, n_times = len(rois), len(freqs), len(times)
# Compute the affiliation vector over the allegiance
aff = np.zeros((n_rois, n_freqs, n_times))
Q = np.zeros((n_freqs, n_times))

# Stack partition
p = partition.transpose("roi", "seeds", "freqs", "times").data

for f in range(n_freqs):
    for t in range(n_times):
        T = _allegiance(p[..., f, t])
        aff[:, f, t], Q[f, t] = bc.modularity.modularity_finetune_und(
            T, seed=500)

aff = xr.DataArray(
    aff,
    dims=("roi", "freqs", "times"),
    coords={"roi": rois, "freqs": freqs},
)

Q = xr.DataArray(Q, dims=("freqs", "times"), coords={"freqs": freqs})

###############################################################################
# Compute trimmer strengths
###############################################################################


def compute_trimer_st(MC, x_s, x_t, av=None):

    # Get number of rois based on source/targets arrays
    n_rois = np.concatenate((x_s, x_t)).astype(int).max() + 1

    # If a affiliation vector is passed compute trimer-strengths
    # per module
    if av is not None:
        # Number of modules
        n_mods = av.max()
        ts = np.zeros((n_mods, n_rois))
        for m in range(1, av.max() + 1):
            # Get indexes of nodes inside module m
            i_m = av == m
            sources, targets = x_s[i_m], x_t[i_m]
            for i in range(n_rois):
                idx = np.logical_or(sources == i, targets == i)
                ts[m - 1, i] = MC[np.ix_(idx, idx)].sum()
    # Otherwise compute for each roi
    else:
        ts = np.zeros(n_rois)
        for i in range(n_rois):
            idx = np.logical_or(x_s == i, x_t == i)
            ts[i] = MC[np.ix_(idx, idx)].sum()
    return ts


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


x_s, x_t = _extract_roi(MC.sources.data, "-")
mapping = creat_roi_mapping(x_s, x_t)
x_s, x_t = area2idx(x_s, mapping), area2idx(x_t, mapping)
n_rois = np.concatenate((x_s, x_t)).max() + 1
# Maximum number of communities
max_n_comm = int(aff.max())
Tst = np.ones((n_rois, max_n_comm, n_freqs, n_times)) * np.nan
for f in tqdm(range(n_freqs)):
    for t in range(n_times):
        A = MC_avg.isel(freqs=f, times=t).data
        av = aff.isel(freqs=f, times=t).data
        n_comm = int(av.max())
        Tst[:, :n_comm, f, t] = compute_trimer_st(
            A, x_s, x_t, av=av.astype(int)).T
Tst = xr.DataArray(
    Tst, dims=("roi", "comm", "freqs", "times"),
    coords=dict(roi=list(mapping.keys()))
)

###############################################################################
# Saving data
###############################################################################
_PATH = os.path.expanduser(
    "~/storage1/projects/GrayData-Analysis/Results/lucy/meta_conn")

# Save MC
MC_avg.to_netcdf(os.path.join(_PATH, f"MC_avg_{session}.nc"))
