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
    tensor (roi, roi, times).
    """

    # Get rois names
    roi_s, roi_t = _extract_roi(MC.sources.data, "-")
    # Create a mapping to track rois and indexes
    mapping = creat_roi_mapping(roi_s, roi_t)
    x_s, x_t = area2idx(roi_s, mapping), area2idx(roi_t, mapping)
    # Get number of rois based on source/targets arrays
    n_rois = int(np.hstack((x_s, x_t)).max() + 1)
    # Get number of times and trials
    n_times = MC.sizes["times"]
    nt = n_times 

    # Stack trials and times
    A = MC.data

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
    Tst = Tst.reshape((n_rois, n_times))

    Tst = xr.DataArray(Tst,
                       dims=("roi", "times"),
                       coords={
                           "roi": list(mapping.keys()),
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
Tst = xr.concat(Tst, "freqs").transpose("roi", "freqs", "times")
Tst.attrs = MC.attrs


###############################################################################
# Saving data
###############################################################################
_PATH = os.path.expanduser(os.path.join(_ROOT,
                                        "Results/lucy/meta_conn"))

# Save MC
MC.to_netcdf(os.path.join(_PATH, f"MC_{session}.nc"))
# Save trimmer strengths
Tst.to_netcdf(os.path.join(_PATH, f"tst_{session}.nc"))
