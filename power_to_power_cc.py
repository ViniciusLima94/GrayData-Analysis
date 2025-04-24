"""
Edge-based encoding analysis done on the coherence dFC
"""

import os
import argparse

import numpy as np
import xarray as xr
from tqdm import tqdm
import numba as nb
from config import get_dates
from GDa.util import _extract_roi
from GDa.signal.surrogates import trial_swap_surrogates


###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to use", type=int)
parser.add_argument("TT", help="type of the trial", type=int)
parser.add_argument("BR", help="behavioral response", type=int)
parser.add_argument("SURR", help="wheter to compute surrogate or not", type=int)
parser.add_argument("ALIGN", help="wheter to align data to cue or match", type=str)
parser.add_argument("MONKEY", help="which monkey to use", type=str)

args = parser.parse_args()

# Index of the session to be load
sidx = args.SIDX
tt = args.TT
br = args.BR
at = args.ALIGN
surr = args.SURR
monkey = args.MONKEY

# early_cue, early_delay = return_delay_split(monkey=monkey, delay_type=0)

sessions = get_dates(monkey)

seed = 18320137
session = sessions[sidx]
##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser("~/funcog/gda")

##############################################################################
# Utility function
#############################################################################


def load_session_power(s_id, z_score=False, avg=0, roi=None, decim=1):
    _FILE_NAME = f"power_tt_{tt}_br_{br}_at_{at}_decim_1_hilbert.nc"
    path_pow = os.path.join(_ROOT, f"Results/{monkey}/{s_id}/session01", _FILE_NAME)
    power = xr.load_dataarray(path_pow)
    if z_score:
        power.values = (power - power.mean("times")) / power.std("times")
    # Averages power for each period (baseline, cue, delay, match) if needed
    # out = average_stages(power, avg, early_cue=early_cue, early_delay=early_delay)

    # if isinstance(roi, str):
    #    out = out.sel(roi=roi)

    trials, stim = power.trials.data, power.stim

    return power[..., ::decim], trials, stim


def power_correlations(power, verbose=False):
    """
    Given z-scored power compute PEC.
    """

    trials, roi, freqs, times = (
        power.trials.data,
        power.roi.data,
        power.freqs.data,
        power.times.data,
    )
    attrs = power.attrs

    n_rois = len(roi)

    data = power.data

    pairs = np.stack(np.triu_indices(n_rois, k=1), -1)
    roi_s, roi_t = roi[pairs[:, 0]], roi[pairs[:, 1]]
    edges = [f"{s}-{t}" for s, t in zip(roi_s, roi_t)]
    attrs["sources"], attrs["targets"] = pairs[:, 0], pairs[:, 1]

    @nb.jit(nopython=True)
    def _per_edge(i, j):
        cc = data[:, i, :, :] * data[:, j, :, :]
        return cc

    _iter = pairs
    if verbose:
        _iter = tqdm(_iter)

    cc = np.stack([_per_edge(i, j) for i, j in _iter], 1)

    cc = xr.DataArray(
        cc.astype(np.float32),
        dims=("trials", "roi", "freqs", "times"),
        coords=(trials, edges, freqs, times),
        attrs=attrs,
    )

    return cc


def convert_to_degree(cc):
    roi_s, roi_t = _extract_roi(cc.roi.data, "-")
    unique_rois = np.unique(np.hstack((roi_s, roi_t)))
    dd = []
    for roi in unique_rois:
        idx_s = (roi_s == roi).astype(int)
        idx_t = (roi_t == roi).astype(int)
        idx = (idx_s + idx_t) == 1
        # idx = np.logical_or(roi_s == roi, roi_t == roi)
        dd += [cc.isel(roi=idx).sum("roi")]
    dd = xr.concat(dd, "roi").assign_coords({"roi": unique_rois})
    return dd


def convert_to_mat(cc):
    """
    Convert CC in stream form to matrix.
    """
    n_trials, _, n_freqs, n_times = cc.shape

    # Get sources and targets
    sources, targets = cc.attrs["sources"], cc.attrs["targets"]
    # Get rois
    rois = cc.roi.data.astype(str)
    # Get pairs of rois
    rois_sep = np.asarray([roi.split("-") for roi in rois])
    roi_s, roi_t = rois_sep[:, 0], rois_sep[:, 1]
    # Index to area name
    idx2roi = dict(
        zip(
            np.hstack((cc.attrs["sources"], cc.attrs["targets"])),
            np.hstack((roi_s, roi_t)),
        )
    )

    n_roi = len(idx2roi)
    areas = [idx2roi[key] for key in range(n_roi)]

    cc_mat = np.zeros((n_trials, n_roi, n_roi, n_freqs, n_times))

    for p, (s, t) in enumerate(zip(sources, targets)):
        cc_mat[:, s, t, :, :] = cc_mat[:, t, s, :, :] = cc[:, p, :, :]

    cc_mat = xr.DataArray(
        cc_mat,
        dims=("trials", "sources", "targets", "freqs", "times"),
        coords=(cc.trials, areas, areas, cc.freqs, cc.times),
    )
    cc_mat.attrs = cc.attrs
    return cc_mat


if __name__ == "__main__":

    power, trials, stim = load_session_power(session, z_score=True, avg=0, roi=None, decim=10)
    if surr:
        power = trial_swap_surrogates(power, seed=seed, verbose=False)
    cc = power_correlations(power, verbose=True)
    attrs = cc.attrs

    # Average for each epoch
    t_match_on = (cc.attrs["t_match_on"] - cc.attrs["t_cue_on"]) / 1000

    out = []
    for i in range(cc.sizes["trials"]):
        stages = [
            [-0.5, -0.2],
            [0, 0.4],
            [0.5, 0.9],
            [0.9, 1.3],
            [t_match_on[i] - 0.4, t_match_on[i]],
        ]
        temp = []
        for t0, t1 in stages:
            temp += [cc.sel(times=slice(t0, t1)).isel(trials=i).mean("times")]
        out += [xr.concat(temp, "times")]
    out = xr.concat(out, "trials")
    cc = out.transpose("trials", "roi", "freqs", "times")
    cc.attrs = attrs
    print(cc.shape)

    # cc_mat = [convert_to_mat(cc.sel(freqs=[freq])) for freq in cc.freqs.values]
    # cc_mat.attrs = attrs

    cc.to_netcdf(os.path.join(_ROOT, "Results", monkey, "pec", f"pec_mat_{session}.nc"))
