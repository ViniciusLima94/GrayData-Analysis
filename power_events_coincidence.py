"""
Fraction of co-crackles
"""

import os
import argparse

import numpy as np
import xarray as xr
from config import get_dates
from GDa.util import _extract_roi
from GDa.signal.surrogates import trial_swap_surrogates

from frites.conn.conn_utils import conn_links
from frites.utils import parallel_func


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
parser.add_argument("THR", help="threshold value to use", type=int)

args = parser.parse_args()

# Index of the session to be load
sidx = args.SIDX
tt = args.TT
br = args.BR
at = args.ALIGN
surr = args.SURR
monkey = args.MONKEY
thr = args.THR

sessions = get_dates(monkey)

session = sessions[sidx]


###############################################################################
# Functions
###############################################################################


def load_session_power(s_id, z_score=False, avg=0, roi=None, decim=1):
    _FILE_NAME = f"power_tt_{tt}_br_{br}_at_{at}_decim_1_hilbert.nc"
    path_pow = os.path.join(_ROOT, f"Results/{monkey}/{s_id}/session01", _FILE_NAME)
    power = xr.load_dataarray(path_pow)
    if z_score:
        power.values = (power - power.mean("times")) / power.std("times")

    trials, stim = power.trials.data, power.stim

    return power[..., ::decim], trials, stim


def convert_to_degree(cc):
    roi_s, roi_t = _extract_roi(cc.roi.data, "-")
    unique_rois = np.unique(np.hstack((roi_s, roi_t)))
    dd = []
    for roi in unique_rois:
        idx = np.logical_or(roi_s == roi, roi_t == roi)
        dd += [cc.isel(roi=idx).sum("roi")]
    dd = xr.concat(dd, "roi").assign_coords({"roi": unique_rois})
    return dd


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def _int(w, x_s, x_t, kw_para):

    # define the power envelope correlations
    def pairwise_int(w_x, w_y):
        # computes fraciton of events above threshold that intersect
        x = w[:, w_x, :, :]
        y = w[:, w_y, :, :]
        prod = x * y
        norm = np.max([x.sum(-1), y.sum(-1)], axis=0)
        norm = np.where(norm == 0, 1, norm)
        return prod

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(pairwise_int, **kw_para)

    # compute the single trial power envelope correlations
    return parallel(p_fun(s, t) for s, t in zip(x_s, x_t))


def power_events_coincidence(
    power, q_l, q_u=None, n_jobs=1, verbose=False, shuffle=False
):

    # Extract dimensions
    dims = power.dims
    trials, roi, freqs, times = (
        power.trials.data,
        power.roi.data,
        power.freqs.data,
        power.times.data,
    )
    ntrials, nroi, nfreqs, ntimes = power.shape

    roi_gp, _ = roi, np.arange(nroi).reshape(-1, 1)
    (x_s, x_t), roi_p = conn_links(roi_gp, {})
    n_pairs = len(x_s)

    quantiles = power.quantile(q_l, "times")

    z_power = (power >= quantiles).values

    if isinstance(q_u, float):
        quantiles = power.quantile(q_u, "times")
        z_power = np.logical_and(z_power, power < quantiles).values

    if shuffle:
        z_power = shuffle_along_axis(z_power, 0)

    kw_para = dict(n_jobs=n_jobs, verbose=verbose, total=n_pairs)

    pec = _int(z_power, x_s, x_t, kw_para)
    pec = np.stack(pec, axis=1)

    # conversion
    pec = xr.DataArray(
        pec,
        dims=dims,
        name="pec",
        coords={"trials": trials, "roi": roi_p, "freqs": freqs, "times": times},
    )

    return pec


##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser("~/funcog/gda")

if __name__ == "__main__":

    power, trials, stim = load_session_power(
        session, z_score=True, avg=0, roi=None, decim=10
    )

    if surr:
        power = trial_swap_surrogates(power, seed=654167, verbose=False)

    pec = power_events_coincidence(
        power, thr / 100, q_u=None, n_jobs=1, verbose=True, shuffle=False
    )

    attrs = power.attrs

    # Average for each epoch
    t_match_on = (attrs["t_match_on"] - attrs["t_cue_on"]) / 1000

    out = []
    for i in range(pec.sizes["trials"]):
        stages = [
            [-0.5, -0.2],
            [0, 0.4],
            [0.5, 0.9],
            [0.9, 1.3],
            [t_match_on[i] - 0.4, t_match_on[i]],
        ]
        temp = []
        for t0, t1 in stages:
            temp += [pec.sel(times=slice(t0, t1)).isel(trials=i).mean("times")]
        out += [xr.concat(temp, "times")]
    out = xr.concat(out, "trials")
    pec = out.transpose("trials", "roi", "freqs", "times")
    pec.attrs = attrs

    # Path in which to save coherence data
    results_path = os.path.join(_ROOT, "Results", monkey, session, "session01")
    # Create results path in case it does not exist
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    file_name = f"pec_{tt}_br_{br}_at_{at}_thr_{thr}.nc"
    path_pec = os.path.join(results_path, file_name)

    pec.to_netcdf(path_pec)
