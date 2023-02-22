import os
import numpy as np
import numba as nb
import xarray as xr
import argparse

from config import get_dates, return_delay_split
from frites.dataset import DatasetEphy
from frites.estimator import GCMIEstimator
from frites.workflow import WfMi
from GDa.util import average_stages
from GDa.session import session_info
from frites.utils import parallel_func
from tqdm import tqdm

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("TT",   help="type of the trial",
                    type=int)
parser.add_argument("BR",   help="behavioral response",
                    type=int)
parser.add_argument("ALIGN", help="wheter to align data to cue or match",
                    type=str)
parser.add_argument("AVERAGED", help="wheter to analyse the avg. power or not",
                    type=int)
parser.add_argument("MONKEY", help="which monkey to use",
                    type=str)
parser.add_argument("SLVR", help="Whether to use SLVR channels or not",
                    type=str)
parser.add_argument("THR", help="Threshold used to compute co-crackle mat.",
                    type=float)

args = parser.parse_args()

# Index of the session to be load
tt = args.TT
br = args.BR
at = args.ALIGN
avg = args.AVERAGED
monkey = args.MONKEY
slvr = args.SLVR
q = args.THR

stages = {}
stages["lucy"] = [[-0.4, 0], [0, 0.4], [0.5, 0.9], [0.9, 1.3], [1.1, 1.5]]
stage_labels = ["P", "S", "D1", "D2", "Dm"]

sessions = get_dates(monkey)

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser('~/funcog/gda')


##############################################################################
# utility functions 
###############################################################################
@nb.jit(nopython=True)
def co_crackling_adj(A):
    """
    Calculate the co-crackling matrix for a given matrix adjacency.

    Parameters:
    A (ndarray): matrix of community assignments where rows are rois and
    columns are time points
    Returns:
    ndarray: 3-D co-crackling matrix, where the last dimension corresponds
    to time point.
    """
    nroi, nobs = A.shape

    out = np.zeros((nroi, nroi, nobs), dtype=np.int8)

    for T in range(nobs):
        idx = A[:, T]
        out[..., T] = np.outer(idx, idx)

    return out


def order_rois(rois, x_s, x_t):
    roi_low = np.asarray([r.lower() for r in rois])
    _xs, _xt = x_s.copy(), x_t.copy()
    x_s, x_t = [], []
    for s, t in zip(_xs, _xt):
        _pair = np.array([roi_low[s], roi_low[t]])
        if np.all(_pair == np.sort(_pair)):
            x_s.append(s)
            x_t.append(t)
        else:
            x_s.append(t)
            x_t.append(s)
    return np.asarray(x_s), np.asarray(x_t)


def co_crackle_mat(A, verbose=False, n_jobs=1):
    """
    Compute co-crackle matrix and consensus vectors for given input data.

    This function computes co-crackle matrix and consensus vectors for the
    given input data. It first computes co-crackling adjacency matrix and then
    computes the mean of the matrix over the observation dimension. Finally,
    it computes consensus vectors for each time step by calling the
    'return_consensus_vector' function.

    Parameters:
    A (xarray.DataArray): Input data.
    nruns (int, optional): Number of runs for consensus vector calculation.
    Default is 100.

    Returns:
    kij (xarray.DataArray): Co-crackle matrix.
    ci (xarray.DataArray): Consensus vector for each time step.
    """
    rois = A.roi.data
    freqs = A.freqs.data
    times = A.times.data
    trials = A.trials.data

    nroi, nfreqs, ntrials, ntimes = A.shape
    
    A = A.stack(obs=("trials", "times")).data

    def _for_freq(f):
        co_k = co_crackling_adj(A[:, f, :])
        return co_k

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_freq, n_jobs=n_jobs, verbose=False, total=nfreqs
    )
    # Compute surrogate for each frequency
    co_k = parallel(p_fun(f) for f in range(nfreqs))
    
    co_k = np.stack(co_k, axis=2).reshape((nroi, nroi, nfreqs,
                                           ntrials, ntimes))
    
    x_s, x_t = np.triu_indices(nroi, k=1)
    npairs = len(x_s)
    x_s, x_t = order_rois(rois, x_s, x_t)

    roi_s, roi_t = rois[x_s], rois[x_t]
    roi_st = np.asarray([f"{s}-{t}" for s, t in zip(roi_s, roi_t)])
    print(roi_st)
    
    co_k_stream = np.zeros((npairs, nfreqs, ntrials, ntimes), dtype=np.int8)

    for p, (i, j) in enumerate(zip(x_s, x_t)):
        co_k_stream[p, ...] = co_k[i, j, ...]
        
    co_k = xr.DataArray(
        co_k_stream,
        dims=("roi", "freqs", "trials", "times"),
        coords=(roi_st, freqs, trials, times),
    )

    return co_k

###############################################################################
# Iterate over all sessions and concatenate power
###############################################################################

sxx = []
stim = []
for s_id in tqdm(sessions):
    _FILE_NAME = f"power_tt_{tt}_br_{br}_at_{at}.nc"
    path_pow = \
        os.path.join(_ROOT,
                     f"Results/{monkey}/{s_id}/session01",
                     _FILE_NAME)
    power = xr.load_dataarray(path_pow)
    attrs = power.attrs

    # Remove SLVR channels
    # if not bool(slvr):
        # info = session_info(
            # raw_path=os.path.join(_ROOT, "GrayLab"),
            # monkey=monkey,
            # date=s_id,
            # session=1,
        # )
        # slvr_idx = info.recording_info["slvr"].astype(bool)
        # power = power.isel(roi=np.logical_not(slvr_idx))

    # Compute activation time-series
    thr = power.quantile(q, ("trials", "times"))
    # Binarized power
    power = (power >= thr).transpose("roi", "freqs", "trials", "times")

    # Computes co-crackling matrices
    co_k = co_crackle_mat(power, verbose=False, n_jobs=10)

    # Average epochs
    out = []
    for t0, t1 in stages[monkey]:
        out += [co_k.sel(times=slice(t0, t1)).mean("times")]
    out = xr.concat(out, "times")
    out = out.transpose("trials", "roi", "freqs", "times")
    out.attrs = attrs
    sxx += [out.isel(roi=[r]) for r in range(len(out['roi']))]
    stim += [out.attrs["stim"].astype(int)]*len(out['roi'])


##############################################################################
# MI Workflow
##############################################################################

# Convert to DatasetEphy
dt = DatasetEphy(sxx, y=stim, nb_min_suj=10,
                 times="times", roi="roi")

mi_type = 'cd'
inference = 'rfx'
kernel = None

if avg:
    mcp = "fdr"
else:
    mcp = "cluster"

mi_type = "cd"

estimator = GCMIEstimator(mi_type="cd", copnorm=True,
                          biascorrect=True, demeaned=False, tensor=True,
                          gpu=False, verbose=None)
wf = WfMi(mi_type, inference, verbose=True, kernel=kernel, estimator=estimator)

kw = dict(n_jobs=30, n_perm=200)
cluster_th = None

mi, pvalues = wf.fit(dt, mcp=mcp, cluster_th=cluster_th, **kw)

###############################################################################
# Saving results
###############################################################################

# Path to results folder
_RESULTS = os.path.join(_ROOT,
                        f"Results/{monkey}/mutual_information/power/")

# path_mi = os.path.join(_RESULTS,
                       # f"mi_pow_tt_{tt}_br_{br}_aligned_{at}_ds_{ds}_avg_{avg}_{mcp}.nc")
# path_tv = os.path.join(_RESULTS,
                       # f"tval_pow_{tt}_br_{br}_aligned_{at}_ds_{ds}_avg_{avg}_{mcp}.nc")
# path_pv = os.path.join(_RESULTS,
                       # f"pval_pow_{tt}_br_{br}_aligned_{at}_ds_{ds}_avg_{avg}_{mcp}.nc")

path_mi = os.path.join(_RESULTS,
                       f"mi_cok_tt_{tt}_br_{br}_aligned_{at}_avg_{avg}_{mcp}_slvr_{slvr}.nc")
path_tv = os.path.join(_RESULTS,
                       f"tval_cok_{tt}_br_{br}_aligned_{at}_avg_{avg}_{mcp}_slvr_{slvr}.nc")
path_pv = os.path.join(_RESULTS,
                       f"pval_cok_{tt}_br_{br}_aligned_{at}_avg_{avg}_{mcp}_slvr_{slvr}.nc")

mi.to_netcdf(path_mi)
wf.tvalues.to_netcdf(path_tv)
pvalues.to_netcdf(path_pv)
