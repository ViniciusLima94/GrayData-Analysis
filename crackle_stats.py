import os
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from brainconn.centrality import participation_coef
from brainconn.modularity import modularity_louvain_dir, modularity_louvain_und
from frites.conn.conn_tf import _create_kernel, _smooth_spectra
from frites.stats import confidence_interval
from frites.utils import parallel_func
from scipy.stats import ks_2samp, mannwhitneyu, ttest_1samp
from statannot import add_stat_annotation
from tqdm import tqdm

from config import get_dates, return_evt_dt
from GDa.graphics import plot
from GDa.loader import loader
from GDa.stats.bursting import find_activation_sequences, find_start_end
from GDa.temporal_network import temporal_network
from GDa.util import _extract_roi, get_areas


##############################################################################
# ARGUMENT PARSING
##############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SESSION", help="which session to load",
                    type=int)
parser.add_argument("MONKEY", help="which monkey to use",
                    type=str)
parser.add_argument("THR", help="which threshold to use",
                    type=float)

args = parser.parse_args()

# Index of the session to be load
sid = args.SESSION
monkey = args.MONKEY
thr = args.THR

_ROOT = os.path.expanduser("~/funcog/gda/")
_RESULTS = os.path.expanduser(f"~/funcog/gda/Results/{monkey}/crk_stats")
stages = [[-0.4, 0], [0, 0.4], [0.5, 0.9], [0.9, 1.3], [1.1, 1.5]]

stage_labels = ["P", "S", "D1", "D2", "Dm"]

sessions = get_dates(monkey)
session = sessions[sid]

data_loader = loader(_ROOT=_ROOT)

##############################################################################
# AUXILIAR FUNCTIONS
##############################################################################
def return_consensus_vector(A, fmod=modularity_louvain_und, nruns=None, verbose=False):
    """
    Return the consensus vector for a given set of community assignments.

    This function takes a matrix of community assignments (A), a modularity function (fmod), and the number of runs (nruns)
    to return the consensus vector of the assignments. The consensus vector is calculated by averaging the affiliation vectors of each run.

    Parameters:
    A (ndarray or xr.DataArray): matrix of community assignments.
    fmod (callable): modularity function used to calculate community assignments. Default is louvain method.
    nruns (int): number of runs to calculate consensus vector over. Default is None.
    Returns:
    Tuple: (idx, q) where idx is the community assignments and q is the modularity score
    """
    nC = len(A)

    if isinstance(A, xr.DataArray):
        A = A.data

    # Creates consensus vector for different runs
    av = []
    for i in tqdm(range(nruns)) if verbose else range(nruns):
        idx, _ = fmod(A, seed=i * 1000)
        av += [idx]
    av = np.stack(av)

    def _for_frame(av):
        # Allegiance for a frame
        T = np.zeros((nC, nC))
        # Affiliation vector
        # For now convert affiliation vector to igraph format
        n_comm = int(av.max() + 1)
        for j in range(n_comm):
            p_lst = np.arange(nC, dtype=int)[av == j]
            grid = np.meshgrid(p_lst, p_lst)
            grid = np.reshape(grid, (2, len(p_lst) ** 2)).T
            T[grid[:, 0], grid[:, 1]] = 1
        np.fill_diagonal(T, 0)
        return T

    T = []
    for i in range(nruns):
        T += [_for_frame(av[i])]

    T = np.stack(T).mean(0)

    return modularity_louvain_und(T)


@nb.jit(nopython=True)
def co_crackling_adj(A):
    """
    Calculate the co-crackling matrix for a given matrix adjacency.

    Parameters:
    A (ndarray): matrix of community assignments where rows are rois and columns are time points
    Returns:
    ndarray: 3-D co-crackling matrix, where the last dimension corresponds to time point.
    """
    nroi, nobs = A.shape

    out = np.zeros((nroi, nroi, nobs))

    for T in range(nobs):
        idx = A[:, T]
        out[..., T] = np.outer(idx, idx)

    return out


def create_typeI_surr(A, seed=0, n_jobs=1, verbose=False):
    """
    create_typeI_surr - function to create type I surrogate data by shuffling the timepoints across trials for each ROI

    Input:
    A (xarray.DataArray) - original data
    seed (int) - seed for random number generator, default = 0
    verbose (bool) - flag to display progress bar, default = False

    Output:
    A_surrI (xarray.DataArray) - type I surrogate data with shuffled timepoints across trials for each ROI
    """

    nrois, nfreqs, ntrials, ntimes = A.shape
    dims, coords = A.dims, A.coords

    A = A.stack(obs=("trials", "times"))

    def _loop_rois(A_np):
        # Store surrogate
        A_surrI = A_np.copy()
        # Vector with trial indexes
        trials_idx = np.arange(0, ntimes * ntrials, 1, dtype=np.int8)
        # Loop over areas
        for i in range(nrois):
            idx_ = np.random.choice(trials_idx, ntimes * ntrials, replace=True)
            A_surrI[i, ...] = A_np[i, idx_]

        return A_surrI

    def _for_freq(f):
        np.random.seed(seed + f * 1000)
        return _loop_rois(A.data[:, f, :])

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_freq, n_jobs=n_jobs, verbose=verbose, total=nfreqs
    )
    # Compute surrogate for each frequency
    A_surrI = parallel(p_fun(f) for f in range(nfreqs))
    A_surrI = np.stack(A_surrI, axis=1).reshape((nrois, nfreqs, ntrials, ntimes))
    A_surrI = xr.DataArray(A_surrI, dims=dims, coords=coords)

    return A_surrI.unstack()


def create_typeII_surr(A, seed=0, verbose=False, n_jobs=1):
    """
    create_typeII_surr - function to create type II surrogate data by shuffling the timepoints within each trial for each ROI

    Input:
    A (xarray.DataArray) - original data
    seed (int) - seed for random number generator, default = 0
    verbose (bool) - flag to display progress bar, default = False

    Output:
    A_surrII (xarray.DataArray) - type II surrogate data with shuffled timepoints within each trial for each ROI
    """

    nrois, nfreqs, ntrials, ntimes = A.shape
    dims, coords = A.dims, A.coords

    def _for_freq(f):

        np.random.seed(seed + f * 1000)

        A_surrII = A.data[:, f, :, :].copy()
        for i in range(nrois):
            for T in range(ntrials):
                idx_ = np.random.choice(range(ntimes), ntimes, replace=False)
                A_surrII[i, T, :] = A.data[i, f, T, idx_]

        return A_surrII

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_freq, n_jobs=n_jobs, verbose=verbose, total=nfreqs
    )
    # Compute surrogate for each frequency
    A_surrII = parallel(p_fun(f) for f in range(nfreqs))
    A_surrII = np.stack(A_surrII, axis=1).reshape((nrois, nfreqs, ntrials, ntimes))
    A_surrII = xr.DataArray(A_surrII, dims=dims, coords=coords)

    return A_surrII


def compute_crackle_duration(
    power, q=0.9, twin=None, surrogate=False, n_boots=200, n_jobs=1, verbose=False
):

    nrois, nfreqs, ntrials, ntimes = power.shape

    rois, freqs, trials, times = (
        power.roi.data,
        power.freqs.data,
        power.trials.data,
        power.times.data,
    )

    dt = times[1] - times[0]

    attrs = power.attrs
    thr = power.quantile(q, ("trials", "times"))
    A = (power >= thr).copy()

    if surrogate:
        A = create_typeI_surr(A, seed=0, verbose=verbose, n_jobs=n_jobs)

    A_ = [
        A.sel(times=slice(ti, tf)).stack(obs=("trials", "times")).data
        for ti, tf in twin
    ]

    del A

    nwin = len(twin)

    def _for_freq(f):
        delta_ci = np.zeros((2, nrois, nwin))
        for t in range(nwin):
            for i in range(nrois):
                delta_ci[:, i, t] = confidence_interval(
                    find_activation_sequences(A_[t][i, f], dt=dt),
                    axis=0,
                    n_boots=n_boots,
                    verbose=False,
                ).squeeze()
        return delta_ci

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_freq, n_jobs=n_jobs, verbose=verbose, total=nfreqs
    )
    # Compute surrogate for each frequency
    delta_ci = parallel(p_fun(f) for f in range(nfreqs))
    delta_ci = np.stack(delta_ci, axis=2)

    delta_ci = xr.DataArray(
        delta_ci,
        dims=("bound", "roi", "freqs", "times"),
        coords=dict(roi=rois, freqs=freqs),
    )

    delta_ci.attrs = attrs

    return delta_ci


def return_co_crackle_mat(A, nruns=100, verbose=False):
    """
    Compute co-crackle matrix and consensus vectors for given input data.

    This function computes co-crackle matrix and consensus vectors for the given input data.
    It first computes co-crackling adjacency matrix and then computes the mean of the matrix over the observation dimension.
    Finally, it computes consensus vectors for each time step by calling the 'return_consensus_vector' function.

    Parameters:
    A (xarray.DataArray): Input data.
    nruns (int, optional): Number of runs for consensus vector calculation. Default is 100.

    Returns:
    kij (xarray.DataArray): Co-crackle matrix.
    ci (xarray.DataArray): Consensus vector for each time step.
    """
    rois = A.roi.data
    times = A.times.data
    trials = A.trials.data

    nroi, ntrials, ntimes = A.shape

    co_k = []
    for ti, tf in tqdm(stages) if verbose else stages:
        co_k += [
            co_crackling_adj(
                A.sel(times=slice(ti, tf)).stack(obs=("trials", "times")).data
            )
        ]

    co_k = np.stack(co_k, axis=3)
    co_k = xr.DataArray(
        co_k,
        dims=("sources", "targets", "obs", "times"),
        coords=dict(sources=A.roi.data, targets=A.roi.data),
    )

    kij = co_k.mean("obs")
    ci = []
    # for i in range(kij.sizes["times"]):
        # ci_temp, _ = return_consensus_vector(kij.sel(times=i), nruns=nruns)
        # ci += [ci_temp]
    # ci = np.stack(ci, axis=1)
    # ci = xr.DataArray(ci, dims=("roi", "times"), coords=dict(roi=rois))

    return kij, ci
##############################################################################
# CRACKLE DURATION
##############################################################################
kw_loader = dict(
    session=session, aligned_at="cue", channel_numbers=False, monkey=monkey
)

power_task = data_loader.load_power(**kw_loader, trial_type=1, behavioral_response=1)
power_fix = data_loader.load_power(**kw_loader, trial_type=2, behavioral_response=0)

thr_task = power_task.quantile(thr, ("trials", "times"))
thr_fix = power_fix.quantile(thr, ("trials", "times"))

A_task = power_task >= thr_task
A_fix = power_fix >= thr_fix

A_surrI_T = create_typeI_surr(A_task, seed=0, verbose=True, n_jobs=1)
A_surrII_T = create_typeII_surr(A_task, seed=0, verbose=True, n_jobs=10)

qs = np.arange(0.5, 1, 0.1)

delta_ci_task = [
    compute_crackle_duration(
        power_task, q=q, twin=stages, n_boots=200, n_jobs=10, verbose=True
    )
    for q in qs
]

delta_ci_surr = [
    compute_crackle_duration(
        power_task,
        q=q,
        twin=stages,
        surrogate=True,
        n_boots=200,
        n_jobs=10,
        verbose=True,
    )
    for q in qs
]

delta_ci_task = xr.concat(delta_ci_task, "q")
delta_ci_surr = xr.concat(delta_ci_surr, "q")

delta_ci_task.to_netcdf(os.path.join(_RESULTS,
                                     f"delta_ci_task_{session}.nc"))


##############################################################################
# CRACKLE DURATION
##############################################################################
kij_task, ci_task = [], []
kij_fix, ci_fix = [], []
for f in A_task.freqs.data:
    out_1, out_2 = return_co_crackle_mat(A_task.sel(freqs=f))
    kij_task += [out_1]
    ci_task += [out_2]
    out_1, out_2 = return_co_crackle_mat(A_fix.sel(freqs=f))
    kij_fix += [out_1]
    ci_fix += [out_2]

kij_task = xr.concat(kij_task, "freqs").assign_coords(dict(freqs=A_task.freqs))
kij_fix = xr.concat(kij_fix, "freqs").assign_coords(dict(freqs=A_task.freqs))

quantile = int(thr * 100)

kij_task.to_netcdf(os.path.join(_RESULTS,
                                f"kij_task_{session}_q_{quantile}.nc"))
kij_fix.to_netcdf(os.path.join(_RESULTS,
                                f"kij_fix_{session}_q_{quantile}.nc"))
