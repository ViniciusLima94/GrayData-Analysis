""" Find temporal communities in the supraadjacency """
import os
import numpy as np
import xarray as xr
import networkx as nx
import igraph as ig
import pickle
import argparse
from brainconn.modularity import modularity_louvain_und
from frites.utils import parallel_func
from tqdm import tqdm
from GDa.loader import loader
from GDa.util import _extract_roi
from config import get_dates, freqs

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to be run", type=int)
parser.add_argument("THR", help="threshold to binarize power", type=int)
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument(
    "SURR", help="whether to compute for surrogate data or not", type=int
)
parser.add_argument("TTYPE", help="which trial type to use", choices=[1, 2], type=int)
parser.add_argument(
    "BEHAVIOR", help="which behavioral response to use", choices=[0, 1], type=int
)
parser.add_argument(
    "EPOCH",
    help="which stage to use",
    choices=["P", "S", "D1", "D2", "Dm", "all"],
    type=str,
)

args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX
thr = args.THR
monkey = args.MONKEY
surr = args.SURR
ttype = args.TTYPE
behav = args.BEHAVIOR
epoch = args.EPOCH

# To be sure of using the right parameter when using fixation trials
if ttype == 2:
    behav = 0

# Get session
sessions = get_dates(monkey)
s_id = sessions[idx]

# Root directory
_ROOT = os.path.expanduser("~/funcog/gda")

data_loader = loader(_ROOT=_ROOT)

stages = [[-0.5, -.1], [0, 0.4], [0.5, 0.9], [0.9, 1.3], [1.1, 1.5], [-0.5, 2.0]]
stage_labels = ["P", "S", "D1", "D2", "Dm", "all"]
stages = dict(zip(stage_labels, stages))


###############################################################################
# Utility functions
###############################################################################
def convert_to_supra_adj(raster, verbose=False):
    """
    Convert a raster matrix to a supra-adjacency matrix.

    Parameters
    ----------
    raster : ndarray
        Input raster matrix.
    verbose : bool, optional
        If True, display progress using tqdm. Defaults to False.

    Returns
    -------
    ndarray
        Supra-adjacency matrix.

    """

    raster = raster.astype(int)
    nroi, ntimes = raster.shape

    supraA = np.zeros((nroi * ntimes, nroi * ntimes), dtype=int)

    __iter = range(ntimes - 1)

    if verbose:
        __iter = tqdm(__iter)

    for t in __iter:
        raster_t = raster[:, t].data
        for x in range(nroi):
            for y in range(x + 1, nroi):
                cxx = raster_t[x] * raster_t[x]
                cyy = raster_t[y] * raster_t[y]
                cxy = raster_t[x] * raster_t[y]
                supraA[x + nroi * t, y + nroi * (t + 1)] = cxy
                supraA[y + nroi * t, x + nroi * (t + 1)] = cxy
                supraA[x + nroi * t, x + nroi * (t + 1)] = cxx
                supraA[y + nroi * t, y + nroi * (t + 1)] = cyy
                supraA[x + nroi * t, y + nroi * t] = cxy
                supraA[y + nroi * t, x + nroi * t] = cxy

    return supraA


def return_connected_components(raster, node_labels, min_size=1, verbose=False):
    """
    Return the connected components of a raster matrix.

    Args:
        raster (ndarray): Input raster matrix.
        node_labels (list): List of node labels.
        min_size (int, optional): Minimum size of connected components
                                  to consider. Defaults to 1.
        verbose (bool, optional): If True, display progress using tqdm.
                                  Defaults to False.

    Returns:
        list: List of connected components.

    Raises:
        None
    """

    supraA = convert_to_supra_adj(raster, verbose=verbose)

    # Convert to networkx graph
    G = nx.from_numpy_array(supraA)
    # Decompose graph
    dG = ig.Graph.from_networkx(G).components(mode="strong")
    # Number of connected components
    ncomp = len(dG)
    # Size of components
    sizes = np.array(dG.sizes())
    # Bigger than min_size components
    idx = np.where(np.array(dG.sizes()) > min_size)[0]
    # Components
    dG = list(dG)

    def _for_component(i):

        return node_labels[dG[i]]

    __iter = tqdm(idx) if verbose else idx

    return [_for_component(i) for i in __iter]


def parallel_wrapper(raster, node_labels, min_size=1, n_jobs=1, verbose=False):

    ntrials = raster.sizes["trials"]

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        return_connected_components, n_jobs=n_jobs, verbose=verbose, total=ntrials
    )

    # Compute the single trial coherence
    out = parallel(
        p_fun(raster[:, T, :], node_labels, min_size, False) for T in range(ntrials)
    )

    return out


def get_areas_times(avalanches, trials, stims):
    """
    Extract the areas and times from a list of avalanches.

    Args:
        avalanches (list): List of avalanches.

    Returns:
        tuple: Two lists containing the extracted areas and times, respectively.

    Raises:
        None
    """
    areas = []
    times = []
    trials_sel = []
    stims_sel = []
    pos = 0
    for avalanche in avalanches:
        for av in avalanche:
            t, a = _extract_roi(av, "_")
            if len(np.unique(a)) > 1:
                times += [t]
                areas += [a]
                trials_sel += [trials[pos]]
                stims_sel += [stims[pos]]
            pos = pos + 1

    trials_sel = np.hstack(trials_sel)
    stims_sel = np.hstack(stims_sel)

    return areas, times, trials_sel, stims_sel


def get_area_mapping(unique_areas):
    area2idx = dict(zip(unique_areas, range(len(unique_areas))))
    return area2idx


def get_coavalanche_matrix(areas, times):
    """
    Calculate the coavalanche matrix, precedence matrix, and delta values.

    Args:
        areas (list): List of areas.
        times (list): List of times.

    Returns:
        tuple: A tuple containing the coavalanche matrix, precedence matrix, and delta values.

    Raises:
        None
    """

    navalanches = len(areas)

    unique_areas = np.unique(np.hstack(areas))
    n_unique_areas = len(unique_areas)
    area2idx = get_area_mapping(unique_areas)

    T = np.zeros((n_unique_areas, n_unique_areas))
    P = np.zeros((n_unique_areas, n_unique_areas))

    delta = np.zeros(navalanches)

    for i in tqdm(range(navalanches)):
        ua = np.unique(areas[i])
        # Index of areas in the avalanche
        idx = [area2idx[area] for area in ua]
        # Time slice of each area
        tava = times[i].astype(int)
        # Avalanche duration
        delta[i] = tava.max() - tava.min()
        # Coactivation
        T[np.ix_(idx, idx)] += 1
        # Precedence
        min_times = []
        for a in ua:
            min_times += [times[i].astype(int)[areas[i] == a].min()]
        min_times = np.array(min_times)
        prec = np.array(min_times)[:, None] < np.array(min_times)
        P[np.ix_(idx, idx)] += prec.astype(int)

    np.fill_diagonal(T, 0)

    T = xr.DataArray(
        T / navalanches,
        dims=("sources", "targets"),
        coords=(unique_areas, unique_areas),
    )
    P = xr.DataArray(
        P / navalanches,
        dims=("sources", "targets"),
        coords=(unique_areas, unique_areas),
    )

    return T, P, delta


def z_score(data):
    return (data - data.mean("times")) / data.std("times")


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


if __name__ == "__main__":

    # Get time range of the epoch analysed
    ti, tf = stages[epoch]
    # For loop over frequency bands
    for freq in freqs.astype(int):
        ######################################################################
        # Loading data
        ######################################################################
        kw_loader = dict(
            session=s_id, aligned_at="cue", channel_numbers=False, monkey=monkey
        )

        power = data_loader.load_power(
            **kw_loader, trial_type=ttype, behavioral_response=behav
        ).sel(freqs=freq, times=slice(ti, tf))

        trials_array = power.trials.data

        if ttype == 1:
            stim = power.attrs["stim"]

        # z-score power
        power = z_score(power)

        # Binarize power
        raster = power >= thr

        # Get roi names
        rois = raster.roi.data

        # Get regions with time labels
        roi_time = []
        for t in range(power.sizes["times"]):
            roi_time += [f"{r}_{t}" for r in rois]
        roi_time = np.hstack(roi_time)

        ######################################################################
        # Compute temporal components
        ######################################################################

        if bool(surr):
            raster_shuffle = shuffle_along_axis(raster.data, 0)

            raster = xr.DataArray(
                raster_shuffle, dims=raster.dims, coords=raster.coords
            )

        avalanches = parallel_wrapper(
            raster, roi_time, min_size=1, n_jobs=30, verbose=False
        )

        # Get trials and stim label
        trials, stims = [], []
        for T in range(raster.sizes["trials"]):
            trials += [[trials_array[T]] * len(avalanches[T])]
            if ttype == 1:
                stims += [[stim[T]] * len(avalanches[T])]

        trials = np.hstack(trials)
        if ttype == 1:
            stims = np.hstack(stims)

        # Areas and times lists
        areas, times, trials, stims = get_areas_times(avalanches, trials,
                                                      stims)

        # Coavalanching and precedence
        T, P, delta = get_coavalanche_matrix(areas, times)

        ######################################################################
        # Save results
        ######################################################################

        _SAVE = os.path.expanduser(f"~/funcog/gda/Results/{monkey}/avalanches")

        names = ["areas", "trials", "times"]
        results = [areas, trials, times]

        if ttype == 1:
            names += ["stim"]
            results += [stims]

        def _fname(name):
            return f"{name}_tt_{ttype}_br_{behav}_{epoch}_{s_id}_freq_{freq}_thr_{thr}_surr_{surr}.pkl"

        for pos, data in enumerate(results):
            fname = _fname(names[pos])
            with open(os.path.join(_SAVE, fname), "wb") as fp:
                pickle.dump(data, fp)

        # Coavalanche and precedence
        fname = f"T_tt_{ttype}_br_{behav}_{epoch}_{s_id}_freq_{freq}_thr_{thr}_surr_{surr}.nc"
        T.to_netcdf(os.path.join(_SAVE, fname))

        fname = f"P_tt_{ttype}_br_{behav}_{epoch}_{s_id}_freq_{freq}_thr_{thr}_surr_{surr}.nc"
        P.to_netcdf(os.path.join(_SAVE, fname))
