import sys

sys.path.insert(1, "/home/vinicius/storage1/projects/GrayData-Analysis")  # noqa

import os
import argparse

import numpy as np
import xarray as xr  # noqa
import pickle

from brainconn.centrality import (
    betweenness_bin,
    betweenness_wei,
)
from brainconn.distance import distance_bin, distance_wei, efficiency_wei
from brainconn.degree import strengths_dir
from tqdm import tqdm
from frites.utils import parallel_func


from config import get_dates, bands


##############################################################################
# ARGUMENTS
##############################################################################

_ROOT = os.path.expanduser("~/funcog/gda/")

parser = argparse.ArgumentParser()
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("THR", help="threshold level used", type=int)
args = parser.parse_args()

monkey = args.MONKEY
thr = args.THR

sessions = get_dates(monkey)

stage_labels = ["P", "S", "D1", "D2", "Dm"]

freqs = np.median(bands[monkey], axis=1)

##############################################################################
# FUNCTIONS
##############################################################################


def distance_inv_wei(G):
    """
    From the source code of brainconn
    https://github.com/fiuneuro/brainconn/blob/c24bd15/brainconn/distance/distance.py#L581
    """
    n = len(G)
    D = np.zeros((n, n))  # distance matrix
    D[np.logical_not(np.eye(n))] = np.inf

    for u in range(n):
        # distance permanence (true is temporary)
        S = np.ones((n,), dtype=bool)
        G1 = G.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest
            for v in V:
                (W,) = np.where(G1[v, :])  # neighbors of smallest nodes
                td = np.array([D[u, W].flatten(), (D[u, v] + G1[v, W]).flatten()])
                D[u, W] = np.min(td, axis=0)

            if D[u, S].size == 0:  # all nodes reached
                break
            minD = np.min(D[u, S])
            if np.isinf(minD):  # some nodes cannot be reached
                break
            (V,) = np.where(D[u, :] == minD)

    np.fill_diagonal(D, 1)
    D = 1 / D
    np.fill_diagonal(D, 0)
    return D


def get_area_mapping(unique_areas):
    area2idx = dict(zip(unique_areas, range(len(unique_areas))))
    return area2idx


def get_unique_areas_mapping(monkey, session, ttype=1, br=1, surr=0):
    unique_areas = []
    for t in range(5):
        for freq in freqs:
            areas, _ = load_areas_times(
                monkey, session, stage_labels[t], freq, ttype=ttype, br=br, surr=surr
            )
            unique_areas += [np.unique(np.hstack(areas))]
    unique_areas = np.unique(np.concatenate(unique_areas))
    return unique_areas, get_area_mapping(unique_areas)


def load_areas_times(
    monkey,
    session,
    epoch,
    freq,
    ttype=1,
    br=1,
    decim=1,
    trials=False,
    surr=0,
    thr_type="relative",
):

    _path_to_ava = os.path.expanduser(f"~/funcog/gda/Results/{monkey}/avalanches/")

    # Load node region label
    fname = os.path.join(
        _path_to_ava,
        f"areas_tt_{ttype}_br_{br}_{epoch}_{session}_freq_{freq}_{thr_type}_thr_{thr}_decim_{decim}_surr_{surr}.pkl",
    )
    with open(fname, "rb") as f:
        areas = pickle.load(f)

    # Load node time label
    fname = os.path.join(
        _path_to_ava,
        f"times_tt_{ttype}_br_{br}_{epoch}_{session}_freq_{freq}_{thr_type}_thr_{thr}_decim_{decim}_surr_{surr}.pkl",
    )
    with open(fname, "rb") as f:
        times = pickle.load(f)

    if trials:
        # Load node time label
        fname = os.path.join(
            _path_to_ava,
            f"trials_tt_{ttype}_br_{br}_{epoch}_{session}_freq_{thr_type}_{freq}_thr_{thr}_decim_{decim}_surr_{surr}.pkl",
        )
        with open(fname, "rb") as f:
            trials = pickle.load(f)

        return areas, times, trials

    return areas, times


def return_propagation_scaffold(areas, times, unique_areas, area2idx):
    """
    Construct a propagation scaffold matrix based on areas and times.

    Inputs:
    ------
    areas : list of 2D arrays
        List of 2D arrays representing the areas in each session.
    times : list of 1D arrays
        List of 1D arrays representing the time points in each session.

    Returns:
    -------
    G : 3D array
        Concatenated propagation scaffold matrix of all sessions.
    """

    # unique_areas = np.unique(np.hstack(areas))
    # area2idx = get_area_mapping(unique_areas)

    nava = len(areas)

    G = []
    for n in range(nava):
        ua = np.unique(areas[n])  # noqa

        # Normalize time from zero
        t = times[n].astype(int)
        t = t - t.min()

        av_raster = np.zeros((len(unique_areas), t.max() + 1))
        for i in np.unique(t):
            idx = [area2idx[area] for area in np.unique(areas[n][t == i])]
            av_raster[idx, i] = 1

        targets = []
        for i in range(t.max()):
            targets += [
                np.logical_and(
                    ~np.logical_and(av_raster[:, i], av_raster[:, i + 1]),
                    av_raster[:, i + 1],
                )
            ]

        g = []
        for i in range(av_raster.shape[1] - 1):
            g += [np.outer(av_raster[:, i], targets[i])]

        G += [np.stack(g)]

    G = np.concatenate(G, axis=0)

    return G, unique_areas


betweenness_func = {"b": betweenness_bin, "w": betweenness_wei}
distance_func = {"b": distance_bin, "w": distance_wei}


def compute_betweenness(monkey, epoch="P", freq=27, type="w", ttype=1, br=1, surr=0):

    betweenness = []
    outStrength = []
    inStrength = []
    efficiency = []

    for session in tqdm(sessions):
        areas, times = load_areas_times(
            monkey, session, epoch, freq, ttype=ttype, br=br, trials=False, surr=surr
        )

        unique_areas, area2idx = get_unique_areas_mapping(
            monkey, session, ttype=ttype, br=br, surr=surr
        )

        G, ua = return_propagation_scaffold(areas, times, unique_areas, area2idx)
        G = G.mean(0)

        if type == "b":
            G = (G > 0).astype(int)

        if type == "b":
            D = distance_bin(1 / G)
        else:
            D = distance_inv_wei(G)

        bet = betweenness_func[type](D)
        inK, outK, K = strengths_dir(G)
        eff = efficiency_wei(G, local=True)

        dims = "roi"
        coords = {"roi": ua}

        bet = xr.DataArray(bet, dims=dims, coords=coords)
        inK = xr.DataArray(inK, dims=dims, coords=coords)
        outK = xr.DataArray(outK, dims=dims, coords=coords)
        eff = xr.DataArray(eff, dims=dims, coords=coords)

        betweenness += [bet]
        inStrength += [inK]
        outStrength += [outK]
        efficiency += [eff]

    betweenness = xr.concat(betweenness, "sessions")
    inStrength = xr.concat(inStrength, "sessions")
    outStrength = xr.concat(outStrength, "sessions")
    efficiency = xr.concat(efficiency, "sessions")

    return betweenness, inStrength, outStrength, efficiency


##############################################################################
# Create propagation scaffolds
##############################################################################

freqs = np.array([3.0, 10.0, 20.0, 34.5, 61.5])


def create_scaffolds(monkey, session, ttype=1, br=1, surr=0):

    unique_areas, area2idx = get_unique_areas_mapping(monkey, session, ttype, br, surr)

    GPS = []
    for t in range(5):
        GPS += [[]]
        for freq in freqs:
            areas, times = load_areas_times(
                monkey, session, stage_labels[t], freq, ttype=ttype, br=br, surr=surr
            )
            G, ua = return_propagation_scaffold(areas, times, unique_areas, area2idx)
            GPS[-1] += [G.mean(axis=0)]

    GPS = np.stack(GPS)
    GPS = xr.DataArray(
        GPS,
        dims=("times", "freqs", "sources", "targets"),
        coords=(range(5), freqs, ua, ua),
    )

    return GPS


def compute_GPS_stats(GPS, t_pow):

    nfreqs, ntimes, nroi = t_pow.shape

    bet = np.zeros((ntimes, nfreqs, nroi))
    phi = np.zeros((ntimes, nfreqs, nroi))

    for t in range(ntimes):
        for f in range(nfreqs):
            G = (
                GPS.sel(sources=t_pow.roi.data)
                .sel(targets=t_pow.roi.data)
                .isel(times=t, freqs=f)
                .fillna(0)
            ).data
            bet[t, f] = betweenness_wei(distance_inv_wei(G))
            ins, outs, _ = strengths_dir(G)
            phi[t, f] = ins - outs
    bet = xr.DataArray(
        bet,
        dims=("times", "freqs", "roi"),
        coords={"roi": t_pow.roi.astype(str), "freqs": t_pow.freqs},
    )
    phi = xr.DataArray(
        phi,
        dims=("times", "freqs", "roi"),
        coords={"roi": t_pow.roi.astype(str), "freqs": t_pow.freqs},
    )

    return bet, phi


def compute_GPS_stats_sessions(GPS, t_pow, n_jobs=1, verbose=False):

    n_sessions = len(GPS)

    parallel, p_fun = parallel_func(
        compute_GPS_stats, n_jobs=n_jobs, verbose=verbose, total=n_sessions
    )
    # Compute the single trial coherence
    out = parallel(p_fun(GPS[i], t_pow) for i in range(n_sessions))

    bet = [out[i][0] for i in range(n_sessions)]
    phi = [out[i][1] for i in range(n_sessions)]

    bet = xr.concat(bet, "sessions").mean("sessions")
    phi = xr.concat(phi, "sessions").mean("sessions")

    return bet, phi


_RESULTS = os.path.join(_ROOT, f"Results/{monkey}/mutual_information/power/")

path_tv = os.path.join(
    _RESULTS, "tval_pow_1_br_1_q_0_aligned_cue_avg_1_fdr_rfx_slvr_0.nc"
)
t_pow = xr.load_dataarray(path_tv)

# Scaffolds
GPS = []
for session in tqdm(get_dates("lucy")):
    GPS += [create_scaffolds("lucy", session, ttype=1, br=1)]
GPS = xr.concat(GPS, "sessions")

GPS.to_netcdf(f"data/GPS_{monkey}.nc")

# Stats
bet, phi = [], []
#for freq in freqs:
#    out = compute_GPS_stats_sessions(GPS, t_pow, freq, True)
#    bet += [out[0]]
#    phi += [out[1]]
#
#bet = xr.concat(bet, "freqs").assign_coords({"freqs": freqs})
#phi = xr.concat(phi, "freqs").assign_coords({"freqs": freqs})


bet, phi = compute_GPS_stats_sessions(GPS, t_pow, 20, True)

bet.to_netcdf(f"data/bet_{monkey}.nc")
phi.to_netcdf(f"data/phi_{monkey}.nc")
