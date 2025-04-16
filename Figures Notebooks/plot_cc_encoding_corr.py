import sys

sys.path.insert(1, "/home/vinicius/storage1/projects/GrayData-Analysis")


import os
from scipy.stats import spearmanr
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from brainconn.centrality import (
    betweenness_bin,
    betweenness_wei,
)
from brainconn.core import score_wu
from brainconn.degree import strengths_dir
from brainconn.distance import distance_bin, distance_wei, efficiency_wei
from mne.stats import fdr_correction
from tqdm import tqdm

from config import get_dates
from GDa.flatmap.flatmap import flatmap
from GDa.loader import loader
from utils import *  # noqa

thr = 95

### Auxiliar functions


def convert_pvalue_to_asterisks(pvalues):
    """
    Convert an array of p-values to an array of asterisk string representations.
    A p-value of <= 0.0001 returns "****", a p-value of <= 0.001 returns "***",
    a p-value of <= 0.01 returns "**", a p-value of <= 0.05 returns "*", and
    any other p-value returns "ns".

    Parameters
    ----------
    pvalues : numpy.ndarray
        The array of p-values to be converted to asterisk string representations.

    Returns
    -------
    numpy.ndarray
        The array of asterisk string representations of the p-values.
    """
    asterisks = np.select(
        [pvalues <= 0.0001, pvalues <= 0.001, pvalues <= 0.01, pvalues <= 0.05],
        ["****", "***", "**", "*"],
        default="ns",
    )
    return asterisks


def plot_adj_modular(
    A, ci, offset=0.5, vmin=0, vmax=0.05, cmap="turbo", lw=5, color="k"
):
    """
    Plot a modular adjacency matrix with community boundaries.

    This function takes a adjacency matrix (A), a vector of community indices (ci), and optional arguments for
    plotting the matrix (offset, vmin, vmax, cmap, lw, color) and plots the adjacency matrix with lines denoting
    the boundaries between communities.

    Parameters:
    A (ndarray or xr.DataArray): adjacency matrix
    ci (ndarray): vector of community indices for each node in the adjacency matrix
    offset (float): offset for the community boundaries. Default is 0.5
    vmin (float): minimum value for color map. Default is 0
    vmax (float): maximum value for color map. Default is 0.05
    cmap (str): name of colormap to use. Default is 'turbo'
    lw (float): width of the lines used to draw the community boundaries. Default is 5
    color (str): color of the lines used to draw the community boundaries. Default is 'k'
    """
    _, c = np.unique(ci, return_counts=True)

    N = len(A)

    c = np.cumsum(c)
    c = np.hstack(([0], c))

    idx = np.argsort(ci)
    rois = A.sources.data[idx]

    if isinstance(A, xr.DataArray):
        plot_data = A.data[np.ix_(idx, idx)]
    else:
        plot_data = A[np.ix_(idx, idx)]
    plt.imshow(plot_data, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
    plt.xticks(range(N), rois, rotation=90)
    plt.yticks(range(N), rois)

    for i in range(1, len(c)):
        plt.hlines(
            c[i - 1] - offset, c[i] - offset, c[i - 1] - offset, lw=lw, color=color
        )
        plt.vlines(
            c[i - 1] - offset, c[i] - offset, c[i - 1] - offset, lw=lw, color=color
        )
        plt.hlines(c[i] - offset, c[i] - offset, c[i - 1] - offset, lw=lw, color=color)
        plt.vlines(c[i] - offset, c[i] - offset, c[i - 1] - offset, lw=lw, color=color)


def plot_brain_areas(ax, values, vmin=0, vmax=1, colormap="hot_r"):

    areas_dict = get_areas()

    area_no = dict(
        motor=0,
        parietal=1,
        prefrontal=2,
        somatosensory=3,
        temporal=4,
        visual=5,
        auditory=6,
    )

    areas = values.roi.data  # np.asarray([area for area in areas_dict.keys()])
    areas = [a.lower() for a in areas]
    fmap = flatmap(values.data, areas)

    fmap.plot(
        ax,
        ax_colorbar=None,
        cbar_title=None,
        alpha=0.4,
        colormap=colormap,
        colors=None,
        vmin=vmin,
        vmax=vmax,
    )


def spearmanr_bootstrap(x, y, sample_size=1, nboots=1):

    assert len(x) == len(y)

    CC = np.zeros(nboots)
    pvalues = np.zeros(nboots)

    for i in range(nboots):

        idx = np.random.randint(0, x.shape[0], size=sample_size)

        CC[i] = spearmanr(x[idx], y[idx]).statistic
        pvalues[i] = spearmanr(x[idx], y[idx]).pvalue

    return CC, pvalues


def compute_correlations(data_x, t_pow, freq, sample_size=100, nboots=100):

    nsessions = data_x.sizes["sessions"]
    correlations = []
    pvalues = []

    for t in range(5):
        # Concatenate sessions and flattent data
        x = data_x.sel(roi=t_pow.roi.values, times=t, freqs=freq)
        x = x.data.flatten()
        y = np.tile(t_pow.sel(times=t, freqs=freq), nsessions)

        # Remove NaN values
        nan_idx = np.isnan(x)
        x = x[~nan_idx]
        y = y[~nan_idx]

        # Compute the correlations
        corr, pval = spearmanr_bootstrap(x, y, nboots=nboots, sample_size=sample_size)
        correlations += [corr]
        pvalues += [pval]

    correlations = np.stack(correlations, axis=0)
    pvalues = np.stack(pvalues, axis=0)

    pvalues, _ = fdr_correction(pvalues, alpha=0.01)

    return correlations


def get_area_mapping(unique_areas):
    area2idx = dict(zip(unique_areas, range(len(unique_areas))))
    return area2idx


def node_s_core(GCS, nlevels=10):
    """
    Compute the node coreness based on strength levels.

    Parameters:
    - A (numpy.ndarray): The adjacency matrix of the graph.
    - nlevels (int, optional): The number of strength levels to consider. Default is 10.

    Returns:
    - scoreness (numpy.ndarray): An array representing the coreness of each node in the graph
      based on the given strength levels.

    This function calculates the coreness of nodes in a graph by considering different
    strength levels. It first computes the node strengths, then divides the strength range
    into `nlevels` levels. For each strength level, it calculates the coreness of nodes
    using a function `score_wu` and assigns the coreness values to the nodes. The final
    `scoreness` array contains the coreness values for each node.
    """

    rois, nrois = GCS.sources.data, len(GCS)
    A = GCS.data.copy()

    strengths = A.sum(axis=1)
    slevels = np.linspace(strengths.min(), strengths.max(), nlevels)
    scoreness = np.zeros_like(strengths)

    for s in slevels:
        scores = score_wu(A, s)[0].sum(1)
        if np.any(scores.sum()):
            idx = np.where(scores > 0)[0]
            scoreness[idx] = s

    scoreness = xr.DataArray(scoreness, dims=("roi"), coords=(rois,))

    return scoreness


def return_CA_mat(
    _path_to_ava,
    session,
    epoch,
    freq=27,
    ttype=1,
    br=1,
    decim=1,
    surr=0,
    thr=None,
    thr_type="relative",
):

    path = os.path.join(
        _path_to_ava,
        f"T_tt_{ttype}_br_{br}_{epoch}_{session}_freq_{freq}_{thr_type}_thr_{thr}_decim_{decim}_surr_{surr}.nc",
    )
    T = xr.load_dataarray(path)

    return T


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


def to_mat(df: pd.DataFrame, as_xr: bool = False):

    # Check if DF contains source and target columns
    assert ("source" in df.columns) and ("target" in df.columns)

    # Get sources and targets
    source, target = df.source.values, df.target.values

    # Get unique names
    regions = np.unique(np.concatenate((source, target)))
    # Number of regions
    nregions = len(regions)
    # Create encoding from regions to a given index
    mapping = create_regions_mapping(regions)
    # Allocate matrix
    FLN = np.empty((nregions, nregions))
    # Fill values
    FLN = FLN[np.ix_(df.source.map(mapping), df.target.map(mapping).values)].ravel()
    FLN = df.weight.values.astype(float)
    FLN = FLN.reshape(nregions, nregions)

    if as_xr:
        FLN = xr.DataArray(FLN, dims=("sources", "targets"), coords=(regions, regions))
    return FLN


def create_regions_mapping(regions: list):

    nregions = len(regions)

    return dict(zip(regions, range(nregions)))


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
        ua = np.unique(areas[n])

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


def get_encoding(monkey):

    path = os.path.expanduser(f"~/funcog/gda/Results/{monkey}/mutual_information/power")

    p_pow = node_xr_remove_sca(
        xr.load_dataarray(
            os.path.join(
                path, f"mi_pow_tt_1_br_1_q_{thr}_aligned_cue_avg_1_fdr_rfx_slvr_0.nc"
            )
        )
    )
    t_pow = node_xr_remove_sca(
        xr.load_dataarray(
            os.path.join(
                path, f"tval_pow_1_br_1_q_{thr}_aligned_cue_avg_1_fdr_rfx_slvr_0.nc"
            )
        )
    )

    return p_pow, t_pow


def get_power(monkey):

    sessions = get_dates(monkey)

    powers = []

    for session in tqdm(sessions):
        kw_loader = dict(
            aligned_at="cue", channel_numbers=False, monkey=monkey, decim=1
        )

        temp = data_loader.load_power(
            **kw_loader, trial_type=1, behavioral_response=1, session=session
        )

        t_match_on = (temp.attrs["t_match_on"] - temp.attrs["t_cue_on"]) / 1000

        out = []
        for i in range(temp.sizes["trials"]):
            stages = [
                [-0.5, -0.2],
                [0, 0.4],
                [0.5, 0.9],
                [0.9, 1.3],
                [t_match_on[i] - 0.4, t_match_on[i]],
            ]
            temp_stages = []
            for t0, t1 in stages:
                temp_stages += [
                    temp.sel(times=slice(t0, t1)).isel(trials=i).mean("times")
                ]
            out += [xr.concat(temp_stages, "times")]
        out = xr.concat(out, "trials").mean("trials")
        out = out.transpose("roi", "freqs", "times").groupby("roi").mean("roi")
        powers += [out]

        # temp_2 = []
        # for ti, tf in stages:
        #    temp_2 += [temp.sel(times=slice(ti, tf)).mean(("times", "trials"))]

        # powers += [xr.concat(temp_2, "times").groupby("roi").mean("roi")]

    powers = xr.concat(powers, "sessions")
    return node_xr_remove_sca(powers)


def get_correlations(feature, encoding, freqs):
    correlations = []
    for freq in tqdm(freqs):
        correlations += [
            compute_correlations(feature, encoding, freq, sample_size=500, nboots=1000)
        ]
    return np.stack(correlations, axis=1)


def get_gcs(monkey, surr=0, thr=80):

    _path_to_ava = os.path.expanduser(f"~/funcog/gda/Results/{monkey}/avalanches/")

    sessions = get_dates(monkey)

    CS = []

    for freq in freqs:
        T_all_sessions = []
        for session in sessions:
            T = []
            for epoch in stage_labels:
                T += [
                    return_CA_mat(
                        _path_to_ava,
                        session,
                        epoch,
                        freq=freq,
                        ttype=1,
                        br=1,
                        surr=surr,
                        thr=thr,
                    ).sum("targets")
                ]
            T_all_sessions += [xr.concat(T, "times")]
        CS += [xr.concat(T_all_sessions, "sessions")]

    CS = xr.concat(CS, "freqs")
    CS = CS.assign_coords({"freqs": freqs})
    CS = CS.rename({"sources": "roi"})

    return CS


### Setting configurations

_ROOT = os.path.expanduser("~/funcog/gda/")

metric = "coh"
monkey = "lucy"

sessions = get_dates(monkey)

stages = [[-0.5, -0.2], [0, 0.4], [0.5, 0.9], [0.9, 1.3], [1.1, 1.5]]
stage_labels = ["P", "S", "D1", "D2", "Dm"]

areas_dict = get_areas()

colors = dict(
    zip(
        [
            "motor",
            "parietal",
            "prefrontal",
            "somatosensory",
            "temporal",
            "visual",
            "auditory",
        ],
        ["r", "aqua", "b", "m", "goldenrod", "green", "brown"],
    )
)

data_loader = loader(_ROOT=_ROOT)

kw_loader = dict(aligned_at="cue", channel_numbers=False, monkey=monkey)

## Coordination and encoding

p_pow_l, t_pow_l = get_encoding("lucy")
p_pow_e, t_pow_e = get_encoding("ethyl")


### Power/Coordination vs. encoding

freqs = t_pow_l.freqs.data.astype(int)

regions_l = np.array([areas_dict[r.lower()] for r in t_pow_l.roi.data])
idx = np.argsort(regions_l)
c_l = [colors[r] for r in regions_l[idx]]

regions_e = np.array([areas_dict[r.lower()] for r in t_pow_e.roi.data])
idx = np.argsort(regions_e)
c_e = [colors[r] for r in regions_e[idx]]

if os.path.isfile("power_l.nc"):
    power_l = xr.load_dataarray("power_l.nc")
else:
    power_l = get_power("lucy")
    power_l.to_netcdf("power_l.nc")

if os.path.isfile("power_e.nc"):
    power_e = xr.load_dataarray("power_e.nc")
else:
    power_e = get_power("ethyl")
    power_e.to_netcdf("power_e.nc")

correlations_POWER_ENC_l = get_correlations(power_l, t_pow_l, freqs)
correlations_POWER_ENC_e = get_correlations(power_e, t_pow_e, freqs)

### GCS vs. encoding

CS_l = get_gcs("lucy", surr=0, thr=thr)
CS_e = get_gcs("ethyl", surr=0, thr=thr)

CS_l.to_netcdf(f"CS_lucy_{thr}.nc")
CS_e.to_netcdf(f"CS_ethyl_{thr}.nc")


correlations_CS_ENC_l = get_correlations(CS_l, t_pow_l, freqs)
correlations_CS_ENC_e = get_correlations(CS_e, t_pow_e, freqs)

CS_ENC_l = np.stack(
    (
        np.quantile(correlations_CS_ENC_l, 0.05, axis=2),
        np.quantile(correlations_CS_ENC_l, 0.95, axis=2),
    )
)

POWER_ENC_l = np.stack(
    (
        np.quantile(correlations_POWER_ENC_l, 0.05, axis=2),
        np.quantile(correlations_POWER_ENC_l, 0.95, axis=2),
    )
)

CS_ENC_e = np.stack(
    (
        np.quantile(correlations_CS_ENC_e, 0.05, axis=2),
        np.quantile(correlations_CS_ENC_e, 0.95, axis=2),
    )
)

POWER_ENC_e = np.stack(
    (
        np.quantile(correlations_POWER_ENC_e, 0.05, axis=2),
        np.quantile(correlations_POWER_ENC_e, 0.95, axis=2),
    )
)

for t in range(1, 5):

    min_val = min(CS_ENC_l[0].min(), CS_ENC_e[0].min()) - 0.4
    max_val = max(CS_ENC_l[1].max(), CS_ENC_e[1].max()) + 0.2

    # Monkey L
    theta_l = np.linspace(0, np.pi, CS_ENC_l.shape[2])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, dpi=300)
    ax.plot(theta_l, np.median(CS_ENC_l[:, t], 0), color="blue", label="CC(GCS, ENC)")
    ax.fill_between(theta_l, CS_ENC_l[0, t], CS_ENC_l[1, t], color="blue", alpha=0.5)

    ax.plot(
        theta_l, np.median(POWER_ENC_l[:, t], 0), color="gray", label="CC(POWER, ENC)"
    )
    ax.fill_between(
        theta_l, POWER_ENC_l[0, t], POWER_ENC_l[1, t], color="gray", alpha=0.5
    )
    plt.legend(frameon=False)

    # Monkey E
    theta_e = np.linspace(np.pi, 2 * np.pi, CS_ENC_e.shape[2])

    ax.plot(theta_e, np.median(CS_ENC_e[:, t], 0), color="blue", ls="--")
    ax.fill_between(theta_e, CS_ENC_e[0, t], CS_ENC_e[1, t], color="blue", alpha=0.5)

    ax.plot(theta_e, np.median(POWER_ENC_e[:, t], 0), color="gray", ls="--")
    ax.fill_between(
        theta_e, POWER_ENC_e[0, t], POWER_ENC_e[1, t], color="gray", alpha=0.5
    )

    ax.set_ylim(min_val, max_val)

    # Add some coloring
    _theta_l = np.linspace(0, np.pi, 5000)
    ax.fill_between(
        _theta_l,
        min_val,
        max_val,
        where=((_theta_l >= 0) & (_theta_l < np.pi)),
        color="lightblue",
        alpha=0.1,
    )

    _theta_e = np.linspace(np.pi, 2 * np.pi, 5000)
    ax.fill_between(
        _theta_e,
        min_val,
        max_val,
        where=((_theta_e >= np.pi) | (_theta_e < 2 * np.pi)),
        color="lightgreen",
        alpha=0.1,
    )

    # Zero correlation line
    ax.patch.set_alpha(0.3)
    ax.grid(True, linestyle="--", linewidth=0.5, color="k")
    # Add vertical line in the middle
    ax.plot([0, 0], [min_val, max_val], color="k", linestyle="-", lw=2)
    ax.plot([np.pi, np.pi], [min_val, max_val], color="k", linestyle="-", lw=2)
    ax.set_xticks(np.hstack((theta_l, theta_e)), np.hstack((freqs, freqs[::-1])))
    ax.plot(
        np.linspace(0, 2 * np.pi, 5000), 5000 * [0], color="k", linestyle="--", lw=2
    )

    ax.set_ylabel("Frequency [Hz]", labelpad=25, fontsize=9)
    ax.set_theta_zero_location("N")
    plt.savefig(f"figures/n5/polar_encoding_{stage_labels[t]}.pdf")
