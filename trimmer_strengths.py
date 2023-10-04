""" Compute the trimmer-strengths for the MCs of each """
import os
import argparse

import numpy as np
import xarray as xr

from frites.utils import parallel_func
from config import get_dates, return_delay_split
from tqdm import tqdm


###############################################################################
# Parsing arguments
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "THR", help="wheter to use the thresholded coherence or not ", type=int
)
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("ALIGNED", help="wheter power was align to cue or match", type=str)
parser.add_argument("DELAY", help="which type of delay split to use", type=int)
args = parser.parse_args()
thr = args.THR
monkey = args.MONKEY
at = args.ALIGNED
ds = args.DELAY

early_cue, early_delay = return_delay_split(monkey=monkey, delay_type=ds)

sessions = get_dates(monkey)

###############################################################################
# Loading meta-connectivity
###############################################################################
_ROOT = os.path.expanduser("~/funcog/gda")
_RESULTS = f"Results/{monkey}/meta_conn"

###############################################################################
# Define functions to compute trimmer strengths
###############################################################################


def _trimmer_strengths(meta_conn, sources, targets, n_nodes=None):
    """Given a MC matrix it computes the trimmer strengths for each node"""
    # Get the number of nodes
    if n_nodes is None:
        n_nodes = np.max([sources, targets]) + 1
    # Store trimmer-strengths for each node
    ts = np.zeros(n_nodes)
    for i in range(n_nodes):
        # Get indexes where node i is present
        idx = np.logical_or(sources == i, targets == i)
        # Get sub-matrix only with meta-edges containing i
        sub_mat = meta_conn[np.ix_(idx, idx)]
        ts[i] = np.triu(sub_mat, 1).sum(axis=(0, 1))
    return ts


def _trimmer_entanglement(meta_conn, sources, targets, n_nodes=None):
    """Given a MC matrix it computes the trimmer strengths for each node"""
    # Get the number of nodes
    if n_nodes is None:
        n_nodes = np.max([sources, targets]) + 1
    # Number of edges in the MC
    n_pairs = len(sources)
    # Store trimmer-strengths for each node
    ets = np.zeros(n_pairs)
    for ii, (i, j) in enumerate(zip(sources, targets)):
        idx_s = np.logical_and(sources == i, targets == j)
        for k in range(n_nodes):
            # Get indexes where node i is present
            idx_t1 = np.logical_or(sources == i, targets == i)
            idx_t2 = np.logical_or(sources == j, targets == j)
            # Get sub-matrix only with meta-edges containing i
            sub_mat1 = meta_conn[np.ix_(idx_s, idx_t1)]
            sub_mat2 = meta_conn[np.ix_(idx_s, idx_t2)]
            ets[ii] = 0.5 * (sub_mat1.sum(-1) + sub_mat2.sum(-1))
    return ets


def tensor_trimmer_strengths(meta_conn, n_jobs=1, verbose=False):

    assert isinstance(meta_conn, xr.DataArray)
    assert meta_conn.ndim == 4
    assert "sources" in meta_conn.attrs.keys()
    assert "targets" in meta_conn.attrs.keys()

    areas = meta_conn.attrs["areas"]
    freqs, times = meta_conn.freqs.data, meta_conn.times.data
    # Number of times and freqs layers
    n_times, n_freqs = len(times), len(freqs)
    # Compute total number of layers
    n_layers = n_times * n_freqs
    # Get list of sources and targets
    sources, targets = meta_conn.attrs["sources"], meta_conn.attrs["targets"]
    # Number of nodes
    n_nodes = np.max([sources, targets]) + 1

    # Stack freqs and times layers
    meta_conn = meta_conn.stack(layers=("freqs", "times"))

    def _for_layer(n):
        return _trimmer_strengths(
            meta_conn[..., n].data, sources, targets, n_nodes=n_nodes
        )

    # Define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_layer, n_jobs=n_jobs, verbose=verbose, total=n_layers
    )
    # Compute the single trial coherence
    ts = parallel(p_fun(n) for n in range(n_layers))
    # Convert to numpy array
    ts = np.stack(ts, -1)

    # Unstack trials and time
    ts = ts.reshape((n_nodes, n_freqs, n_times))
    # Convert to xarray
    ts = xr.DataArray(
        ts,
        dims=("roi", "freqs", "times"),
        coords={"roi": areas, "freqs": freqs, "times": times},
    )
    return ts


def tensor_trimmer_entanglement(meta_conn, n_jobs=1, verbose=False):

    assert isinstance(meta_conn, xr.DataArray)
    assert meta_conn.ndim == 4
    assert "sources" in meta_conn.attrs.keys()
    assert "targets" in meta_conn.attrs.keys()

    areas = meta_conn.attrs["areas"]
    meta_links = meta_conn.sources.data
    freqs, times = meta_conn.freqs.data, meta_conn.times.data
    # Number of times and freqs layers
    n_times, n_freqs = len(times), len(freqs)
    # Compute total number of layers
    n_layers = n_times * n_freqs
    # Get list of sources and targets
    sources, targets = meta_conn.attrs["sources"], meta_conn.attrs["targets"]
    # Number of nodes
    n_nodes = np.max([sources, targets]) + 1
    # Number of edges
    n_pairs = len(sources)

    # Stack freqs and times layers
    meta_conn = meta_conn.stack(layers=("freqs", "times"))

    def _for_layer(n):
        return _trimmer_entanglement(
            meta_conn[..., n].data, sources, targets, n_nodes=n_nodes
        )

    # Define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_layer, n_jobs=n_jobs, verbose=verbose, total=n_layers
    )
    # Compute the single trial coherence
    ets = parallel(p_fun(n) for n in range(n_layers))
    # Convert to numpy array
    ets = np.stack(ets, -1)

    # Unstack trials and time
    ets = ets.reshape((n_pairs, n_freqs, n_times))
    # Convert to xarray
    ets = xr.DataArray(
        ets,
        dims=("roi", "freqs", "times"),
        coords={"roi": meta_links, "freqs": freqs, "times": times},
    )
    return ets


###############################################################################
# Compute and store trimmer-strengths
###############################################################################


metric = "coh"

for session in tqdm(sessions):

    _MCPATH = os.path.join(
        _ROOT, _RESULTS, f"MC_{metric}_{session}_at_{at}_ds_{ds}_thr_{thr}.nc"
    )
    save_path_ts = os.path.join(
        _ROOT, _RESULTS, f"ts_{metric}_{session}_at_{at}_ds_{ds}_thr_{thr}.nc"
    )
    save_path_ent = os.path.join(
        _ROOT, _RESULTS, f"ent_{metric}_{session}_at_{at}_ds_{ds}_thr_{thr}.nc"
    )

    # Loading MC
    MC = xr.load_dataarray(_MCPATH)
    # Computing trimmer-strength
    ts = tensor_trimmer_strengths(MC, n_jobs=1, verbose=False)
    ts.to_netcdf(save_path_ts)
    # Computing trimmer entanglement
    # ets = tensor_trimmer_entanglement(MC, n_jobs=20, verbose=False)
    # ets.to_netcdf(save_path_tsent)
    # Computing entanglement
    ent = MC.sum("targets").rename({"sources": "roi"})
    ent.to_netcdf(save_path_ent)
