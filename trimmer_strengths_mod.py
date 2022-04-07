""" Compute the trimmer-strengths for the MCs of each """
import os
import argparse

import numpy as np
import xarray as xr

from frites.utils import parallel_func
from config import sessions

###############################################################################
# Argument parsing
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("METRIC",
                    help="which network metric to use",
                    type=str)
parser.add_argument("SIDX", help="index of the session to run",
                    type=int)
args = parser.parse_args()
# Which FC metric to use
metric = args.METRIC
# The index of the session to use
idx = args.SIDX
session = sessions[idx]

###############################################################################
# Loading meta-connectivity
###############################################################################
_ROOT = os.path.expanduser("~/funcog/gda")
_RESULTS = "Results/lucy/meta_conn"
_MCPATH = os.path.join(_ROOT, _RESULTS, f"MC_{metric}_{session}.nc")

MC = xr.load_dataarray(_MCPATH)

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

###############################################################################
# Compute and store trimmer-strengths
###############################################################################


ts = tensor_trimmer_strengths(MC, n_jobs=1, verbose=False)
save_path = os.path.join(_ROOT, _RESULTS, f"ts_{metric}_{session}.nc")
ts.to_netcdf(save_path)
# ts.to_dataframe(name="ts").reset_index().to_csv(save_path)
