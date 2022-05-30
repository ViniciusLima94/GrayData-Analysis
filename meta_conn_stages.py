import os
import sys
import numpy as np
import numba as nb
import xarray as xr
from config import sessions
from GDa.util import _extract_roi
from tqdm import tqdm

surr = bool(sys.argv[-1])

##############################################################################
# Auxiliar function
##############################################################################


def convert_to_stream(adj, sources, targets, dtype=np.float32):
    """
    Convert adjacency tensor to edge time-series tensor.


    Parameters:
    ----------
    adj: array_like
        The adjacency tensor (roi, roi, freqs, trials, times)
    sources: array_like
        list of source nodes.
    targets: array_like
        list of target nodes.

    Returns
    -------
    The edge time-series tensor (roi,freqs,trials,times).
    """

    assert adj.ndim == 4

    # Adjacency tensor dimensions
    n_roi, _, n_bands, n_times = adj.shape
    # Number of edges
    n_pairs = int(n_roi * (n_roi - 1) / 2)

    nb.jit(nopython=True)

    def _convert(adj):
        # Edge time-series tensor
        tensor = np.zeros([n_pairs, n_bands, n_times], dtype=dtype)

        for p in range(n_pairs):
            i, j = sources[p], targets[p]
            tensor[p, ...] = adj[i, j, ...]
        return tensor

    tensor = _convert(adj)

    # In case adj is and DataArray the output
    # will also be
    if isinstance(adj, xr.DataArray):
        # Check if dimensions have the apropriate labels
        dims = ["sources", "targets", "freqs", "times"]
        np.testing.assert_array_equal(adj.dims, dims)
        # Get sources and targets roi names
        x_s, x_t = adj.sources.data, adj.targets.data
        roi_c = np.c_[x_s[sources], x_t[targets]]
        idx = np.argsort(np.char.lower(roi_c.astype(str)), axis=1)
        roi = np.c_[[r[i] for r, i in zip(roi_c, idx)]]
        rois = []
        rois += [f"{s}~{t}" for s, t in roi]
        # Convert to DataArray
        tensor = xr.DataArray(
            tensor, dims=("roi", "freqs", "times"),
            coords={"roi": rois, "freqs": adj.freqs.data,
                    "times": adj.times.data}
        )
    return tensor


##############################################################################
# Reading data
##############################################################################
_ROOT = os.path.expanduser("~/funcog/gda")
_RESULTS = "Results/lucy/meta_conn"


def get_file_name(session):
    if not surr:
        return os.path.join(_ROOT, _RESULTS, f"MC_coh_{session}.nc")
    return os.path.join(_ROOT, _RESULTS, f"MC_coh_{session}_surr.nc")


# Load all MCs and average over channels |-> ROI
MC = []
for session in tqdm(sessions):
    temp = xr.load_dataarray(get_file_name(session))
    temp = temp.groupby("sources").mean(
        "sources").groupby("targets").mean("targets")
    temp.attrs["sources"], temp.attrs["targets"] = np.triu_indices(
        temp.shape[0], 1)
    for f in range(temp.shape[-2]):
        for t in range(temp.shape[-1]):
            np.fill_diagonal(temp[..., f, t].values, 1)
    MC += [temp]

# From matrix to stream representation
MCg = []
for mat in tqdm(MC):
    out = convert_to_stream(mat, mat.attrs["sources"], mat.attrs["targets"])
    MCg += [out]

# Average over repeated meta-edges across sessions
MC_avg = xr.concat(MCg, dim="roi").groupby("roi").mean("roi")

# Convert back to matrix
x_s, x_t = _extract_roi(MC_avg.roi.values, "~")  # Get rois
unique_rois = np.unique(np.stack((x_s, x_t)))
index_rois = dict(zip(unique_rois, range(len(unique_rois))))

temp = xr.DataArray(
    np.zeros((len(unique_rois), len(unique_rois), 10, 5)),
    dims=("sources", "targets", "freqs", "times"),
    coords=(unique_rois, unique_rois, MC_avg.freqs.values,
            MC_avg.times.values),
)
for i, (s, t) in enumerate(zip(x_s, x_t)):
    i_s, i_t = index_rois[s], index_rois[t]
    temp[i_s, i_t, ...] = temp[i_t, i_s, ...] = MC_avg[i]

##############################################################################
# Save global MC
##############################################################################
if not surr:
    temp.to_netcdf(os.path.join(_ROOT, _RESULTS, "MC_coh.nc"))
else:
    temp.to_netcdf(os.path.join(_ROOT, _RESULTS, "MC_coh_surr.nc"))
