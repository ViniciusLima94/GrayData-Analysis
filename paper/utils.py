import numpy as np
import numba 
import scipy.stats

import sys

sys.path.insert(1, "/home/vinicius/storage1/projects/GrayData-Analysis")

from GDa.util import _extract_roi

#@numba.njit
def pearson_r(x, y):
    """
    Compute plug-in estimate for the Pearson correlation coefficient.
    """
    """
    return (
        np.sum((x - np.mean(x)) * (y - np.mean(y)))
        / np.std(x)
        / np.std(y)
        / np.sqrt(len(x))
        / np.sqrt(len(y))
    )
    """
    return scipy.stats.pearsonr(x, y).statistic


#@numba.njit
def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))


#@numba.njit
def draw_bs_pairs(x, y):
    """Draw a pairs bootstrap sample."""
    inds = np.arange(len(x))
    bs_inds = draw_bs_sample(inds)

    return x[bs_inds], y[bs_inds]


#@numba.njit
def draw_bs_pairs_reps_pearson(x, y, size=1):
    """
    Draw bootstrap pairs replicates.
    """
    out = np.empty(size)
    for i in range(size):
        out[i] = pearson_r(*draw_bs_pairs(x, y))
    return out

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

def add_stats_annot(pval, x1, x2, y, h, col):
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text(
        (x1 + x2) * 0.5,
        y + h,
        convert_pvalue_to_asterisks(pval),
        ha="center",
        va="bottom",
        color=col,
    )

def _create_roi_area_mapping(roi):
    """
    Create a mapping between pairs of rois and integer indexes
    Parameters
    ----------
    roi: array_like
       Array of size (n_edges) containing the name of the rois
       (i, j) that form the FC link separated by "-". 
       Ex: ["V1-a3", "V6a-LIP", ..., "a46d-a8"]
    Returns
    -------
    roi_s: array_like
        The name of the source areas
    roi_t: array_like
        The name of the target areas
    roi_is: array_like
        The index of the source areas
    roi_it: array_like
        The index of the target areas
    areas: array_like
        The name of the areas
    mapping: dict
        The mapping area-index created
    """

    # Get sources and target names
    roi_s, roi_t = _extract_roi(roi, "-")
    # Get unique area names
    areas = np.unique(np.stack((roi_s, roi_t)))
    # Get number of unique areas
    n_areas = len(areas)
    # Assign a index for each area
    mapping = dict(zip(areas, range(n_areas)))
    # Convert roi_s roi_t to integer indexes
    roi_is = np.array([mapping[s] for s in roi_s])
    roi_it = np.array([mapping[t] for t in roi_t])
    # return rois names from roi
    # the index for each edge roi
    # and the mapping
    return roi_s, roi_t, roi_is, roi_it, areas, mapping
    
    
def to_mat(df, key):

    rois = df.roi.values
    roi_s, roi_t, _, _, unique_rois, mapping = _create_roi_area_mapping(rois)
    mat = np.zeros((len(unique_rois), len(unique_rois)))

    for i, (s, t) in enumerate(zip(roi_s, roi_t)):
        i_s, i_t = mapping[s], mapping[t]
        mat[i_s, i_t] = mat[i_t, i_s] = df[key].values[i]

    return mat, unique_rois
    
    
def remove_same_roi(df):

    rois = df.roi.values
    roi_s, roi_t = _extract_roi(rois, "-")
    return df.iloc[~(roi_s == roi_t), :]

def xr_remove_same_roi(xar):
    
    roi_s, roi_t = _extract_roi(xar.roi.data, "-")
    return xar.isel(roi=~(roi_s == roi_t))

def xr_remove_same_roi_mc(xar):
    
    roi_s, roi_t = _extract_roi(xar.roi.data, "~")
    
    roi_s_1, roi_s_2 = _extract_roi(roi_s, "-")
    roi_t_1, roi_t_2 = _extract_roi(roi_t, "-")
    
    idx1 = (roi_s == roi_t)
    idx2 = (roi_s_1 == roi_t_1)
    idx3 = (roi_s_2 == roi_t_2)
    
    idx = np.logical_or(idx1, idx2)
    idx = np.logical_or(idx, idx3)
    
    return xar.isel(roi=~idx)

def remove_sca(df):
    
    sca = ["Caudate", "Claustrum", "Thal", "Putamen"]
    roi_s, roi_t = _extract_roi(df.roi.values, "-")
    idx = np.logical_or([s in sca for s in roi_s],
                         [t in sca for t in roi_t])
    return df.iloc[~idx, :]

def node_remove_sca(df):
    
    sca = ["Caudate", "Claustrum", "Thal", "Putamen"]
    idx = np.array([r in sca for r in df.roi.values])
    return df.iloc[~idx, :]

def node_xr_remove_sca(xar):
    
    sca = ["Caudate", "Claustrum", "Thal", "Putamen"]
    idx = np.array([r in sca for r in xar.roi.data])
    return xar.isel(roi=~idx)

def edge_xr_remove_sca(xar):
    
    sca = ["Caudate", "Claustrum", "Thal", "Putamen"]
    roi_s, roi_t = _extract_roi(xar.roi.data, "-")
    idx = np.logical_or([s in sca for s in roi_s],
                         [t in sca for t in roi_t])
    return xar.isel(roi=~idx)

def mc_edge_xr_remove_sca(xar):
    
    sca = ["Caudate", "Claustrum", "Thal", "Putamen"]
    roi_s, roi_t = _extract_roi(xar.roi.data, "~")
    
    roi_s_1, roi_s_2 = _extract_roi(roi_s, "-")
    roi_t_1, roi_t_2 = _extract_roi(roi_t, "-")
    
    idx_1 = np.logical_or([s in sca for s in roi_s_1],
                         [t in sca for t in roi_t_1])
    
    idx_2 = np.logical_or([s in sca for s in roi_s_2],
                          [t in sca for t in roi_t_2])
    
    
    return xar.isel(roi=~np.logical_or(idx_1, idx_2))
    
    
    
    