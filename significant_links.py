"""
Compute number of siginificant links between regions across
sessions.
"""
import argparse
import os

import numpy as np
import xarray as xr
from frites.utils import parallel_func
from scipy.stats import ks_2samp, ttest_ind
from config import sessions
from GDa.util import _extract_roi, _create_roi_area_mapping

#######################################################################
# Argument parsing
#######################################################################
parser = argparse.ArgumentParser()
parser.add_argument("IDX", help="index of the session to run",
                    type=int)
parser.add_argument("METRIC", help="which dFC metric to use",
                    type=str)

args = parser.parse_args()
# The index of the session to use
idx = args.IDX
# Get name of the dFC metric
metric = args.METRIC

at = 'cue'
_SEED = 8179273

#######################################################################
# Define path to load and save data
#######################################################################

_ROOT = os.path.expanduser("~/funcog/gda")
_RESULTS = os.path.join("Results", "lucy", "significance_analysis")
# Path to coherence
_COH_PATH = os.path.join(
    _ROOT, f"Results/lucy/{sessions[idx]}/session01", f"{metric}_at_cue.nc"
)
# Path to surrogate coherence
_COH_PATH_SURR = os.path.join(
    _ROOT, f"Results/lucy/{sessions[idx]}/session01", f"{metric}_at_cue_surr.nc"
)


#######################################################################
# Load original and surrogate coherence
#######################################################################
coh = xr.load_dataarray(_COH_PATH).astype(np.float32)
coh = coh.transpose("roi", "freqs", "trials", "times")

coh_surr = xr.load_dataarray(_COH_PATH_SURR).astype(np.float32)
coh_surr = coh_surr.transpose("roi", "freqs", "trials", "times")


#######################################################################
# Statistical testing the distributions (KS-test and t-test)
#######################################################################
"""
The null hypothesis is that the two distributions are identical,
F(x)=G(x) for all x; the alternative is that they are not identical.

**ks-test:** If the KS statistic is small or the p-value is high,
then we cannot reject the null hypothesis in favor of the alternative.

**t-test:**  A p-value larger than a chosen threshold (e.g. 5% or 1%) indicates
that our observation is not so unlikely to have occurred by chance.
"""


def significance_test(coh=None, verbose=False, n_jobs=5):
    """
    Method to compute the ks- and t-test for each frequency band
    in parallel.
    """
    n_rois, n_bands, n_trials, n_times = coh.shape

    def _for_band(band):

        # Store p-value for KS-test
        ks = np.zeros(coh.shape[0])

        for i in range(coh.shape[0]):
            ks[i] = ks_2samp(
                coh[i, band, ...].values.flatten(),
                coh_surr[i, band, ...].values.flatten(),
                alternative="two-sided",
            )[1]

        tt = ttest_ind(
            coh[:, band, :, :].values.reshape(n_rois, n_trials * n_times),
            coh_surr[:, band, :, :].values.reshape(n_rois, n_trials * n_times),
            alternative="greater",
            axis=-1,
            #equal_var=False,
        )[1]
        return np.array([ks, tt])

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_band, n_jobs=n_jobs, verbose=verbose, total=n_bands
    )
    p_values = parallel(p_fun(band) for band in range(n_bands))

    p_values = np.asarray(p_values).T

    return p_values


# Compute p-values
if coh.nbytes * 1e-9 * 2 > 70:
    n_jobs = 1
else:
    n_jobs = 5

p_values = significance_test(coh, verbose=True, n_jobs=n_jobs)
# Convert to xarray
p_values = xr.DataArray(
    p_values,
    dims=("roi", "p", "freqs"),
    coords=dict(roi=coh.roi.data, freqs=coh.freqs.data, p=["ks", "t"]),
)

#######################################################################
# Compute number of sig. links between each pair of regions
#######################################################################

# Remove with ROI edges
roi_s, roi_t = _extract_roi(p_values.roi.data, "-")
p_values = p_values.isel(roi=np.logical_not(roi_s == roi_t))

# Significance level
alpha = 0.0001
# Number of regions between pairs of channels
p_sig = (p_values <= alpha).astype(int).groupby("roi").mean("roi")

# Remove links with less than 10 samples
rois, counts = np.unique(p_values.roi.data, return_counts=True)
p_sig = p_sig.sel(roi=rois[counts >= 10])

#######################################################################
# Convert p_sig to matrix
#######################################################################

# Get rois
roi = p_sig.roi.data
# Get map from roi to index
roi_s, roi_t, roi_is, roi_it, areas, mapping = _create_roi_area_mapping(roi)
# # Number of rois
n_rois = len(areas)
# # Number of edges
n_pairs = len(roi_s)

sources, targets = [], []
sources += [mapping[s] for s in roi_s]
targets += [mapping[t] for t in roi_t]

p_mat = xr.DataArray(
    np.zeros((n_rois, n_rois, 2, p_sig.sizes["freqs"])),
    dims=("sources", "targets", "p", "freqs"),
    coords=dict(sources=areas, targets=areas, p=[
                "ks", "t"], freqs=p_sig.freqs.data),
)

for p, roi in enumerate(roi):
    x, y = roi.split("-")
    s, t = mapping[x], mapping[y]
    p_mat[s, t] = p_mat[t, s] = p_sig[p]


#######################################################################
# Saving to data frame
#######################################################################

df = p_sig.to_dataframe(name='n_edges').reset_index()
x_s, x_t = _extract_roi(df['roi'].values, '-')
df['sources'] = x_s
df['targets'] = x_t
# df = p_values.to_dataframe(name='pval').reset_index()
file_path = os.path.join(
    _ROOT, _RESULTS, f"nedges_{metric}_{sessions[idx]}_sum.csv")
# p_values.to_netcdf(file_path)
df.to_csv(file_path)
