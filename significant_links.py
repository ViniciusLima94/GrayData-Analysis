"""
Compute number of siginificant links between regions across
sessions.
"""
import numpy as np
import xarray as xr
import argparse
import os

from config import (sm_times, sm_kernel, sm_freqs, decim,
                    mode, freqs, n_cycles, sessions,
                    return_evt_dt)
from frites.utils import parallel_func
from frites.conn.conn_spec import conn_spec
from scipy.stats import ks_2samp, ttest_ind
from GDa.signal.surrogates import trial_swap_surrogates
from GDa.session import session
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
evt_dt = return_evt_dt(at)

#######################################################################
# Compute coherence for original and surrogate data
#######################################################################

_ROOT = os.path.expanduser("~/funcog/gda")
_RESULTS = os.path.join("Results", "lucy", "significance_analysis")
_COH_PATH = os.path.join(_ROOT,
                         f"Results/lucy/{sessions[idx]}/session01",
                         f"{metric}_k_0.3_multitaper_at_cue.nc")

#######################################################################
# Load original coherence
#######################################################################
coh = xr.load_dataarray(_COH_PATH).astype(np.float32)

#######################################################################
# Compute surrogate coherence
#######################################################################
#  Instantiating session
ses = session(raw_path="GrayLab/",
              monkey="lucy",
              date=sessions[idx],
              session=1, slvr_msmod=True,
              align_to=at, evt_dt=evt_dt)
# Load data
ses.read_from_mat()

kw = dict(
    freqs=freqs, times="time", roi="roi", foi=None,
    n_jobs=20, pairs=None, sfreq=ses.data.attrs['fsample'],
    mode=mode, n_cycles=n_cycles, decim=decim, metric=metric,
    sm_times=sm_times, sm_freqs=sm_freqs, sm_kernel=sm_kernel, block_size=4
)

data_surr = trial_swap_surrogates(
    ses.data.astype(np.float32), seed=_SEED, verbose=False)

coh_surr = conn_spec(data_surr, **kw).astype(np.float32)

coh = coh.transpose("roi", "freqs", "trials", "times")
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


def significance_test(verbose=False, n_jobs=5):
    """
    Method to compute the ks- and t-test for each frequency band
    in parallel.
    """
    n_bands = coh.shape[1]

    def _for_band(band):
        # Store p-value for KS-test
        ks = np.zeros(coh.shape[0])
        # Store p-value for t-test
        tt = np.zeros(coh.shape[0])
        for i in range(coh.shape[0]):
            ks[i] = ks_2samp(
                coh[i, band, ...].values.flatten(),
                coh_surr[i, band, ...].values.flatten(),
                alternative="two-sided",
            )[1]
            tt[i] = ttest_ind(
                coh[i, band, ...].values.flatten(),
                coh_surr[i, band, ...].values.flatten(),
                alternative="two-sided",
                equal_var=False,
            )[1]
        return np.array([ks, tt])

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_band, n_jobs=n_jobs, verbose=verbose, total=n_bands
    )
    p_values = parallel(p_fun(band) for band in range(n_bands))
    return np.asarray(p_values).T


# Compute p-values
if coh.nbytes * 1e-9 * 2 > 70:
    n_jobs = 1
else:
    n_jobs = 5

p_values = significance_test(verbose=True, n_jobs=n_jobs)
# Convert to xarray
p_values = xr.DataArray(p_values, dims=("roi", "p", "freqs"),
                        coords=dict(roi=coh.roi.data,
                                    freqs=coh.freqs.data,
                                    p=["ks", "t"]))

#######################################################################
# Compute number of sig. links between each pair of regions
#######################################################################

# Significance level
alpha = 0.01
# Number of regions between pairs of channels
p_sig = (p_values <= alpha).groupby("roi").sum("roi")
# Get rois
roi = p_sig.roi.data
# Get map from roi to index
roi_s, roi_t, roi_is, roi_it, areas, mapping = _create_roi_area_mapping(roi)
# Number of rois
n_rois = len(areas)
# Number of edges
n_pairs = len(roi_s)

sources, targets = [], []
sources += [mapping[s] for s in roi_s]
targets += [mapping[t] for t in roi_t]

p_mat = xr.DataArray(np.zeros((n_rois, n_rois, 2, p_sig.sizes["freqs"])),
                     dims=("sources", "targets", "p", "freqs"),
                     coords=dict(sources=areas,
                                 targets=areas,
                                 p=["ks", "t"],
                                 freqs=p_sig.freqs.data)
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
file_path = os.path.join(
    _ROOT, _RESULTS, f"nedges_{metric}_{sessions[idx]}.csv")
df.to_csv(file_path)
