"""
Compute number of siginificant links between regions across
sessions.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from config import (sm_times, sm_kernel, sm_freqs, decim,
                    mode, freqs, n_cycles, sessions,
                    return_evt_dt)
from mne.viz import circular_layout, plot_connectivity_circle
from frites.utils import parallel_func
from frites.conn.conn_spec import conn_spec
from tqdm import tqdm
from scipy.stats import ks_2samp, ttest_ind
from GDa.temporal_network import temporal_network
from GDa.signal.surrogates import trial_swap_surrogates
from GDa.session import session
from GDa.net.util import convert_to_adjacency
from GDa.util import _extract_roi

idx = 0
metric = 'coh'
at = 'cue'
_SEED = 8179273
evt_dt = return_evt_dt(at)

#######################################################################
# Compute coherence for original and surrogate data
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
    sm_times=sm_times, sm_freqs=sm_freqs, sm_kernel=sm_kernel, block_size=2
)

# compute the coherence
coh = conn_spec(ses.data, **kw).astype(np.float32, keep_attrs=True)
# reordering dimensions
coh = coh.transpose("roi", "freqs", "trials", "times")
# Add areas as attribute
coh.attrs['areas'] = ses.data.roi.values.astype('str')

# compute surrogate coherence
data_surr = trial_swap_surrogates(
    ses.data.astype(np.float32), seed=_SEED, verbose=False)
coh_surr = conn_spec(data_surr, **kw)

coh = coh.transpose("roi", "freqs", "trials", "times")
coh_surr = coh_surr.transpose("roi", "freqs", "trials", "times")

#######################################################################
# Statistical testing the distributions (KS-test and t-test)
#######################################################################
"""
The null hypothesis is that the two distributions are identical, F(x)=G(x) for all x;
the alternative is that they are not identical.

**ks-test:** If the KS statistic is small or the p-value is high,
then we cannot reject the null hypothesis in favor of the alternative.

**t-test:**  A p-value larger than a chosen threshold (e.g. 5% or 1%) indicates
that our observation is not so unlikely to have occurred by chance.
"""


def significance_test(verbose=False, n_jobs=1):
    """
    Method to compute the ks- and t-test for each frequency band
    in parallel.
    """

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
        _for_band, n_jobs=n_jobs, verbose=verbose, total=coh.shape[1]
    )
    p_values = parallel(p_fun(band) for band in range(coh.shape[1]))
    return np.asarray(p_values).T


# Compute p-values
p_values = significance_test(verbose=True, n_jobs=10)
# Significance level
alpha = 0.05
# Convert to xarray
p_values = xr.DataArray(p_values, dims=("roi", "p", "freqs"),
                        coords=dict(roi=coh.roi.data,
                                    freqs=coh.freqs.data,
                                    p=["ks", "t"]))

#######################################################################
# Compute number of sig. links between each pair of regions
#######################################################################

# Number of regions between pairs of channels
p_sig = (p_values <= 0.05).groupby("roi").sum("roi")
# Get rois
roi_s, roi_t = _extract_roi(p_sig.roi.data, '-')
unique_rois = np.unique(np.concatenate((roi_s, roi_t)))
mapping = dict(zip(unique_rois, range(len(unique_rois))))
n_rois = len(unique_rois)
n_pairs = len(roi_s)

sources, targets = [], []
sources += [mapping[s] for s in roi_s]
targets += [mapping[t] for t in roi_t]

p_mat = xr.DataArray(np.zeros((n_rois, n_rois, 2, p_sig.sizes["freqs"])),
                     dims=("sources", "targets", "p", "freqs"),
                     coords=dict(sources=unique_rois,
                                 targets=unique_rois,
                                 p=["ks", "t"],
                                 freqs=p_sig.freqs.data)
                     )

for p, roi in enumerate(p_sig.roi.data):
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
df.to_csv(f'Results/lucy/significance_analysis/n_edges_{sessions[idx]}.csv')
