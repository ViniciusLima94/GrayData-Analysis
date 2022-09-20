import os
import xarray as xr
import matplotlib.pyplot as plt

from tqdm import tqdm
from config import sessions
from GDa.util import create_stages_time_grid
from frites.dataset import DatasetEphy
from frites.estimator import GCMIEstimator
from frites.workflow import WfMi

metric = "modularity"

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')

mod = []
stim = []
for s_id in tqdm(sessions):
    _FILE_NAME = f"coh_{metric}_thr_1_at_cue.nc"
    path_metric = \
        os.path.join(_ROOT,
                     f"Results/lucy/{s_id}/session01",
                     _FILE_NAME)
    # Load network feature
    feature = xr.load_dataarray(path_metric)
    # Add dummy roi dim if modularity
    if metric == 'modularity':
        feature = feature.expand_dims('roi', 1)

    # Convert to format required by the MI workflow
    mod += [feature.isel(roi=[r])
            for r in range(len(feature['roi']))]
    stim += [feature.attrs["stim"].astype(int)] \
        * len(feature['roi'])
    del feature

###############################################################################
# MI Workflow
###############################################################################

# Convert to DatasetEphy
dt = DatasetEphy(mod, y=stim, nb_min_suj=10,
                 times="times", roi="roi")

mi_type = 'cd'
inference = 'rfx'
kernel = None
estimator = GCMIEstimator(mi_type='cd', relative=False, copnorm=True,
                          biascorrect=False, demeaned=False, tensor=True,
                          gpu=False, verbose=None)
wf = WfMi(mi_type, inference, verbose=True, kernel=kernel, estimator=estimator)

kw = dict(n_jobs=20, n_perm=100)
cluster_th = None  # {float, None, 'tfce'}

mi, pvalues = wf.fit(dt, mcp="cluster", cluster_th=cluster_th, **kw)

###############################################################################
# Saving results
###############################################################################

# Path to results folder
_RESULTS = "Results/lucy/mutual_information"

path_mi = os.path.join(_ROOT,
                       _RESULTS,
                       f"mi_{metric}_avg_1_thr_1.nc")
path_tv = os.path.join(_ROOT,
                       _RESULTS,
                       f"t_{metric}_avg_1_thr_1.nc")
path_pv = os.path.join(_ROOT,
                       _RESULTS,
                       f"pval_{metric}_avg_1_thr_1.nc")

mi.to_netcdf(path_mi)
wf.tvalues.to_netcdf(path_tv)
pvalues.to_netcdf(path_pv)


###############################################################################
# Plotting
###############################################################################

if metric == "modularity":
    mi_sig = mi * (pvalues <= 0.05)
    mi_sig.squeeze().plot.imshow(x="times", y="freqs", cmap="turbo", vmax=0.01)
    plt.savefig(f"figures/mi_{metric}.png")
