"""
Compute mutual information between the stimulus and the
trimmer strengths.
"""
import os
import xarray as xr
import argparse

from tqdm import tqdm
from config import sessions
from GDa.util import create_stages_time_grid, average_stages
from GDa.session import session_info
from frites.dataset import DatasetEphy
from frites.estimator import GCMIEstimator
from frites.workflow import WfMi

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')
_DATA = "Results/lucy/meta_conn/"

tst = []
stim = []
for s_id in tqdm(sessions):
    _FILE_NAME = os.path.join(_ROOT, _DATA, f"tst_{s_id}.nc")
    # Get stim labels for this session
    # trial_info = session_info(date=s_id).trial_info
    # s = trial_info[trial_info.behavioral_response.eq(
        # 1) & trial_info.trial_type.eq(1)].sample_image.values
    # Load network feature
    out = xr.load_dataarray(_FILE_NAME).transpose("trials", "roi", "freqs", "times")
    # out.attrs["stim"] = s
    # Convert to format required by the MI workflow
    tst += [out.isel(roi=[r])
            for r in range(len(out['roi']))]
    stim += [out.attrs["stim"].astype(int)] \
        * len(out['roi'])

###############################################################################
# MI Workflow
###############################################################################

# Convert to DatasetEphy
dt = DatasetEphy(tst, y=stim, nb_min_suj=10,
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
                       f"mi_tst_avg.nc")
path_tv = os.path.join(_ROOT,
                       _RESULTS,
                       f"t_tst.nc")
path_pv = os.path.join(_ROOT,
                       _RESULTS,
                       f"pval_tst.nc")

mi.to_netcdf(path_mi)
wf.tvalues.to_netcdf(path_tv)
pvalues.to_netcdf(path_pv)
