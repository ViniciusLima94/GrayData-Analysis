import os
import xarray as xr
import argparse

from tqdm import tqdm
from config import sessions
from GDa.util import average_stages
from frites.dataset import DatasetEphy
from frites.estimator import GCMIEstimator
from frites.workflow import WfMi

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("METRIC",
                    help="which FC metric to use",
                    type=str)
parser.add_argument("FEATURE",
                    help="which network feature to use",
                    type=str)
parser.add_argument("AVERAGED",
                    help="wheter to analyse the avg. power or not",
                    type=int)

args = parser.parse_args()

metric = args.METRIC
feat = args.FEATURE
avg = args.AVERAGED

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser('~/funcog/gda')

###############################################################################
# Iterate over all sessions and concatenate coherece
###############################################################################

coh = []
stim = []
for s_id in tqdm(sessions):
    _FILE_NAME = f"{metric}_{feat}_thr_1_at_cue.nc"
    path_metric = \
        os.path.join(_ROOT,
                     f"Results/lucy/{s_id}/session01",
                     _FILE_NAME)
    # Load network feature
    feature = xr.load_dataarray(path_metric)
    # Average if needed
    out = average_stages(feature, avg)
    # Convert to format required by the MI workflow
    coh += [out.isel(roi=[r])
            for r in range(len(out['roi']))]
    stim += [out.attrs["stim"].astype(int)] \
        * len(out['roi'])
    del feature


###############################################################################
# MI Workflow
###############################################################################

# Convert to DatasetEphy
dt = DatasetEphy(coh, y=stim, nb_min_suj=10,
                 times="times", roi="roi")

mi_type = 'cd'
inference = 'rfx'
kernel = None

if avg:
    # mcp = "maxstats"
    # check
    mcp = "fdr"
else:
    mcp = "cluster"

estimator = GCMIEstimator(mi_type='cd', copnorm=True,
                          biascorrect=True, demeaned=False, tensor=True,
                          gpu=False, verbose=None)
wf = WfMi(mi_type, inference, verbose=True, kernel=kernel, estimator=estimator)

kw = dict(n_jobs=20, n_perm=100)
cluster_th = None  # {float, None, 'tfce'}

mi, pvalues = wf.fit(dt, mcp=mcp, cluster_th=cluster_th, **kw)

###############################################################################
# Saving results
###############################################################################

# Path to results folder
_RESULTS = os.path.join(_ROOT,
                        "Results/lucy/mutual_information")

path_mi = os.path.join(_RESULTS,
                       f"mi_{metric}_{feat}_avg_{avg}_thr_1_{mcp}.nc")
path_tv = os.path.join(_RESULTS,
                       f"t_{metric}_{feat}_avg_{avg}_thr_1_{mcp}.nc")
path_pv = os.path.join(_RESULTS,
                       f"pval_{metric}_{feat}_avg_{avg}_thr_1_{mcp}.nc")

mi.to_netcdf(path_mi)
wf.tvalues.to_netcdf(path_tv)
pvalues.to_netcdf(path_pv)
