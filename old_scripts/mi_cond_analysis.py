"""
Computes the cond. mutual information between power a network feature
and the stimulus for all sessions.
"""
import os
import xarray as xr
import argparse

from config import sessions
from frites.dataset import DatasetEphy
from frites.estimator import GCMIEstimator
from frites.workflow import WfMi
from GDa.util import create_stages_time_grid, average_stages
from tqdm import tqdm

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("METRIC",
                    help="which network metric to use",
                    type=str)
parser.add_argument("AVERAGED",
                    help="wheter to analyse the avg. power or not",
                    type=int)

args = parser.parse_args()

metric = args.METRIC
avg = args.AVERAGED

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')

###############################################################################
# Iterate over all sessions and concatenate power
###############################################################################

sxx = []
for s_id in tqdm(sessions):
    # Uses power for trial type 1
    # behavioral response 1 aligned at cue
    _FILE_NAME = "power_tt_1_br_1_at_cue.nc"
    path_pow = \
        os.path.join(_ROOT,
                     f"Results/lucy/{s_id}/session01",
                     _FILE_NAME)
    power = xr.load_dataarray(path_pow)
    # Averages power for each period (baseline, cue, delay, match) if needed
    out = average_stages(power, avg)

    sxx += [out.isel(roi=[r]) for r in range(len(out['roi']))]

###############################################################################
# Iterate over all sessions and concatenate coherece
###############################################################################

coh = []
stim = []
for s_id in tqdm(sessions):
    _FILE_NAME = f"coh_{metric}_thr_1_at_cue.nc"
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
dt = DatasetEphy(sxx, y=coh, z=stim, nb_min_suj=10,
                 times="times", roi="roi")

mi_type = 'ccd'
inference = 'rfx'
kernel = None
estimator = GCMIEstimator(mi_type=mi_type, relative=False, copnorm=True,
                          biascorrect=False, demeaned=False, tensor=True,
                          gpu=False, verbose=None)
wf = WfMi(mi_type, inference, verbose=True, kernel=kernel, estimator=estimator)

kw = dict(n_jobs=20, n_perm=100)
cluster_th = None  # {float, None, 'tfce'}

mi, pvalues = wf.fit(dt, mcp="cluster", cluster_th=cluster_th, **kw)
