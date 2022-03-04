import os
import xarray as xr
import argparse

from config import sessions
from frites.dataset import DatasetEphy
from frites.estimator import GCMIEstimator
from frites.workflow import WfMi
from GDa.util import average_stages
from tqdm import tqdm

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("TT",   help="type of the trial",
                    type=int)
parser.add_argument("BR",   help="behavioral response",
                    type=int)
parser.add_argument("ALIGN", help="wheter to align data to cue or match",
                    type=str)
parser.add_argument("AVERAGED", help="wheter to analyse the avg. power or not",
                    type=int)

args = parser.parse_args()

# Index of the session to be load
tt = args.TT
br = args.BR
at = args.ALIGN
avg = args.AVERAGED

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser('~/funcog/gda')

###############################################################################
# Iterate over all sessions and concatenate power
###############################################################################

sxx = []
stim = []
for s_id in tqdm(sessions):
    _FILE_NAME = f"power_tt_{tt}_br_{br}_at_{at}.nc"
    path_pow = \
        os.path.join(_ROOT,
                     f"Results/lucy/{s_id}/session01",
                     _FILE_NAME)
    power = xr.load_dataarray(path_pow)
    # Averages power for each period (baseline, cue, delay, match) if needed
    out = average_stages(power, avg)

    sxx += [out.isel(roi=[r]) for r in range(len(out['roi']))]
    stim += [out.attrs["stim"].astype(int)]*len(out['roi'])


###############################################################################
# MI Workflow
###############################################################################

# Convert to DatasetEphy
dt = DatasetEphy(sxx, y=stim, nb_min_suj=10,
                 times="times", roi="roi")

mi_type = 'cd'
inference = 'rfx'
kernel = None

if avg:
    mcp = "fdr"
else:
    mcp = "cluster"

estimator = GCMIEstimator(mi_type='cd', copnorm=True,
                          biascorrect=True, demeaned=False, tensor=True,
                          gpu=False, verbose=None)
wf = WfMi(mi_type, inference, verbose=True, kernel=kernel, estimator=estimator)

kw = dict(n_jobs=20, n_perm=100)
cluster_th = None

mi, pvalues = wf.fit(dt, mcp=mcp, cluster_th=cluster_th, **kw)

###############################################################################
# Saving results
###############################################################################

# Path to results folder
_RESULTS = os.path.join(_ROOT,
                        "Results/lucy/mutual_information")

path_mi = os.path.join(_RESULTS,
                       f"mi_pow_tt_{tt}_br_{br}_aligned_{at}_avg_{avg}_{mcp}.nc")
path_tv = os.path.join(_RESULTS,
                       f"tval_pow_{tt}_br_{br}_aligned_{at}_avg_{avg}_{mcp}.nc")
path_pv = os.path.join(_RESULTS,
                       f"pval_pow_{tt}_br_{br}_aligned_{at}_avg_{avg}_{mcp}.nc")

mi.to_netcdf(path_mi)
wf.tvalues.to_netcdf(path_tv)
pvalues.to_netcdf(path_pv)

