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

_ROOT = os.path.expanduser('~/funcog/gda/')

###############################################################################
# Iterate over all sessions and concatenate power
###############################################################################

# Source power
sxx_s = []
# Target power
sxx_t = []
# Stimulus
stim = []
# ROI names
rois = []
for s_id in tqdm(sessions):
    _FILE_NAME = f"power_tt_{tt}_br_{br}_at_{at}.nc"
    path_pow = \
        os.path.join(_ROOT,
                     f"Results/lucy/{s_id}/session01",
                     _FILE_NAME)
    power = xr.load_dataarray(path_pow)
    # Averages power for each period (baseline, cue, delay, match) if needed
    out = average_stages(power, avg)
    # Re-order sources and targets
    sources = out.attrs['sources']
    targets = out.attrs['targets']
    n_pairs = len(sources)

    sxx_s += [out.isel(roi=[s]).data for s in sources]
    sxx_t += [out.isel(roi=[t]).data for t in targets]
    stim += [out.attrs["stim"].astype(int)] * n_pairs
    # Save roi names
    roi_s, roi_t = out.roi[sources].data, out.roi[targets].data
    rois += [f"{s}-{t}" for s, t in zip(roi_s, roi_t)]
    # Time array
    times = out.times.data


###############################################################################
# MI Workflow
###############################################################################

# Convert to DatasetEphy
dt = DatasetEphy(x=sxx_s, y=sxx_t, z=stim, nb_min_suj=10,
                 times=times, roi=rois)

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

###############################################################################
# Saving results
###############################################################################

# Path to results folder
_RESULTS = os.path.expanduser("Results/lucy/mutual_information")

path_mi = os.path.join(_ROOT,
                       _RESULTS,
                       f"mi_3_pow_tt_{tt}_br_{br}_aligned_{at}_avg_{avg}.nc")
path_tv = os.path.join(_ROOT,
                       _RESULTS,
                       f"tval_3_pow_{tt}_br_{br}_aligned_{at}_avg_{avg}.nc")
path_pv = os.path.join(_ROOT,
                       _RESULTS,
                       f"pval_3_pow_{tt}_br_{br}_aligned_{at}_avg_{avg}.nc")

mi.to_netcdf(path_mi)
wf.tvalues.to_netcdf(path_tv)
pvalues.to_netcdf(path_pv)
