import os
import xarray as xr
import argparse

from tqdm import tqdm
from config import sessions
from GDa.util import create_stages_time_grid
from frites.dataset import DatasetEphy
from frites.estimator import GCMIEstimator
from frites.workflow import WfMi

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
# Iterate over all sessions and concatenate coherece
###############################################################################


def average_stages(feature, avg):
    """
    Loads the network feature DataArray and average it for each task
    stage if needed (avg=1) otherwise return the feature itself
    (avg=0).
    """
    if avg == 1:
        out = []
        # Creates stage mask
        mask = create_stages_time_grid(feature.attrs['t_cue_on']-0.2,
                                       feature.attrs['t_cue_off'],
                                       feature.attrs['t_match_on'],
                                       feature.attrs['fsample'],
                                       feature.times.data,
                                       feature.sizes["trials"],
                                       early_delay=0.3,
                                       align_to="cue",
                                       flatten=False)
        for stage in mask.keys():
            mask[stage] = xr.DataArray(mask[stage], dims=('trials', 'times'),
                                       coords={"trials": feature.trials.data,
                                               "times": feature.times.data
                                               })
        for stage in mask.keys():
            # Number of observation in the specific stage
            n_obs = xr.DataArray(mask[stage].sum("times"), dims="trials",
                                 coords={"trials": feature.trials.data})
            out += [(feature * mask[stage]).sum("times") / n_obs]

        out = xr.concat(out, "times")
        out = out.transpose("trials", "roi", "freqs", "times")
        out.attrs = feature.attrs
    else:
        out = feature
    return out


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
dt = DatasetEphy(coh, y=stim, nb_min_suj=10,
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
                       f"mi_{metric}_avg_{avg}_thr_1.nc")
path_tv = os.path.join(_ROOT,
                       _RESULTS,
                       f"t_{metric}_avg_{avg}_thr_1.nc")
path_pv = os.path.join(_ROOT,
                       _RESULTS,
                       f"pval_{metric}_avg_{avg}_thr_1.nc")

mi.to_netcdf(path_mi)
wf.tvalues.to_netcdf(path_tv)
pvalues.to_netcdf(path_pv)
