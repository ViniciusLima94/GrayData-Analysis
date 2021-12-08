import os
import xarray as xr
import numpy as np
from frites.dataset import DatasetEphy
from frites.workflow import WfMi
from GDa.util import create_stages_time_grid
from tqdm import tqdm
import argparse

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
# Get root path and session names
###############################################################################

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')
# Get session number
sessions = np.loadtxt("GrayLab/lucy/sessions.txt", dtype=str)

###############################################################################
# Iterate over all sessions and concatenate power
###############################################################################


def average_stages(power, avg):
    """
    Loads the power DataArray and average it for each task
    stage if needed (avg=1) otherwise return the power itself
    (avg=0).
    """
    if avg == 1:
        out = []
        # Creates stage mask
        mask = create_stages_time_grid(power.attrs['t_cue_on'],
                                       power.attrs['t_cue_off'],
                                       power.attrs['t_match_on'],
                                       power.attrs['fsample'],
                                       power.times.data,
                                       power.sizes["trials"], align_to=at,
                                       flatten=False)
        for stage in mask.keys():
            mask[stage] = xr.DataArray(mask[stage], dims=('trials', 'times'),
                                       coords={"trials": power.trials.data,
                                               "times": power.times.data
                                               })
        for stage in mask.keys():
            # Number of observation in the specific stage
            n_obs = xr.DataArray(mask[stage].mean("times"), dims="trials",
                                 coords={"trials": power.trials.data})
            out += [(power * mask[stage]).sum("times") / n_obs]

        out = xr.concat(out, "times")
        out = out.transpose("trials", "roi", "freqs", "times")
        out.attrs = power.attrs
    else:
        out = power
    return out


###############################################################################
# Organizes the data to perform the rfx for the MI
###############################################################################
sxx = []
stim = []
for s_id in tqdm(sessions[:60]):
    _FILE_NAME = f"power_tt_{tt}_br_{br}_aligned_{at}.nc"
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
dt = DatasetEphy(sxx, y=stim, nb_min_suj=2,
                 times="times", roi="roi")

mi_type = 'cd'
inference = 'rfx'
kernel = None
wf = WfMi(mi_type, inference, verbose=True, kernel=kernel)

kw = dict(n_jobs=20, n_perm=200)
cluster_th = None  # {float, None, 'tfce'}

mi, pvalues = wf.fit(dt, mcp="fdr", cluster_th=cluster_th, **kw)

# Path to results foldeer
_RESULTS = "Results/lucy/mi_pow_rfx"

path_mi = os.path.join(_ROOT,
                       _RESULTS,
                       f"mi_pow_tt_{tt}_br_{br}_aligned_{at}_avg_{avg}.nc")
path_tv = os.path.join(_ROOT,
                       _RESULTS,
                       f"tval_pow_{tt}_br_{br}_aligned_{at}_avg_{avg}.nc")
path_pv = os.path.join(_ROOT,
                       _RESULTS,
                       f"pval_pow_{tt}_br_{br}_aligned_{at}_avg_{avg}.nc")

mi.to_netcdf(path_mi)
wf.tvalues.to_netcdf(path_tv)
pvalues.to_netcdf(path_pv)
