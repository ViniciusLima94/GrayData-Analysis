import os
import numpy as np
import xarray as xr
import argparse

from config import get_dates, return_delay_split
from frites.dataset import DatasetEphy
from frites.estimator import GCMIEstimator
from frites.workflow import WfMi
from GDa.util import average_stages
from GDa.session import session_info
from GDa.loader import loader
from tqdm import tqdm

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("METRIC", help="whether to use power or zpower", type=str)
parser.add_argument("TT", help="type of the trial", type=int)
parser.add_argument("BR", help="behavioral response", type=int)
parser.add_argument("ALIGN", help="wheter to align data to cue or match", type=str)
parser.add_argument(
    "AVERAGED", help="wheter to analyse the avg. power or not", type=int
)
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("SLVR", help="Whether to use SLVR channels or not", type=str)

args = parser.parse_args()

# Index of the session to be load
metric = args.METRIC
tt = args.TT
br = args.BR
at = args.ALIGN
avg = args.AVERAGED
monkey = args.MONKEY
slvr = args.SLVR


stage_labels = ["P", "S", "D1", "D2", "Dm"]

assert metric in ["pow", "zpow"]

sessions = get_dates(monkey)

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser("~/funcog/gda")
data_loader = loader(_ROOT=_ROOT)

###############################################################################
# Iterate over all sessions and concatenate power
###############################################################################

kw_loader = dict(aligned_at="cue", channel_numbers=False, monkey=monkey)

sxx = []
stim = []
for s_id in tqdm(sessions):
    power = data_loader.load_power(
        **kw_loader, trial_type=tt, behavioral_response=br, session=s_id,
        decim=5
    )
    attrs = power.attrs

    t_match_on = ( power.attrs["t_match_on"] - power.attrs["t_cue_on"] ) / 1000

    # Remove SLVR channels
    if not bool(slvr):
        info = session_info(
            raw_path=os.path.join(_ROOT, "GrayLab"),
            monkey=monkey,
            date=s_id,
            session=1,
        )
        slvr_idx = info.recording_info["slvr"].astype(bool)
        power = power.isel(roi=np.logical_not(slvr_idx))

    if metric == "zpow":
        power = (power - power.mean("times")) / power.std("times")

    # Average epochs
    out = []
    if avg:
        for i in range(power.sizes["trials"]):
            stages = [[-0.5, -0.2], [0, 0.4], [0.5, 0.9], [0.9, 1.3],
                      [t_match_on[i] - .4, t_match_on[i]]]
            temp = []
            for t0, t1 in stages:
                temp += [power.sel(times=slice(t0, t1)).isel(trials=i).mean("times")]
            out += [xr.concat(temp, "times")]
        out = xr.concat(out, "trials")
        out = out.transpose("trials", "roi", "freqs", "times")
    else:
        out = power
    out.attrs = attrs
    sxx += [out.isel(roi=[r]) for r in range(len(out["roi"]))]
    stim += [out.attrs["stim"].astype(int)] * len(out["roi"])


###############################################################################
# MI Workflow
###############################################################################

# Convert to DatasetEphy
dt = DatasetEphy(sxx, y=stim, nb_min_suj=10, times="times", roi="roi")

mi_type = "cd"
# inference = 'rfx'
inference = "ffx"
kernel = None

if avg:
    mcp = "fdr"
else:
    mcp = "cluster"

mi_type = "cd"

estimator = GCMIEstimator(
    mi_type="cd",
    copnorm=True,
    biascorrect=True,
    demeaned=False,
    tensor=True,
    gpu=False,
    verbose=None,
)
wf = WfMi(mi_type, inference, verbose=True, kernel=kernel, estimator=estimator)

kw = dict(n_jobs=30, n_perm=200)
cluster_th = None

mi, pvalues = wf.fit(dt, mcp=mcp, cluster_th=cluster_th, **kw)

###############################################################################
# Saving results
###############################################################################

# Path to results folder
_RESULTS = os.path.join(_ROOT, f"Results/{monkey}/mutual_information/power/")

# path_mi = os.path.join(_RESULTS,
# f"mi_pow_tt_{tt}_br_{br}_aligned_{at}_ds_{ds}_avg_{avg}_{mcp}.nc")
# path_tv = os.path.join(_RESULTS,
# f"tval_pow_{tt}_br_{br}_aligned_{at}_ds_{ds}_avg_{avg}_{mcp}.nc")
# path_pv = os.path.join(_RESULTS,
# f"pval_pow_{tt}_br_{br}_aligned_{at}_ds_{ds}_avg_{avg}_{mcp}.nc")

path_mi = os.path.join(
    _RESULTS,
    f"mi_{metric}_tt_{tt}_br_{br}_aligned_{at}_avg_{avg}_{mcp}_{inference}_slvr_{slvr}.nc",
)
path_tv = os.path.join(
    _RESULTS,
    f"tval_{metric}_{tt}_br_{br}_aligned_{at}_avg_{avg}_{mcp}_{inference}_slvr_{slvr}.nc",
)
path_pv = os.path.join(
    _RESULTS,
    f"pval_{metric}_{tt}_br_{br}_aligned_{at}_avg_{avg}_{mcp}_{inference}_slvr_{slvr}.nc",
)

mi.to_netcdf(path_mi)
pvalues.to_netcdf(path_pv)
if inference == "rfx":
    wf.tvalues.to_netcdf(path_tv)
