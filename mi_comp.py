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
from frites.workflow import WfStats
from tqdm import tqdm

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("METRIC", help="whether to use power or zpower", type=str)
parser.add_argument("ALIGN", help="wheter to align data to cue or match", type=str)
parser.add_argument(
    "AVERAGED", help="wheter to analyse the avg. power or not", type=int
)
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("SLVR", help="Whether to use SLVR channels or not", type=str)
parser.add_argument("NPERM", help="Number of permuations", type=int)

args = parser.parse_args()

# Index of the session to be load
metric = args.METRIC
at = args.ALIGN
avg = args.AVERAGED
monkey = args.MONKEY
slvr = args.SLVR
nperm = args.NPERM


stages = {}
stages["lucy"] = [[-0.5, -0.2], [0, 0.4], [0.5, 0.9], [0.9, 1.3], [1.1, 1.5]]
stages["ethyl"] = [[-0.5, -0.2], [0, 0.4], [0.5, 0.9], [0.9, 1.3], [1.1, 1.5]]
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
def sel_power(power, slvr, metric, avg):
    attrs = power.attrs
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
        for t0, t1 in stages[monkey]:
            out += [power.sel(times=slice(t0, t1)).mean("times")]
        out = xr.concat(out, "times")
        out = out.transpose("trials", "roi", "freqs", "times")
    else:
        out = power
    out.attrs = attrs

    return out


kw_loader = dict(aligned_at="cue", channel_numbers=False, monkey=monkey)

sxx_corr = []
sxx_incorr = []
stim_corr = []
stim_incorr = []
for s_id in tqdm(sessions):
    # Correct trials
    power_corr = data_loader.load_power(
        **kw_loader, trial_type=1, behavioral_response=1, session=s_id
    )
    attrs_corr = power_corr.attrs

    # Incorrect trials
    power_incorr = data_loader.load_power(
        **kw_loader, trial_type=1, behavioral_response=0, session=s_id
    )
    attrs_incorr = power_incorr.attrs

    power_corr = sel_power(power_corr, slvr, metric, avg)
    power_incorr = sel_power(power_incorr, slvr, metric, avg)

    sxx_corr += [power_corr.isel(roi=[r]) for r in range(len(power_corr["roi"]))]
    stim_corr += [power_corr.attrs["stim"].astype(int)] * len(power_corr["roi"])

    sxx_incorr += [power_incorr.isel(roi=[r]) for r in range(len(power_incorr["roi"]))]
    stim_incorr += [power_incorr.attrs["stim"].astype(int)] * len(power_incorr["roi"])


###############################################################################
# MI Workflow
###############################################################################
mi_type = "cd"
# inference = 'rfx'
inference = "ffx"
kernel = None
if avg:
    mcp = "fdr"
else:
    mcp = "cluster"


def run_single_mi(sxx, stim):

    # Convert to DatasetEphy
    dt = DatasetEphy(sxx, y=stim, nb_min_suj=10, times="times", roi="roi")

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

    kw = dict(n_jobs=30, n_perm=nperm)
    cluster_th = None

    mi, pvalues = wf.fit(dt, mcp=mcp, cluster_th=cluster_th, **kw)

    return wf, mi.dims, mi.coords


# Run MI for correct and incorrect trials
wf_corr, dims, coords = run_single_mi(sxx_corr, stim_corr)
wf_incorr, _, _ = run_single_mi(sxx_incorr, stim_incorr)

mi_a, mi_p_a = wf_corr.mi, wf_corr.mi_p
mi_b, mi_p_b = wf_incorr.mi, wf_incorr.mi_p

# subtract mi and mi_p
mi = [a - b for a, b in zip(mi_a, mi_b)]
mi_p = [a - b for a, b in zip(mi_p_a, mi_p_b)]

###############################################################################
# Stats
###############################################################################
kw_stats = dict(inference=inference, mcp=mcp, tail=1)
out, _ = WfStats().fit(mi, mi_p, **kw_stats)
out = xr.DataArray(out, dims=dims, coords=coords)


###############################################################################
# Saving results
###############################################################################

# Path to results folder
_RESULTS = os.path.join(_ROOT, f"Results/{monkey}/mutual_information/power/")
