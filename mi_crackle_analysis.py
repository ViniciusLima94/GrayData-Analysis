import os
import xarray as xr
import argparse

from config import get_dates, return_delay_split
from frites.dataset import DatasetEphy
from frites.estimator import GCMIEstimator
from frites.workflow import WfMi
from GDa.util import average_stages
from tqdm import tqdm

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("TT", help="type of the trial", type=int)
parser.add_argument("BR", help="behavioral response", type=int)
parser.add_argument("Q", help="threshold used to binarize the data", type=float)
parser.add_argument("ALIGN", help="wheter to align data to cue or match", type=str)
parser.add_argument("MONKEY", help="which monkey to use", type=str)

args = parser.parse_args()

# Index of the session to be load
tt = args.TT
br = args.BR
q = args.Q
at = args.ALIGN
monkey = args.MONKEY


stages = {}
stages["lucy"] = [[-0.4, 0], [0, 0.4], [0.5, 0.9], [0.9, 1.3], [1.1, 1.5]]
stages["ethyl"] = [[-0.4, 0], [0, 0.4], [0.5, 0.9], [0.9, 1.3], [1.1, 1.5]]
stage_labels = ["P", "S", "D1", "D2", "Dm"]

sessions = get_dates(monkey)

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser("~/funcog/gda")

###############################################################################
# Iterate over all sessions and concatenate power
###############################################################################

sxx = []
stim = []
for s_id in tqdm(sessions):
    # Load power
    _FILE_NAME = f"power_tt_{tt}_br_{br}_at_{at}.nc"
    path_pow = os.path.join(_ROOT, f"Results/{monkey}/{s_id}/session01", _FILE_NAME)
    power = xr.load_dataarray(path_pow)
    attrs = power.attrs

    # Compute and apply thresholds
    thr = power.quantile(q, ("trials", "times"))
    power.values = (power >= thr).values

    out = []
    for t0, t1 in stages[monkey]:
        out += [power.sel(times=slice(t0, t1)).mean("times")]
    out = xr.concat(out, "times")
    out = out.transpose("trials", "roi", "freqs", "times")
    out.attrs = attrs

    sxx += [out.isel(roi=[r]) for r in range(len(out["roi"]))]
    stim += [out.attrs["stim"].astype(int)] * len(out["roi"])


###############################################################################
# MI Workflow
###############################################################################

# Convert to DatasetEphy
dt = DatasetEphy(sxx, y=stim, nb_min_suj=10, times="times", roi="roi")

mi_type = "cd"
inference = "rfx"
kernel = None

mcp = "fdr"

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

qint = int(q * 100)

path_mi = os.path.join(
    _RESULTS, f"mi_crackle_{tt}_br_{br}_aligned_{at}_q_{qint}_{mcp}.nc"
)
path_tv = os.path.join(
    _RESULTS, f"tval_crackle_{tt}_br_{br}_aligned_{at}_q_{qint}_{mcp}.nc"
)
path_pv = os.path.join(
    _RESULTS, f"pval_crackle_{tt}_br_{br}_aligned_{at}_q_{qint}_{mcp}.nc"
)

mi.to_netcdf(path_mi)
wf.tvalues.to_netcdf(path_tv)
pvalues.to_netcdf(path_pv)
