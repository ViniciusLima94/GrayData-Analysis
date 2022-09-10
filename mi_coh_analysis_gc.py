"""
Edge-based encoding analysis done on the coherence dFC
"""
import os
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from frites.dataset import DatasetEphy
from frites.estimator import GCMIEstimator
from frites.workflow import WfMi
from config import sessions
from GDa.util import average_stages, _extract_roi
from GDa.temporal_network import temporal_network

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
parser.add_argument("SURR",
                    help="wheter to use the surrogate coherence or not",
                    type=int)
parser.add_argument("TRIALS",
                    help="which kind of trials to use",
                    type=int)

args = parser.parse_args()

metric = args.METRIC
avg = args.AVERAGED
surr = args.SURR
tt = args.TRIALS

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser('~/funcog/gda')

###############################################################################
# Iterate over all sessions and concatenate coherece
###############################################################################
if not surr:
    coh_file = f'{metric}_at_cue.nc'
    coh_sig_file = f'thr_{metric}_at_cue_surr.nc'
else:
    coh_file = f'{metric}_at_cue_surr.nc'
    coh_sig_file = None


coh = []
stim = []
for s_id in tqdm(sessions):
    # Load FP matrix
    Om = pd.read_csv(
        f"/home/vinicius/funcog/gda/Results/lucy/ghost_coherence/om_coh_{s_id}.csv"
    )

    net = temporal_network(coh_file=coh_file,
                           coh_sig_file=coh_sig_file, wt=None,
                           date=s_id, trial_type=[1],
                           behavioral_response=[1])
    # Get sources and targets
    sources = net.super_tensor.attrs["sources"].astype(int)
    targets = net.super_tensor.attrs["targets"].astype(int)
    # Channel labels
    channels = net.super_tensor.attrs["channels_labels"]
    chn_s, chn_t = channels[sources], channels[targets]
    roi_s, roi_t = _extract_roi(net.super_tensor["roi"].data, "-")

    # Average if needed
    out = average_stages(net.super_tensor.sel(freqs=slice(25, 40)), avg)
    # To save memory
    del net

    for r in range(len(out['roi'])):
        roi_key = f"{roi_s[r]} ({chn_s[r]})-{roi_t[r]} ({chn_t[r]})"
        x = Om.loc[(Om.freqs == 2) & (Om.roi == roi_key)]
        x = np.stack((x.trials.values, x.Om.values), 1)
        idx = np.logical_not(np.logical_or(x[:, 1] == 1, x[:, 1] == 2))
        if np.sum(idx) <= 0:
            continue
        trials = x[idx][:, 0]
        coh += [out.isel(roi=[r]).sel(trials=trials)]
        stim += [out.attrs["stim"].astype(int)[idx]]

###############################################################################
# MI Workflow
###############################################################################

# Convert to DatasetEphy
dt = DatasetEphy(coh, y=stim, nb_min_suj=10,
                 times="times", roi="roi")

mi_type = 'cd'
inference = 'rfx'

if avg:
    mcp = "fdr"
else:
    mcp = "cluster"

kernel = None
estimator = GCMIEstimator(mi_type='cd', copnorm=True,
                          biascorrect=True, demeaned=False, tensor=True,
                          gpu=False, verbose=None)
wf = WfMi(mi_type, inference, verbose=True, kernel=kernel, estimator=estimator)

kw = dict(n_jobs=20, n_perm=200)
cluster_th = None  # {float, None, 'tfce'}

mi, pvalues = wf.fit(dt, mcp=mcp, cluster_th=cluster_th, **kw)

###############################################################################
# Saving results
###############################################################################

# Path to results folder
_RESULTS = os.path.join(_ROOT,
                        "Results/lucy/mutual_information/coherence/")

path_mi = os.path.join(_RESULTS,
                       f"mi_{metric}_avg_{avg}_{mcp}_tt_{tt}.nc")
path_tv = os.path.join(_RESULTS,
                       f"tval_{metric}_avg_{avg}_{mcp}_tt_{tt}.nc")
path_pv = os.path.join(_RESULTS,
                       f"pval_{metric}_avg_{avg}_{mcp}_tt_{tt}.nc")

mi.to_netcdf(path_mi)
wf.tvalues.to_netcdf(path_tv)
pvalues.to_netcdf(path_pv)
