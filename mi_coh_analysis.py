"""
Edge-based encoding analysis done on the coherence dFC
"""
import os
import argparse

from tqdm import tqdm
from frites.dataset import DatasetEphy
from frites.estimator import GCMIEstimator
from frites.workflow import WfMi
from config import sessions
from GDa.util import average_stages
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

args = parser.parse_args()

metric = args.METRIC
avg = args.AVERAGED
surr = args.SURR

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
    net = temporal_network(coh_file=coh_file,
                           coh_sig_file=coh_sig_file, wt=None,
                           date=s_id, trial_type=[1],
                           behavioral_response=[1])
    # Average if needed
    out = average_stages(net.super_tensor, avg)
    # To save memory
    del net
    # Convert to format required by the MI workflow
    coh += [out.isel(roi=[r])
            for r in range(len(out['roi']))]
    stim += [out.attrs["stim"].astype(int)] \
        * len(out['roi'])

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
                       f"mi_{metric}_avg_{avg}_{mcp}.nc")
path_tv = os.path.join(_RESULTS,
                       f"tval_{metric}_avg_{avg}_{mcp}.nc")
path_pv = os.path.join(_RESULTS,
                       f"pval_{metric}_avg_{avg}_{mcp}.nc")

mi.to_netcdf(path_mi)
wf.tvalues.to_netcdf(path_tv)
pvalues.to_netcdf(path_pv)
