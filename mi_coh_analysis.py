"""
Edge-based encoding analysis done on the coherence dFC
"""
import os
import argparse

from tqdm import tqdm
from frites.dataset import DatasetEphy
from frites.estimator import GCMIEstimator
from frites.workflow import WfMi
from config import get_dates, return_delay_split
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
parser.add_argument("MONKEY", help="which monkey to use",
                    type=str)
parser.add_argument("ALIGNED", help="wheter power was align to cue or match",
                    type=str)
parser.add_argument("DELAY", help="which type of delay split to use",
                    type=int)

args = parser.parse_args()

metric = args.METRIC
avg = args.AVERAGED
at = args.ALIGNED
ds = args.DELAY
monkey = args.MONKEY

if not avg:
    ds = 0

early_cue, early_delay = return_delay_split(monkey=monkey, delay_type=ds)
print(early_cue, early_delay)
sessions = get_dates(monkey)

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser('~/funcog/gda')

###############################################################################
# Iterate over all sessions and concatenate coherece
###############################################################################

if metric == "coh":
    coh_file = f'{metric}_at_{at}.nc'
    coh_sig_file = f'thr_{metric}_at_{at}_surr.nc'
    tt = br = [1]
elif metric == "pec":
    coh_file = "pec_tt_1_br_1_at_cue.nc"
    coh_sig_file = None
    tt = br = None
print(coh_sig_file)

coh = []
stim = []
for s_id in tqdm(sessions):
    net = temporal_network(coh_file=coh_file, early_delay=early_delay,
                           early_cue=early_cue, align_to=at,
                           coh_sig_file=coh_sig_file, wt=None,
                           date=s_id, trial_type=tt, monkey=monkey,
                           behavioral_response=br)
    # Average if needed
    out = average_stages(net.super_tensor, avg, early_cue=early_cue,
                         early_delay=early_delay)
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

kw = dict(n_jobs=30, n_perm=200)
cluster_th = None  # {float, None, 'tfce'}

mi, pvalues = wf.fit(dt, mcp=mcp, cluster_th=cluster_th, **kw)

###############################################################################
# Saving results
###############################################################################

# Path to results folder
_RESULTS = os.path.join(_ROOT,
                        f"Results/{monkey}/mutual_information/coherence/")

path_mi = os.path.join(_RESULTS,
                       f"mi_{metric}_at_{at}_ds_{ds}_avg_{avg}_{mcp}.nc")
path_tv = os.path.join(_RESULTS,
                       f"tval_{metric}_at_{at}_ds_{ds}_avg_{avg}_{mcp}.nc")
path_pv = os.path.join(_RESULTS,
                       f"pval_{metric}_at_{at}_ds_{ds}_avg_{avg}_{mcp}.nc")

mi.to_netcdf(path_mi)
wf.tvalues.to_netcdf(path_tv)
pvalues.to_netcdf(path_pv)
