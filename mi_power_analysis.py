import os
import xarray as xr
import numpy as np
from frites.dataset import DatasetEphy
from frites.workflow import WfMi
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

args = parser.parse_args()

# Index of the session to be load
tt = args.TT
br = args.BR

###############################################################################
# Get root path and session names
###############################################################################

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')
# Get session number
sessions = np.loadtxt("GrayLab/lucy/sessions.txt", dtype=str)

###############################################################################
# Iterate over all sessions and concatenate power
###############################################################################

sxx = []
stim = []
roi = []
for s_id in tqdm(sessions[:60]):
    path_pow = \
        os.path.join(_ROOT,
                     f"Results/lucy/{s_id}/session01/power_tt_{tt}_br_{br}.nc")
    out = xr.load_dataarray(path_pow)
    sxx += [out.isel(roi=[r]) for r in range(len(out['roi']))]
    stim += [out.attrs["stim"].astype(int)]*len(out['roi'])
    roi += [out['roi'].data.tolist()]

roi = [item for sublist in roi for item in sublist]

###############################################################################
# MI Workflow
###############################################################################

# Convert to DatasetEphy
dt = DatasetEphy(sxx, y=stim, nb_min_suj=2,
                 times="times", roi="roi")

mi_type = 'cd'
inference = 'rfx'
kernel = np.hanning(1)
wf = WfMi(mi_type, inference, verbose=True, kernel=kernel)

kw = dict(n_jobs=20, n_perm=200)
cluster_th = None  # {float, None, 'tfce'}

mi, pvalues = wf.fit(dt, mcp="cluster", cluster_th=cluster_th, **kw)

path_mi = os.path.join(_ROOT,
                       f"Results/lucy/mi_pow_rfx/mi_pow_tt_{tt}_br_{br}.nc")
path_pv = os.path.join(_ROOT,
                       f"Results/lucy/mi_pow_rfx/pval_pow_{tt}_br_{br}.nc")

mi.to_netcdf(path_mi)
pvalues.to_netcdf(path_pv)
