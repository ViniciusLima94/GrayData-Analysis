import os
import xarray as xr
import numpy as np
from frites.dataset import DatasetEphy
from frites.workflow import WfMi

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')
# Get session number
sessions = np.loadtxt("GrayLab/lucy/sessions.txt", dtype=str)

###############################################################################
# Iterate over all sessions and concatenate power
###############################################################################

sxx = []
stim = []
roi = []
for s_id in sessions[:60]:
    path_pow = os.path.join(_ROOT, f"Results/lucy/{s_id}/session01/power.nc")
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

path_mi = os.path.join(_ROOT, "Results/lucy/mi_power_rfx/mi_power.nc")
path_pv = os.path.join(_ROOT, "Results/lucy/mi_power_rfx/pval_power.nc")

mi.to_netcdf(path_mi)
pvalues.to_netcdf(path_pv)
