import numpy as np
import xarray as xr
import os

# Get session number
sessions = np.loadtxt("GrayLab/lucy/sessions.txt", dtype=str)

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')

p = []
for s_id in sessions[:12]:
    path_st = os.path.join(_ROOT,
                           f"Results/lucy/{s_id}/session01/power_p_values.nc")
    p_val = xr.load_dataarray(path_st)
    p += [p_val.groupby("roi").mean("roi")]
