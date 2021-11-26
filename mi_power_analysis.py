import os
import xarray as xr
import numpy as np
from frites.dataset import DatasetEphy
from frites.workflow import WfMi
# import matplotlib.pyplot as plt

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

# # Setting roi names for mi and pvalues
# mi = mi.assign_coords({"roi": data.roi.values})
# pvalues = pvalues.assign_coords({"roi": data.roi.values})

###############################################################################
# Saving data
###############################################################################

# path_st = os.path.join(_ROOT, f"Results/lucy/{s_id}/session01")
# path_mi = os.path.join(path_st, "power_mi_values.nc")
# path_pval = os.path.join(path_st, "power_p_values.nc")

# Not storing for storage issues
# mi.to_netcdf(path_mi)
# pvalues.to_netcdf(path_pval)

# p_values_thr = pvalues <= 0.05
# p_values_thr.to_netcdf(path_pval)

###############################################################################
# Plotting values with p<=0.05
###############################################################################
# out = mi * p_values_thr

# plt.figure(figsize=(20, 6))
# for i in range(10):
# plt.subplot(2, 5, i+1)
# idx = out.isel(freqs=i).sum("times") > 0
# out.isel(freqs=i, roi=idx).plot(x="times", hue="roi")
# plt.legend([])
# plt.tight_layout()

# plt.figure(figsize=(20, 6))
# for i in range(10):
# plt.subplot(2, 5, i+1)
# plt.imshow(out.isel(freqs=i).T,
# cmap="turbo", aspect="auto", origin="lower",
# extent=[out.times.values[0], out.times.values[-1], 0, 57])
# plt.yticks(range(57), out.roi.values)
# plt.title(f"f={out.freqs.values[i]}")


# plt.savefig(f"figures/mi_power/mi_power_{s_id}.png", dpi=150)
