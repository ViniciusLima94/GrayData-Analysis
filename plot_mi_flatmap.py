import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from GDa.flatmap.flatmap import flatmap

###########################################################################
# Load MI files
###########################################################################
mi = xr.load_dataarray("Results/lucy/mi_pow_rfx/mi_pow_tt_1_br_1.nc")
p = xr.load_dataarray("Results/lucy/mi_pow_rfx/pval_pow_1_br_1.nc")
# Compute siginificant MI values
mi_sig = mi * (p <= 0.05)

# Define sub-cortical areas names
sca = np.array(['thal', 'putamen', 'claustrum', 'caudate'])

areas = mi.roi.data
areas = [a.lower() for a in areas]
index = np.where(np.isin(areas, sca))
_areas_nosca = np.delete(areas, index)

# Time windows to integrate MI
t_win = np.array([[0, 0.5],
                  [0.5, 1.0],
                  [1.0, 1.5],
                  [1.5, 2.0],
                  [2.0, 2.5]])

n_freqs = mi.sizes["freqs"]
freqs = mi.freqs.data

count = 1
for f in range(n_freqs):
    for t, (t0, t1) in enumerate(t_win):
        fig = plt.figure(figsize=(20, 15), dpi=150)
        values = mi_sig.sel(times=slice(t0, t1)).mean("times")
        values = values.isel(freqs=f).data
        values = np.delete(values, index)
        fmap = flatmap(values, _areas_nosca)
        fmap.plot(cbar_title="MI [bits]", colorbar=True, fig=fig, colormap="hot_r", vmax=0.02)
        plt.title(f"f={freqs[f]} Hz, t={t0}-{t1} s")
        plt.savefig(f"figures/{count}.png")
        plt.close()
        count += 1
