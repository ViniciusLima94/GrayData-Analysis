"""
Plot the flatmaps of MI for the MIs estimated for the average power
over task stages
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from GDa.flatmap.flatmap import flatmap
import argparse

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("ALIGN", help="wheter to align data to cue or match",
                    type=str)
args = parser.parse_args()

# Wheter to align to cue or match
at = args.ALIGN

###########################################################################
# Load MI files
###########################################################################
# Define paths to read the files
_ROOT = "Results/lucy/mi_pow_rfx"
_MI = os.path.join(_ROOT, f"mi_pow_tt_1_br_1_aligned_{at}_avg_1.nc")
_PV = os.path.join(_ROOT, f"pval_pow_1_br_1_aligned_{at}_avg_1.nc")
_TV = os.path.join(_ROOT, f"tval_pow_1_br_1_aligned_{at}_avg_1.nc")

mi = xr.load_dataarray(_MI)
p = xr.load_dataarray(_PV)
tv = xr.load_dataarray(_TV)
# Compute siginificant MI values
mi_sig = mi * (p <= 0.05)

# Define sub-cortical areas names
sca = np.array(['thal', 'putamen', 'claustrum', 'caudate'])

# Get area names
areas = mi.roi.data
areas = [a.lower() for a in areas]
index = np.where(np.isin(areas, sca))
_areas_nosca = np.delete(areas, index)

freqs = mi.freqs.data
times = mi.times.data
n_freqs = len(freqs)
n_times = len(times)

###########################################################################
# Plot in the flatmap
###########################################################################

# Name of each stage to use in plot titles
stage = ['baseline', 'cue', 'delay', 'match']

# Create canvas in which the flatmaps will be drawn
fig = plt.figure(figsize=(8, 15), dpi=600)
gs1 = fig.add_gridspec(nrows=n_freqs, ncols=n_times,
                       left=0.05, right=0.87, bottom=0.05, top=0.95)
gs2 = fig.add_gridspec(nrows=n_freqs, ncols=1,
                       left=0.89, right=0.91, bottom=0.05, top=0.95)

# Will store the axes of the figure
ax, ax_cbar = [], []
# Plot flatmap for different freuquencies and times
for f_i, f in enumerate(range(n_freqs)):
    ax_cbar += [plt.subplot(gs2[f_i])]  # Colorbar axis
    for t_i, t in enumerate(range(n_times)):
        ax += [plt.subplot(gs1[t_i+n_times*f_i])]  # Flatmap axis
        # Get values to plot in the flatmap
        values = mi_sig.isel(times=t)
        values = values.isel(freqs=f).data
        # Delete values for subcortical areas
        values = np.delete(values, index)
        # Instantiate flatmap
        fmap = flatmap(values, _areas_nosca)
        # Only plot colorbar for last column
        if t == 3:
            fmap.plot(ax[t_i+n_times*f_i], ax_colorbar=ax_cbar[f_i],
                      cbar_title="MI [bits]",
                      colormap="hot_r", vmax=0.02)
        else:
            fmap.plot(ax[t_i+n_times*f_i], ax_colorbar=None,
                      cbar_title="MI [bits]",
                      colormap="hot_r", vmax=0.02)
        # Place titles
        if f == 0:
            plt.title(stage[t], fontsize=12)
plt.savefig(f"figures/flatmap_{at}.png")
plt.close()