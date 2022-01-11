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
parser.add_argument("METRIC", help="wheter to plot power or coherence MI",
                    type=str)
parser.add_argument("THRESHOLD",
                    help="wheter to threshold the thresholded metric or not",
                    type=int)
args = parser.parse_args()

# Wheter to align to cue or match
metric = args.METRIC
thr = args.THRESHOLD

###########################################################################
# Load MI files
###########################################################################
# Define paths to read the files
_ROOT = "Results/lucy/mutual_information"
if metric == "power":
    _MI = os.path.join(_ROOT, "mi_pow_tt_1_br_1_aligned_cue_avg_1.nc")
    _PV = os.path.join(_ROOT, "pval_pow_1_br_1_aligned_cue_avg_1.nc")
    _TV = os.path.join(_ROOT, "tval_pow_1_br_1_aligned_cue_avg_1.nc")
else:
    _MI = os.path.join(_ROOT, f"mi_{metric}_avg_1_thr_{thr}.nc")
    _PV = os.path.join(_ROOT, f"pval_{metric}_avg_1_thr_{thr}.nc")
    _TV = os.path.join(_ROOT, f"t_{metric}_avg_1_thr_{thr}.nc")

mi = xr.load_dataarray(_MI)
p = xr.load_dataarray(_PV)
tv = xr.load_dataarray(_TV)
# Compute siginificant MI values
mi_sig = mi# * (p <= 0.05)

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
stage = ['baseline', 'cue', 'delay_e', 'delay_l', 'match']

# Create canvas in which the flatmaps will be drawn
fig = plt.figure(figsize=(8, 15), dpi=600)
gs1 = fig.add_gridspec(nrows=n_freqs, ncols=n_times,
                       left=0.05, right=0.87, bottom=0.05, top=0.95)
gs2 = fig.add_gridspec(nrows=n_freqs, ncols=1,
                       left=0.89, right=0.91, bottom=0.05, top=0.95)

# Will store the axes of the figure
ax, ax_cbar = [], []
# Plot flatmap for different freuquencies and times
for f in range(n_freqs):
    ax_cbar += [plt.subplot(gs2[f])]  # Colorbar axis
    vmin = None  # mi_sig.isel(freqs=f).min()
    vmax = None  # mi_sig.isel(freqs=f).max()
    for t in range(n_times):
        ax += [plt.subplot(gs1[t+n_times*f])]  # Flatmap axis
        # Get values to plot in the flatmap
        values = mi_sig.isel(times=t)
        values = values.isel(freqs=f).data
        # Delete values for subcortical areas
        values = np.delete(values, index)
        # Instantiate flatmap
        fmap = flatmap(values, _areas_nosca)
        # Only plot colorbar for last column
        if t == 3:
            fmap.plot(ax[t+n_times*f], ax_colorbar=ax_cbar[f],
                      cbar_title="MI [bits]",
                      colormap="hot_r", vmin=vmin, vmax=vmax)
        else:
            fmap.plot(ax[t+n_times*f], ax_colorbar=None,
                      cbar_title="MI [bits]",
                      colormap="hot_r", vmin=vmin, vmax=vmax)
        # Place titles
        if f == 0:
            plt.title(stage[t], fontsize=12)
plt.savefig(f"figures/flatmap_mi_{metric}_{thr}.png")
plt.close()
