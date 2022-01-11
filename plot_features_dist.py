"""
Plot the spacial distribution of the encoding. 
- blue=pow only
- red=coherence only
- green=both pow and coherence.
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import argparse

from tqdm import tqdm
from config import sessions
from GDa.util import create_stages_time_grid
from GDa.flatmap.flatmap import flatmap


# Define paths to read the files
_ROOT = "Results/lucy/mutual_information"

######################################################################
# p-values
######################################################################
p_power = xr.load_dataarray(
    os.path.join(_ROOT, 'pval_pow_1_br_1_aligned_cue_avg_1.nc'))
p_degree = xr.load_dataarray(
    os.path.join(_ROOT, 'pval_degree_avg_1_thr_1.nc'))
p_coreness = xr.load_dataarray(
    os.path.join(_ROOT, 'pval_coreness_avg_1_thr_1.nc'))
p_efficiency = xr.load_dataarray(
    os.path.join(_ROOT, 'pval_efficiency_avg_1_thr_1.nc'))

# Find where there are significant values
power = p_power <= 0.05
degree = p_degree <= 0.05
coreness = p_coreness <= 0.05
efficiency = p_efficiency <= 0.05

coh = 2 * ((degree+coreness+efficiency) > 0)

features = power + coh

###########################################################################
# Load MI files
###########################################################################

# Define sub-cortical areas names
sca = np.array(['thal', 'putamen', 'claustrum', 'caudate'])

# Get area names
areas = features.roi.data
areas = [a.lower() for a in areas]
index = np.where(np.isin(areas, sca))
_areas_nosca = np.delete(areas, index)

freqs = features.freqs.data
times = features.times.data
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
                       left=0.05, right=0.95, bottom=0.05, top=0.95)

# Will store the axes of the figure
ax, ax_cbar = [], []
# Plot flatmap for different freuquencies and times
for f in range(n_freqs):
    vmin = None  # mi_sig.isel(freqs=f).min()
    vmax = None  # mi_sig.isel(freqs=f).max()
    for t in range(n_times):
        ax += [plt.subplot(gs1[t+n_times*f])]  # Flatmap axis
        # Get values to plot in the flatmap
        values = features.isel(times=t)
        values = values.isel(freqs=f).data
        # Delete values for subcortical areas
        values = np.delete(values, index)
        # Instantiate flatmap
        fmap = flatmap(values, _areas_nosca)
        # Only plot colorbar for last column
        fmap.plot(ax[t+n_times*f], ax_colorbar=None,
                  cbar_title="MI [bits]",
                  colormap="hot_r", vmin=vmin, vmax=vmax)
        # Place titles
        if f == 0:
            plt.title(stage[t], fontsize=12)
plt.savefig(f"figures/flatmap_spatial.png")
plt.close()

