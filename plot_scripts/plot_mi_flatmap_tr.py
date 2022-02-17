"""
Plot the flatmaps of MI for the MIs estimated for the time-series of power
over task stages
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
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
# _MI = os.path.join(_ROOT, f"mi_pow_tt_1_br_1_aligned_{at}_avg_0.nc")
# _PV = os.path.join(_ROOT, f"pval_pow_1_br_1_aligned_{at}_avg_0.nc")
# _TV = os.path.join(_ROOT, f"tval_pow_1_br_1_aligned_{at}_avg_0.nc")
_MI = os.path.join(_ROOT, "mi_coh_avg_0.nc")
_PV = os.path.join(_ROOT, "pval_coh_avg_0.nc")

mi = xr.load_dataarray(_MI)
p = xr.load_dataarray(_PV)
# tv = xr.load_dataarray(_TV)
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


def create_canvas():
    """
    Create canvas in which the flatmaps will be drawn
    """
    fig = plt.figure(figsize=(8, 3), dpi=300)
    gs1 = fig.add_gridspec(nrows=2, ncols=int(n_freqs/2),
                           left=0.05, right=0.95, bottom=0.05, top=0.87)
    # gs2 = fig.add_gridspec(nrows=n_freqs, ncols=1,
                           # left=0.89, right=0.91, bottom=0.05, top=0.95)
    return fig, gs1#, gs2


# Plot flatmap for different freuquencies and times
for t in tqdm(range(n_times)):
    # One canvas per time stamp
    fig, gs1 = create_canvas()
    # Will store the axes of the figure
    ax, ax_cbar = [], []
    for f in range(n_freqs):
        ax += [plt.subplot(gs1[f])]  # Flatmap axis
        # ax_cbar += [plt.subplot(gs2[f_i])]  # Colorbar axis
        # Get values to plot in the flatmap
        values = mi_sig.isel(times=t)
        values = values.isel(freqs=f).data
        # Delete values for subcortical areas
        values = np.delete(values, index)
        # Instantiate flatmap
        fmap = flatmap(values, _areas_nosca)
        # Only plot colorbar for last column
        fmap.plot(ax[f], #ax_colorbar=ax_cbar[f_i],
                  cbar_title="MI [bits]",
                  colormap="hot_r", vmax=0.01)
        # Place titles
        plt.title(f"f={freqs[f]} Hz", fontsize=8)
    plt.suptitle(f"t = {np.round(times[t],3)} s", fontsize=10)
    plt.savefig(f"figures/flatmaps/flatmap_{at}_t_{t}.png")
    plt.close()
