"""
Plot the scatter of features at channel level over task stages
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import argparse

from tqdm import tqdm
from config import sessions
from GDa.util import create_stages_time_grid
from GDa.graphics.plot_brain_sketch import plot_node_brain_sketch

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("FEATURE", help="which feature should be ploted",
                    type=str)
parser.add_argument("PLOT", help="what to plot",
                    type=str)
args = parser.parse_args()

# Wheter to align to cue or match
feature = args.FEATURE
pl = args.PLOT

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')

###############################################################################
# Argument parsing
###############################################################################

assert feature in ["power",
                   "degree",
                   "coreness",
                   "efficiency"
                   "eccentricity"]

assert pl in ["mean",
              "cv",
              "95p"]

# Name of the file containing the feature
if feature == "power":
    _FILE_NAME = "power_tt_1_br_1_at_cue.nc"
else:
    _FILE_NAME = f"coh_{feature}_thr_1_at_cue.nc"

###############################################################################
# Loading feature
###############################################################################


def return_file_path(_ROOT, _FILE_NAME, s_id):
    path_metric = \
        os.path.join(_ROOT,
                     f"Results/lucy/{s_id}/session01",
                     _FILE_NAME)
    return path_metric


def average_stages(data):
    """
    Loads the data DataArray and average it for each task
    stage if needed (avg=1) otherwise return the data itself
    (avg=0).
    """
    out = []
    # Creates stage mask
    attrs = data.attrs
    mask = create_stages_time_grid(attrs['t_cue_on']-0.2,
                                   attrs['t_cue_off'],
                                   attrs['t_match_on'],
                                   attrs['fsample'],
                                   data.times.data,
                                   data.sizes["trials"],
                                   early_delay=0.3,
                                   align_to="cue",
                                   flatten=True)
    for stage in mask.keys():
        mask[stage] = xr.DataArray(mask[stage], dims=("observations"))

    data = data.stack(observations=("trials", "times"))
    for stage in mask.keys():
        aux = data.isel(observations=mask[stage])
        if pl == "mean":
            out += [aux.mean("observations", skipna=True)]
        elif pl == "95p":
            out += [aux.quantile(0.95, "observations",
                                 skipna=True)]
        elif pl == "cv":
            mu = aux.mean("observations", skipna=True)
            sig = aux.std("observations", skipna=True)
            out += [sig / mu]

    out = xr.concat(out, "times")
    out.attrs = attrs
    return out


# Load data for all sessions
data = []
for s_id in tqdm(sessions):
    # Get path to file
    path_metric = return_file_path(_ROOT, _FILE_NAME, s_id)
    # Load network feature
    out = xr.load_dataarray(path_metric)
    # Average if needed
    out = average_stages(out)
    # Reassing roi coord
    out = out.assign_coords({'roi': out.attrs['channels_labels']})
    # Save in data
    data += [out]

# Group channels and average over the same channel labels
data = xr.concat(data, 'roi')
data = data.groupby('roi').mean('roi', skipna=True)

freqs = data.freqs.data
times = data.times.data
channel_labels = data.roi.data
n_freqs = len(freqs)
n_times = len(times)
n_rois = len(channel_labels)

###############################################################################
#  Plotting brain skecth
###############################################################################

# Name of each stage to use in plot titles
stage = ['baseline', 'cue', 'delay_e', 'delay_l', 'match']

# Create canvas in which the flatmaps will be drawn
fig = plt.figure(figsize=(8, 15), dpi=600)
gs1 = fig.add_gridspec(nrows=n_freqs, ncols=n_times,
                       left=0.05, right=0.93, bottom=0.05, top=0.95)

# Will store the axes of the figure
ax, ax_cbar = [], []
# Plot flatmap for different freuquencies and times
for f in range(n_freqs):
    for t in range(n_times):
        ax += [plt.subplot(gs1[t+n_times*f])]  # Flatmap axis
        # Get values to plot in the flatmap
        features = data.isel(times=t, freqs=f).data
        # Plot in the brain skect
        plot_node_brain_sketch(channel_labels,
                               features,
                               None,
                               None,
                               'hot_r',
                               size=10,
                               sketch=True)
        # Place titles
        if f == 0:
            plt.title(stage[t], fontsize=12)
plt.savefig(f"figures/scatter_mi_{feature}_{pl}.png")
plt.close()
