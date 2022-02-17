"""
Plot the time-frequency plots of MI for the MIs estimated
for the average power over task stages
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
args = parser.parse_args()

# Wheter to align to cue or match
metric = args.METRIC

###########################################################################
# Load MI files
###########################################################################
# Define paths to read the files
_ROOT = "Results/lucy/mutual_information"
if metric == "power":
    _MI = os.path.join(_ROOT, "mi_pow_tt_1_br_1_aligned_cue_avg_0.nc")
    _PV = os.path.join(_ROOT, "pval_pow_1_br_1_aligned_cue_avg_0.nc")
    _TV = os.path.join(_ROOT, "tval_pow_1_br_1_aligned_cue_avg_0.nc")
else:
    _MI = os.path.join(_ROOT, f"mi_{metric}_avg_0_thr_1.nc")
    _PV = os.path.join(_ROOT, f"pval_{metric}_avg_0_thr_1.nc")
    _TV = os.path.join(_ROOT, f"t_{metric}_avg_0_thr_1.nc")

mi = xr.load_dataarray(_MI)
p = xr.load_dataarray(_PV)
tv = xr.load_dataarray(_TV)
# Compute siginificant MI values
mi_sig = tv * (p <= 0.05)

n_freqs, n_rois = mi_sig.sizes['freqs'], mi_sig.sizes['roi']
freqs, times = mi_sig.freqs.data, mi_sig.times.data
rois = mi_sig.roi.data
# Index of the sorted rois
iroi = np.argsort(np.char.lower(rois.astype(str)))
###########################################################################
# Creating plot 
###########################################################################
plt.figure(figsize=(15, 6))
for i in range(n_freqs):
    plt.subplot(2, 5, i+1)
    extent = [times[0], times[-1], 0, n_rois]
    plt.imshow(mi_sig.isel(freqs=i, roi=iroi).T, aspect="auto",
               cmap="turbo", origin="lower", extent=extent)
    plt.colorbar()
    plt.vlines(0, 0, n_rois, color='r', ls='--')
    plt.vlines(0.5, 0, n_rois, color='r', ls='--')
    plt.vlines(0.8, 0, n_rois, color='r', ls='--')
    plt.vlines(2, 0, n_rois, color='r', ls='--')
    if i == 0 or i == 5:
        plt.yticks(range(n_rois), rois[iroi], fontsize=4)
    else:
        plt.yticks([])
    if i < 5:
        plt.xticks([])
    plt.title(f'{freqs[i]:.0f} Hz', fontsize=12)
plt.savefig(f'figures/mi_{metric}_tf.png', dpi=200, bbox_inches='tight')
