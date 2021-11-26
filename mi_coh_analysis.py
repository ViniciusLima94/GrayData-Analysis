import os
import numpy as np
from GDa.temporal_network import temporal_network
from frites.dataset import DatasetEphy
from frites.workflow import WfMi
# import matplotlib.pyplot as plt
import argparse

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX",   help="index of the session to run",
                    type=int)

args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')
# Get session number
sessions = np.loadtxt("GrayLab/lucy/sessions.txt", dtype=str)
s_id = sessions[idx]

###############################################################################
# Loading coherence dFC
###############################################################################

# Instantiating temporal network
net = temporal_network(coh_file='coh_k_0.3_morlet.nc',
                       coh_sig_file='coh_k_0.3_morlet_surr.nc',
                       date=s_id, trial_type=[1],
                       behavioral_response=[1])

net.super_tensor \
    = net.super_tensor.transpose("trials", "roi", "freqs", "times")

###############################################################################
# Iterate over all sessions and concatenate power
###############################################################################

coh = [net.super_tensor.isel(roi=[r])
       for r in range(len(net.super_tensor['roi']))]
stim = [net.super_tensor.attrs["stim"].astype(int)] \
        * len(net.super_tensor['roi'])


###############################################################################
# MI Workflow
###############################################################################

# Convert to DatasetEphy
dt = DatasetEphy([coh], y=stim, nb_min_suj=2,
                 times="times", roi="roi")

mi_type = 'cd'
inference = 'rfx'
kernel = np.hanning(1)
wf = WfMi(mi_type, inference, verbose=True, kernel=kernel)

kw = dict(n_jobs=20, n_perm=200)
cluster_th = None  # {float, None, 'tfce'}

mi, pvalues = wf.fit(dt, mcp="cluster", cluster_th=cluster_th, **kw)

# Setting roi names for mi and pvalues
# mi = mi.assign_coords({"roi": net.super_tensor.roi.values})
# pvalues = pvalues.assign_coords({"roi": net.super_tensor.roi.values})

###############################################################################
# Saving data
###############################################################################

path_st = os.path.join(_ROOT, f"Results/lucy/{s_id}/session01")
path_mi = os.path.join(path_st, "coh_mi_values.nc")
path_pval = os.path.join(path_st, "coh_p_values.nc")

# Not storing for storage issues
# mi.to_netcdf(path_mi)
# pvalues.to_netcdf(path_pval)

# Thresholded p-values
# p_values_thr = pvalues <= 0.05

# p_values_thr.to_netcdf(path_pval)

###############################################################################
# Plotting values with p<=0.05
###############################################################################

# out = mi * p_values_thr

# plt.figure(figsize=(20, 6))
# for i in range(10):
    # plt.subplot(2, 5, i+1)
    # plt.imshow(out.isel(freqs=i), aspect="auto", origin="lower",
               # cmap="turbo")
    # plt.colorbar()
# plt.tight_layout()

# plt.savefig(f"figures/mi_coh/mi_coh_{s_id}.png", dpi=150)
