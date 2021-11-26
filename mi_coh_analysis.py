import os
import numpy as np
from GDa.temporal_network import temporal_network
from frites.dataset import DatasetEphy
from frites.workflow import WfMi

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')
# Get session number
sessions = np.loadtxt("GrayLab/lucy/sessions.txt", dtype=str)

###############################################################################
# Iterate over all sessions and concatenate coherece
###############################################################################

coh = []
stim = []
for s_id in sessions[:2]:
    # Instantiating temporal network
    net = temporal_network(coh_file='coh_k_0.3_morlet.nc',
                           # coh_sig_file='coh_k_0.3_morlet_surr.nc',
                           date=s_id, trial_type=[1],
                           behavioral_response=[1])

    net.super_tensor \
        = net.super_tensor.transpose("trials", "roi", "freqs", "times")

    coh += [net.super_tensor.isel(roi=[r])
            for r in range(len(net.super_tensor['roi']))]
    stim += [net.super_tensor.attrs["stim"].astype(int)] \
        * len(net.super_tensor['roi'])


###############################################################################
# MI Workflow
###############################################################################

# Convert to DatasetEphy
dt = DatasetEphy(coh, y=stim, nb_min_suj=2,
                 times="times", roi="roi")

mi_type = 'cd'
inference = 'rfx'
kernel = np.hanning(1)
wf = WfMi(mi_type, inference, verbose=True, kernel=kernel)

kw = dict(n_jobs=20, n_perm=200)
cluster_th = None  # {float, None, 'tfce'}

mi, pvalues = wf.fit(dt, mcp="cluster", cluster_th=cluster_th, **kw)

path_mi = os.path.join(_ROOT, "Results/lucy/mi_power_rfx/mi_coh.nc")
path_pv = os.path.join(_ROOT, "Results/lucy/mi_power_rfx/pval_coh.nc")

mi.to_netcdf(path_mi)
pvalues.to_netcdf(path_pv)
