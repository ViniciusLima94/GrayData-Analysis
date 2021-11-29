import os
import numpy as np
from tqdm import tqdm
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
for s_id in tqdm(sessions[:10]):
    # Instantiating temporal network
    net = temporal_network(coh_file='coh_k_0.3_morlet.nc',
                           # coh_sig_file='coh_k_0.3_morlet_surr.nc',
                           date=s_id, trial_type=[1],
                           behavioral_response=[1])

    # Convert to degree
    net.convert_to_adjacency()

    degree = net.A.sum("sources")

    del net.A

    degree = degree.rename({"targets": "roi"})

    degree \
        = degree.transpose("trials", "roi", "freqs", "times")

    coh += [degree.isel(roi=[r])
            for r in range(len(degree['roi']))]
    stim += [net.super_tensor.attrs["stim"].astype(int)] \
        * len(degree['roi'])

del net

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

for i in range(10):
    plt.subplot(2,5,i+1)
    idx = mi_sig.isel(freqs=i).sum("times")>0
    try:
        mi_sig.isel(freqs=i,roi=idx).plot(x="times", hue="roi")
    except AttributeError:
        continue
