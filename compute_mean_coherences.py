import os
from itertools import compress
from tqdm import tqdm
import numpy as np
import xarray as xr
from config import sessions
from GDa.util import average_stages
from GDa.temporal_network import temporal_network


def get_rois(xrlist):
    rois = []
    for out in xrlist:
        rois += [out.roi.data[0]]
    return np.array(rois)


coh_file = 'coh_at_cue.nc'
coh_sig_file = 'thr_coh_at_cue_surr.nc'

coh = []
for s_id in tqdm(sessions):
    net = temporal_network(coh_file=coh_file,
                           coh_sig_file=coh_sig_file, wt=None,
                           date=s_id, trial_type=[1],
                           behavioral_response=[1])
    # Average if needed
    out = average_stages(net.super_tensor, 1)
    # To save memory
    del net
    # Convert to format required by the MI workflow
    coh += [out.isel(roi=[r])
            for r in range(len(out['roi']))]

rois = get_rois(coh)

out = []
for roi in tqdm(np.unique(rois.data)):
    idx = rois == roi
    out += [xr.concat(list(compress(coh, idx)), "trials")]

rois = get_rois(out)

save_path = os.path.expanduser("~/funcog/gda/Results/lucy/mean_coherences")

for i in range(len(out)):
    file_names = f"mean_coh_at_cue_{i}.nc"
    path = os.path.join(save_path, file_names)
    out[i].to_netcdf(path)
