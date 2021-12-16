import os
import numpy as np
import xarray as xr
from tqdm import tqdm
from config import mode
from GDa.temporal_network import temporal_network
from frites.dataset import DatasetEphy
from frites.workflow import WfMi
import argparse

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("AVERAGED", help="wheter to analyse the avg. power or not",
                    type=int)

args = parser.parse_args()

# Index of the session to be load
avg = args.AVERAGED

_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')
# Get session number
sessions = np.loadtxt("GrayLab/lucy/sessions.txt", dtype=str)

###############################################################################
# Iterate over all sessions and concatenate coherece
###############################################################################


def average_stages(degrees, mask, avg):
    """
    Loads the temporal network and average it for each task
    stage if needed (avg=1) otherwise return the power itself
    (avg=0).
    """
    if avg == 1:
        out = []

        for stage in mask.keys():
            # Number of observation in the specific stage
            n_obs = xr.DataArray(mask[stage].mean("times"), dims="trials",
                                 coords={
                                     "trials": degrees.trials.data
                                 })
            out += [(degrees * mask[stage]).sum("times")/n_obs]

        out = xr.concat(out, "times")
        out = out.transpose("trials", "roi", "freqs", "times")
        out.attrs = degrees.attrs
    else:
        out = degrees.transpose("trials", "roi", "freqs", "times")

    return out


coh = []
stim = []
for s_id in tqdm(sessions[:60]):
    # Instantiating temporal network
    net = temporal_network(coh_file=f'coh_k_0.3_{mode}.nc',
                           coh_sig_file=f'coh_k_0.3_{mode}.nc',
                           date=s_id, trial_type=[1],
                           behavioral_response=[1])
    # Create adjacency matrix
    net.convert_to_adjacency()
    # Create masks
    net.create_stage_masks()
    s_mask = net.s_mask
    # Compute degrees
    degrees = net.A.sum("targets")
    degrees = degrees.rename({"sources": "roi"})
    degrees.attrs = net.super_tensor.attrs
    del net

    out = average_stages(degrees, s_mask, avg)
    del degrees

    coh += [out.isel(roi=[r])
            for r in range(len(out['roi']))]
    stim += [out.attrs["stim"].astype(int)] \
        * len(out['roi'])


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

kw = dict(n_jobs=20, n_perm=100)
cluster_th = None  # {float, None, 'tfce'}

mi, pvalues = wf.fit(dt, mcp="cluster", cluster_th=cluster_th, **kw)

path_mi = os.path.join(_ROOT,
                       f"Results/lucy/mi_pow_rfx/mi_coh_avg_{avg}.nc")
path_pv = os.path.join(_ROOT,
                       f"Results/lucy/mi_pow_rfx/pval_coh_avg_{avg}.nc")

mi.to_netcdf(path_mi)
pvalues.to_netcdf(path_pv)
