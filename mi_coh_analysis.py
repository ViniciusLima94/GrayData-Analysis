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


def average_stages(net, avg):
    """
    Loads the temporal network and average it for each task
    stage if needed (avg=1) otherwise return the power itself
    (avg=0).
    """
    if avg == 1:
        out = []
        # Creates stage mask
        net.create_stage_masks(flatten=False)
        mask = net.s_mask

        for stage in mask.keys():
            # Number of observation in the specific stage
            n_obs = xr.DataArray(mask[stage].mean("times"), dims="trials",
                                 coords={
                                     "trials": net.super_tensor.trials.data
                                 })
            out += [net.get_data_from(stage=stage,
                                      pad=True).unstack().sum("times")/n_obs]

        out = xr.concat(out, "times")
        out = out.transpose("trials", "roi", "freqs", "times")
        out.attrs = net.super_tensor.attrs
    else:
        out = net.super_tensor.transpose("trials", "roi", "freqs", "times")

    return out


coh = []
stim = []
for s_id in tqdm(sessions[:10]):
    # Instantiating temporal network
    net = temporal_network(coh_file=f'coh_k_0.3_{mode}.nc',
                           # coh_sig_file='coh_k_0.3_morlet_surr.nc',
                           date=s_id, trial_type=[1],
                           behavioral_response=[1])

    out = average_stages(net, avg)

    del net

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

kw = dict(n_jobs=20, n_perm=200)
cluster_th = None  # {float, None, 'tfce'}

mi, pvalues = wf.fit(dt, mcp="fdr", cluster_th=cluster_th, **kw)

path_mi = os.path.join(_ROOT,
                       f"Results/lucy/mi_pow_rfx/mi_coh_avg_{avg}.nc")
path_pv = os.path.join(_ROOT,
                       f"Results/lucy/mi_pow_rfx/pval_coh_avg_{avg}.nc")

mi.to_netcdf(path_mi)
pvalues.to_netcdf(path_pv)
