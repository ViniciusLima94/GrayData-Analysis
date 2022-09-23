import os
import argparse
from tqdm import tqdm
import numpy as np
import xarray as xr
from config import get_dates
from GDa.util import average_stages
from GDa.temporal_network import temporal_network

parser = argparse.ArgumentParser()
parser.add_argument("MONKEY", help="which monkey to use",
                    type=str)

args = parser.parse_args()

monkey = args.MONKEY

sessions = get_dates(monkey)

coh_file = 'coh_at_cue.nc'
coh_sig_file = 'thr_coh_at_cue_surr.nc'

data = []
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
    data += [out.isel(roi=[r]).mean("trials")
             for r in range(len(out['roi']))]

# Concatenate channels
data = xr.concat(data, dim="roi")
# Get unique rois
urois, counts = np.unique(data.roi.data, return_counts=True)
# Get unique rois that has at leats 10 channels
urois = urois[counts >= 10]
# Average channels withn the same roi
data = data.groupby("roi").mean("roi", skipna=True)
data = data.sel(roi=urois)

save_path = os.path.expanduser(
    f"~/funcog/gda/Results/{monkey}/mean_coherences/mean_coh.nc")

data.to_netcdf(save_path)
