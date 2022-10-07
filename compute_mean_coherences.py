import os
import argparse
from tqdm import tqdm
import numpy as np
import xarray as xr
from config import get_dates, return_delay_split
from GDa.util import average_stages
from GDa.temporal_network import temporal_network

parser = argparse.ArgumentParser()
parser.add_argument("METRIC", help="which metric to use",
                    type=str)
parser.add_argument("MONKEY", help="which monkey to use",
                    type=str)
parser.add_argument("ALIGNED", help="wheter power was align to cue or match",
                    type=str)
parser.add_argument("DELAY", help="which type of delay split to use",
                    type=int)

args = parser.parse_args()

metric = args.METRIC
monkey = args.MONKEY
at = args.ALIGNED
ds = args.DELAY

sessions = get_dates(monkey)
early_cue, early_delay = return_delay_split(monkey=monkey, delay_type=ds)

coh_file = 'coh_at_cue.nc'
coh_sig_file = 'thr_coh_at_cue_surr.nc'
tt = br = [1]
if metric == "pec":
    coh_file = "pec_tt_1_br_1_at_cue.nc"
    coh_sig_file = None
    tt = br = None

data = []
for s_id in tqdm(sessions):
    net = temporal_network(coh_file=coh_file, early_delay=early_delay,
                           early_cue=early_cue, align_to=at,
                           coh_sig_file=coh_sig_file, wt=None,
                           date=s_id, trial_type=tt, monkey=monkey,
                           behavioral_response=br)
    # Average if needed
    out = average_stages(net.super_tensor, 1, early_cue=early_cue,
                         early_delay=early_delay)
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
    f"~/funcog/gda/Results/{monkey}/mean_coherences/mean_{metric}_at_{at}_ds_{ds}.nc")

data.to_netcdf(save_path)
