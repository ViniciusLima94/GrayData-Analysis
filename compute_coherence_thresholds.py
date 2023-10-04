"""
Edge-based encoding analysis done on the coherence dFC
"""
import os
import argparse
import xarray as xr

from tqdm import tqdm
from config import get_dates


##############################################################################
# Argument parsing
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("METRIC", help="which connectivity metric to use", type=str)
parser.add_argument("MONKEY", help="which monkey to use", type=str)

args = parser.parse_args()

# The connectivity metric that should be used
metric = args.METRIC
# Wheter to use Lucy or Ethyl's data
monkey = args.MONKEY

sessions = get_dates(monkey)

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser(f"~/funcog/gda/Results/{monkey}")

###############################################################################
# Iterate over all sessions load surrogate coherence and compute threshold
###############################################################################
coh_file = f"{metric}_at_cue_surr.nc"
thr_file = f"thr_{metric}_at_cue_surr.nc"


# def _quantile(data, target='cpu', q=0.95, axis=0):
# if target == 'gpu':
# import cupy as cp
# data = cp.array(data)
# return cp.quantile(data, q, axis=axis)
# else:
# return np.quantile(data, q, axis=axis)

for s_id in ["141024"]:  # tqdm(sessions):

    _FILE_PATH = os.path.join(_ROOT, s_id, "session01")

    thr = xr.load_dataarray(os.path.join(_FILE_PATH, coh_file))
    roi, freqs, times = thr.roi.data, thr.freqs.data, thr.times.data
    attrs = thr.attrs

    # thr = _quantile(thr.data, target='cpu', axis=0)
    thr = thr.quantile(0.95, "trials")

    # thr = xr.DataArray(thr, dims=("roi", "freqs", "times"),
    # coords=(roi, freqs, times))
    thr.attrs = attrs

    thr.to_netcdf(os.path.join(_FILE_PATH, thr_file))
