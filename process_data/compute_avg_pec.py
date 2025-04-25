import sys

sys.path.insert(1, "/home/vinicius/storage1/projects/GrayData-Analysis")

import os
import xarray as xr

from config import get_dates
from tqdm import tqdm
from GDa.loader import loader
import argparse

_ROOT = os.path.expanduser("~/funcog/gda/")

parser = argparse.ArgumentParser()
parser.add_argument("MONKEY", help="which monkey to use", type=str)
args = parser.parse_args()

monkey = args.MONKEY

sessions = get_dates(monkey)

##############################################################
# PEC
# Different from power, PEC is already avereged for each epoch
##############################################################

data_loader = loader(_ROOT=_ROOT)

kw_loader = dict(aligned_at="cue", channel_numbers=False, monkey=monkey)

pec = []

for session in tqdm(sessions):

    pec += [
        xr.load_dataarray(
            os.path.join(_ROOT, "Results", monkey, "pec", f"pec_mat_{session}.nc")
        )
    ]

pec = xr.concat(pec, "sessions")

pec.to_netcdf(f"data/pec_{monkey}.nc")
