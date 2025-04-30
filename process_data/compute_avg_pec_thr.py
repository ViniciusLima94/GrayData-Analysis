import sys

sys.path.insert(1, "/home/vinicius/storage1/projects/GrayData-Analysis")

import os
import numpy as np
import xarray as xr

from config import get_dates
from tqdm import tqdm
from GDa.loader import loader
import argparse

_ROOT = os.path.expanduser("~/funcog/gda/")

parser = argparse.ArgumentParser()
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("THR", help="which threshold to use", type=str)
args = parser.parse_args()

monkey = args.MONKEY
thr = args.THR

sessions = get_dates(monkey)

# Extract area names
def _extract_roi(roi, sep):
    # Code by Etiene
    x_s, x_t = [], []
    for r in roi:
        _x_s, _x_t = r.split(sep)
        x_s.append(_x_s), x_t.append(_x_t)
    roi_c = np.c_[x_s, x_t]
    idx = np.argsort(np.char.lower(roi_c.astype(str)), axis=1)
    roi_s, roi_t = np.c_[[r[i] for r, i in zip(roi_c, idx)]].T
    return roi_s, roi_t


##############################################################
# PEC
# Different from power, PEC is already avereged for each epoch
##############################################################

data_loader = loader(_ROOT=_ROOT)

kw_loader = dict(aligned_at="cue", channel_numbers=False, monkey=monkey)

pec_str = []

for session in tqdm(sessions):

    # Load pec
    pec = xr.load_dataarray(
        os.path.join(
            _ROOT,
            "Results",
            monkey,
            session,
            "session01",
            f"pec_1_br_1_at_cue_thr_{thr}.nc",
        )
    )

    edges = pec.roi.values

    sources, targets = _extract_roi(edges, "-")

    unique_rois = np.unique(np.hstack((sources, targets)))

    coordination = []

    for roi in unique_rois:

        indexes = np.logical_or(roi == sources, roi == targets)
        coordination += [pec.isel(roi=indexes).sum("roi").mean("trials")]

    pec_str += [xr.concat(coordination, "roi").transpose(*pec.dims[1:])]
    pec_str[-1] = pec_str[-1].assign_coords({"roi": unique_rois})

pec_str = xr.concat(pec_str, "sessions")

pec_str.to_netcdf(f"data/pec_str_{monkey}_thr_{thr}.nc")
