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
# POWER
##############################################################

data_loader = loader(_ROOT=_ROOT)

kw_loader = dict(aligned_at="cue", channel_numbers=False, monkey=monkey)


def get_power(monkey, decim=1):

    sessions = get_dates(monkey)

    powers = []

    for session in tqdm(sessions):
        kw_loader = dict(
            aligned_at="cue", channel_numbers=False, monkey=monkey, decim=1
        )

        temp = data_loader.load_power(
            **kw_loader, trial_type=1, behavioral_response=1, session=session
        )[..., ::decim]

        t_match_on = (temp.attrs["t_match_on"] - temp.attrs["t_cue_on"]) / 1000

        out = []
        for i in range(temp.sizes["trials"]):
            stages = [
                [-0.5, -0.2],
                [0, 0.4],
                [0.5, 0.9],
                [0.9, 1.3],
                [t_match_on[i] - 0.4, t_match_on[i]],
            ]
            temp_stages = []
            for t0, t1 in stages:
                temp_stages += [
                    temp.sel(times=slice(t0, t1)).isel(trials=i).mean("times")
                ]
            out += [xr.concat(temp_stages, "times")]
        out = xr.concat(out, "trials").mean("trials")
        out = out.transpose("roi", "freqs", "times").groupby("roi").mean("roi")
        powers += [out]

    powers = xr.concat(powers, "sessions")

    return powers


power = get_power(monkey)
power.to_netcdf(f"data/power_{monkey}.nc")
