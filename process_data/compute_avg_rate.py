import sys

sys.path.insert(1, "/home/vinicius/storage1/projects/GrayData-Analysis")

import os
import xarray as xr
import scipy

from tqdm import tqdm
from config import get_dates
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


def get_power(monkey, session, decim=1, trial_type=1, behavioral_response=1):

    sessions = get_dates(monkey)

    kw_loader = dict(aligned_at="cue", channel_numbers=False, monkey=monkey, decim=1)

    temp = data_loader.load_power(
        **kw_loader,
        trial_type=trial_type,
        behavioral_response=behavioral_response,
        session=session,
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
            temp_stages += [temp.sel(times=slice(t0, t1)).isel(trials=i).mean("times")]
        out += [xr.concat(temp_stages, "times")]
    out = xr.concat(out, "trials")
    out = out.transpose("trials", "roi", "freqs", "times")

    return out


pvalues = []

for session in tqdm(sessions):

    power_task = get_power(monkey, session)
    power_fix = get_power(monkey, session, trial_type=2, behavioral_response=0)

    pvalues_ = scipy.stats.ttest_ind(power_task, power_fix, axis=0).pvalue < 0.001

    pvalues_ = xr.DataArray(
        pvalues_,
        dims=("roi", "freqs", "times"),
        coords={
            "roi": power_task.roi.values,
            "freqs": power_task.freqs.values,
            "times": power_task.times.values,
        },
    )

    pvalues += [pvalues_.groupby("roi").mean("roi")]


pvalues = xr.concat(pvalues, "sessions")
pvalues.to_netcdf(f"data/pvalues_{monkey}.nc")
