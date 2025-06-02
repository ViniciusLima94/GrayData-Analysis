import sys

sys.path.insert(1, "/home/vinicius/storage1/projects/GrayData-Analysis")

import os
import xarray as xr

from config import get_dates
from tqdm import tqdm
from GDa.loader import loader
import argparse
import scipy
from mne.stats import fdr_correction

_ROOT = os.path.expanduser("~/funcog/gda/")

parser = argparse.ArgumentParser()
parser.add_argument("MONKEY", help="which monkey to use", type=str)
args = parser.parse_args()

monkey = args.MONKEY
tt = args.TT
br = args.BR

sessions = get_dates(monkey)

##############################################################
# POWER
##############################################################

data_loader = loader(_ROOT=_ROOT)

kw_loader = dict(aligned_at="cue", channel_numbers=False, monkey=monkey)


def load_epoched_power(session, tt, br, decim):

    kw_loader = dict(aligned_at="cue", channel_numbers=False, monkey=monkey, decim=1)

    temp = data_loader.load_power(
        **kw_loader, trial_type=tt, behavioral_response=br, session=session
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

    return out.transpose("roi", "freqs", "trials", "times")


def get_incidences(monkey, decim=1):

    incidences = []

    for session in tqdm(sessions):

        power_task = load_epoched_power(session, 1, 1, 10)
        power_fix = load_epoched_power(session, 2, 0, 10)

        pvalues = fdr_correction(
            scipy.stats.mannwhitneyu(power_task.data, power_fix.data, axis=2).pvalue,
            0.001,
        )[0]

        pvalues = xr.DataArray(
            pvalues,
            dims=("roi", "freqs", "times"),
            coords={"roi": power_task.roi.values},
            name=session,
        )

        incidences += [pvalues.groupby("roi").mean("roi")]

    incidences = xr.concat(incidences, "sessions")

    return incidences


incidences = get_incidences(monkey)
incidences.to_netcdf(f"data/incidences_{monkey}.nc")
