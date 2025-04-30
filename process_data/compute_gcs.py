import sys

sys.path.insert(1, "/home/vinicius/storage1/projects/GrayData-Analysis")  # noqa

import os
import argparse

import numpy as np
import xarray as xr

from config import get_dates, bands


##############################################################################
# ARGUMENTS
##############################################################################

_ROOT = os.path.expanduser("~/funcog/gda/")

parser = argparse.ArgumentParser()
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("THR", help="threshold level used", type=int)
args = parser.parse_args()

monkey = args.MONKEY
thr = args.THR

sessions = get_dates(monkey)

stage_labels = ["P", "S", "D1", "D2", "Dm"]

freqs = np.median(bands[monkey], axis=1)

##############################################################################
# FUNCTIONS
##############################################################################


def return_CA_mat(
    _path_to_ava,
    session,
    epoch,
    freq=27,
    ttype=1,
    br=1,
    decim=1,
    surr=0,
    thr=None,
    thr_type="relative",
):

    path = os.path.join(
        _path_to_ava,
        f"T_tt_{ttype}_br_{br}_{epoch}_{session}_freq_{freq}_{thr_type}_thr_{thr}_decim_{decim}_surr_{surr}.nc",
    )
    T = xr.load_dataarray(path)

    return T


def get_gcs(monkey, surr=0, thr=80):

    _path_to_ava = os.path.expanduser(f"~/funcog/gda/Results/{monkey}/avalanches/")

    sessions = get_dates(monkey)

    CS = []

    for freq in freqs:
        T_all_sessions = []
        for session in sessions:
            T = []
            for epoch in stage_labels:
                T += [
                    return_CA_mat(
                        _path_to_ava,
                        session,
                        epoch,
                        freq=freq,
                        ttype=1,
                        br=1,
                        surr=surr,
                        thr=thr,
                    )  # .sum("targets")
                ]
            T_all_sessions += [xr.concat(T, "times")]
        CS += [xr.concat(T_all_sessions, "sessions")]

    CS = xr.concat(CS, "freqs")
    CS = CS.assign_coords({"freqs": freqs})
    # CS = CS.rename({"sources": "roi"})

    return CS


CS_l = get_gcs(monkey, surr=0, thr=thr)
CS_l.to_netcdf(f"data/CS_{monkey}_{thr}.nc")
