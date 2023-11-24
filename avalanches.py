""" Find temporal communities in the supraadjacency """
import os
import numpy as np
import xarray as xr
import networkx as nx
import igraph as ig
import pickle
import argparse
from brainconn.modularity import modularity_louvain_und
from frites.utils import parallel_func
from tqdm import tqdm
from GDa.loader import loader
from GDa.util import _extract_roi
from GDa.stats.bursting import find_start_end
from GDa.misc.downsample import downsample
from config import get_dates, freqs
import jax
import jax.numpy as jnp

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to be run", type=int)
parser.add_argument("THR", help="threshold to binarize power", type=int)
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument(
    "SURR", help="whether to compute for surrogate data or not", type=int
)
parser.add_argument("TTYPE", help="which trial type to use", choices=[1, 2], type=int)
parser.add_argument(
    "BEHAVIOR", help="which behavioral response to use", choices=[0, 1], type=int
)
parser.add_argument(
    "EPOCH",
    help="which stage to use",
    choices=["P", "S", "D1", "D2", "Dm", "all"],
    type=str,
)
parser.add_argument(
    "DECIM", help="decimation used to compute power",  type=int
)

args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX
thr = args.THR
monkey = args.MONKEY
surr = args.SURR
ttype = args.TTYPE
behav = args.BEHAVIOR
epoch = args.EPOCH
decim = args.DECIM

# To be sure of using the right parameter when using fixation trials
if ttype == 2:
    behav = 0

# Get session
sessions = get_dates(monkey)
s_id = sessions[idx]

# Root directory
_ROOT = os.path.expanduser("~/funcog/gda")

data_loader = loader(_ROOT=_ROOT)

stages = [[-0.5, -0.2], [0, 0.4], [0.5, 0.9], [0.9, 1.3], [1.1, 1.5], [-0.5, 2.0]]
stage_labels = ["P", "S", "D1", "D2", "Dm", "all"]
stages = dict(zip(stage_labels, stages))


###############################################################################
# Utility functions
###############################################################################
# Define your function that computes the outer product of a vector with itself
def outer_product(v):
    return jnp.outer(v, v)

# Vectorize the outer_product function using jax.vmap
vectorized_outer_product = jax.vmap(outer_product, in_axes=1, out_axes=0)


def get_area_mapping(unique_areas):
    area2idx = dict(zip(unique_areas, range(len(unique_areas))))
    return area2idx


def z_score(data):
    return (data - data.mean("times")) / data.std("times")


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


if __name__ == "__main__":

    # Get time range of the epoch analysed
    ti, tf = stages[epoch]
    # For loop over frequency bands
    for freq in freqs.astype(int):
        ######################################################################
        # Loading data
        ######################################################################
        kw_loader = dict(
            session=s_id, aligned_at="cue", channel_numbers=False,
            monkey=monkey, decim=decim
        )

        power = data_loader.load_power(
            **kw_loader, trial_type=ttype, behavioral_response=behav
        ).sel(freqs=freq)

        times_array = power.times.data
        trials_array = power.trials.data
        rois = power.roi.data
        attrs = power.attrs

        # dt
        dt = np.diff(times_array)[0]

        if ttype == 1:
            stim = power.attrs["stim"]
        else:
            stim = np.zeros_like(power.trials.data)

        # z-score power
        power = z_score(power)
        power.attrs = attrs

        # Binarize power
        raster = power >= thr
        # Summed activity
        rho = raster.sum("roi")
        # Get baseline level
        Q = rho.quantile(.7, "times")
        # Filter raster
        mask = rho >= Q

        # select epoch
        if epoch == "Dm":
            temp = []
            t_match_on = ( power.attrs["t_match_on"] - power.attrs["t_cue_on"] ) / 1000
            new_time_array = np.arange(-0.4, 0, dt)
            for i in range(power.sizes["trials"]):
                ti, tf = t_match_on[i] - .4, t_match_on[i]
                temp += [raster.sel(times=slice(ti, tf)).isel(trials=i)]
                temp[-1] = temp[-1].assign_coords({"times": new_time_array})
            temp = xr.concat(temp, "trials")
        else:
            raster = raster.sel(times=slice(ti, tf))

        # Downsample
        # if decim in [1, 5]:
            # _gamma = {1: 20, 5: 3}
            # raster = downsample(
                # raster.transpose("trials", "roi", "times"),
                # dt * _gamma[decim],
                # freqs=False,
            # ).transpose('roi', 'trials', 'times')


        # Get regions with time labels
        roi_time = []
        for t in range(power.sizes["times"]):
            roi_time += [f"{r}_{t}" for r in rois]
        roi_time = np.hstack(roi_time)

        ######################################################################
        # Compute co-crackling graphlets
        ######################################################################
        cc_graphlet = []
        trials_array = []
        stim_array = []
        for trial in tqdm(range(raster.sizes["trials"])):
            x = find_start_end(mask.isel(trials=trial).data)
            y = raster.isel(trials=trial)
            for ti, tf in x:
                temp = vectorized_outer_product(y.isel(times=slice(ti, tf)).data).mean(axis=0)
                cc_graphlet += [
                    xr.DataArray(temp, dims=("sources", "targets"), coords=(rois, rois))
                ]
                trials_array += [trial]
                stim_array = [stim[trial]]
        cc_graphlet = xr.concat(cc_graphlet, "trials").assign_coords({"trials": trials_array})
        cc_graphlet.attrs["stim"] = stim_array


        ######################################################################
        # Save results
        ######################################################################

        _SAVE = os.path.expanduser(f"~/funcog/gda/Results/{monkey}/avalanches")

        # Coavalanche and precedence
        fname = f"T_tt_{ttype}_br_{behav}_{epoch}_{s_id}_freq_{freq}_thr_{thr}_decim_{decim}_surr_{surr}.nc"
        cc_graphlet.to_netcdf(os.path.join(_SAVE, fname))
