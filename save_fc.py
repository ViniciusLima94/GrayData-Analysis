import argparse
import os
import numpy as np
import xarray as xr


from GDa.loader import loader
from config import freqs, get_dates

from frites.conn import conn_io
from frites.io import set_log_level
from frites.utils import parallel_func

from tqdm import tqdm


###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()


parser.add_argument("SIDX", help="index of the session to load", type=int)
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("ALIGNED", help="wheter power was align to cue or match", type=str)
parser.add_argument("THR", help="which threshold value to use", type=int)

args = parser.parse_args()

sid = args.SIDX
monkey = args.MONKEY
at = args.ALIGNED
thr = args.THR


###############################################################################
# Funtions
###############################################################################


def binFC(
    data,
    q,
    ttype="relative",
    sfreq=None,
    roi=None,
    times=None,
    pairs=None,
    n_jobs=-1,
    verbose=None,
    dtype=np.float32,
):
    set_log_level(verbose)

    # _________________________________ INPUTS ________________________________
    # inputs conversion
    data, cfg = conn_io(
        data,
        times=times,
        roi=roi,
        agg_ch=False,
        win_sample=None,
        block_size=None,
        freqs=None,
        verbose=verbose,
        name="crackle to burst index",
    )

    if ttype == "relative":
        thr = data.quantile(q, dim=("times"))
    elif ttype == "absolute":
        thr = data.quantile(q, dim=("roi", "times"))

    data = (data >= thr).astype(bool)

    # extract variables
    x = data.data
    times = data["times"].data
    x_s, x_t, roi_p = cfg["x_s"], cfg["x_t"], cfg["roi_p"]
    n_pairs = len(x_s)

    def _loop(i_s, i_t):
        s, t = x[:, i_s], x[:, i_t]

        cc = (s * t).mean()
        return cc

    kw_para = dict(n_jobs=n_jobs, verbose=verbose, total=n_pairs)
    # define the function to compute in parallel
    parallel, p_fun = parallel_func(_loop, **kw_para)

    out = parallel(p_fun(s, t) for s, t in zip(x_s, x_t))

    out = xr.DataArray(np.stack(out), dims=("roi"), coords=(roi_p,))

    # compute the single trial coherence
    return out


def resample_trials(data, seed=0):

    np.random.seed(seed)
    X = data.values

    T, R, N = X.shape  # Extract dimensions
    # Generate a shuffled set of trial indices for each row
    sampled_trials = np.array([np.random.permutation(T) for _ in range(T)])

    # Create the new array where each ROI gets data from a different trial
    resampled_X = np.zeros((T, R, N))  # Placeholder for sampled data

    for row in range(T):
        trial_indices = sampled_trials[row]  # Get shuffled trials for this row
        for roi in range(R):
            resampled_X[row, roi, :] = X[trial_indices[roi], roi, :]

    resampled_X = xr.DataArray(
        resampled_X,
        dims=data.dims,
        coords={"times": data.times.values, "roi": data.roi.values},
    )

    return resampled_X


###############################################################################
# Setting configuration
###############################################################################


_ROOT = os.path.expanduser("~/funcog/gda/")
_SAVE = os.path.join(_ROOT, "Results", monkey, "FC")

sessions = get_dates(monkey)


###############################################################################
# Loading power
###############################################################################


data_loader = loader(_ROOT=_ROOT)
sid = sessions[sid]
kw_loader = dict(
    session=sid,
    aligned_at=at,
    channel_numbers=True,
    monkey=monkey,
    decim=1,
    mode="morlet",
)


POWER = (
    data_loader.load_power(**kw_loader, trial_type=1, behavioral_response=1)
    .sel(times=slice(-0.5, 2))
    .transpose("trials", "roi", "times", "freqs")
)

FCcrk, FCbst = [], []
FCcrk_thr, FCbst_thr = [], []

for freq in freqs:

    power = POWER.sel(freqs=freq)

    power_surorgates = [resample_trials(power, i + 1000) for i in tqdm(range(10))]

    ################################### Data #################################
    FCcrk += [
        binFC(
            power,
            thr / 100,
            "relative",
            1000,
            roi="roi",
            times="times",
            n_jobs=10,
            verbose=False,
        )
    ]

    FCbst += [
        binFC(
            power,
            thr / 100,
            "absolute",
            1000,
            roi="roi",
            times="times",
            n_jobs=10,
            verbose=False,
        )
    ]

    ################################ Surrogates ##############################
    FCcrk_surrogates = [
        binFC(
            power_,
            thr / 100,
            "relative",
            1000,
            roi="roi",
            times="times",
            n_jobs=10,
            verbose=False,
        )
        for power_ in tqdm(power_surorgates)
    ]

    FCbst_surrogates = [
        binFC(
            power_,
            thr / 100,
            "absolute",
            1000,
            roi="roi",
            times="times",
            n_jobs=10,
            verbose=False,
        )
        for power_ in tqdm(power_surorgates)
    ]

    FCcrk_thr += [xr.concat(FCcrk_surrogates, "boot").quantile(0.95, "boot")]
    FCbst_thr += [xr.concat(FCbst_surrogates, "boot").quantile(0.95, "boot")]

FCcrk = xr.concat(FCcrk, "freqs").assign_coords({"freqs": freqs})
FCbst = xr.concat(FCbst, "freqs").assign_coords({"freqs": freqs})
FCcrk_thr = xr.concat(FCcrk_thr, "freqs").assign_coords({"freqs": freqs})
FCbst_thr = xr.concat(FCbst_thr, "freqs").assign_coords({"freqs": freqs})

FCcrk.to_netcdf(os.path.join(_SAVE, f"FC_crackle_{sid}_at_{at}_q_{thr}.nc"))
FCbst.to_netcdf(os.path.join(_SAVE, f"FC_burst_{sid}_at_{at}_q_{thr}.nc"))
FCcrk_thr.to_netcdf(os.path.join(_SAVE, f"FC_crackle_thr_{sid}_at_{at}_q_{thr}.nc"))
FCbst_thr.to_netcdf(os.path.join(_SAVE, f"FC_burst_thr_{sid}_at_{at}_q_{thr}.nc"))
