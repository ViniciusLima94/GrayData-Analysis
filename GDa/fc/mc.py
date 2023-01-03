import numpy as np
import xarray as xr

from frites.utils import parallel_func


def np_import(target):
    if target == "cpu":
        import numpy as xp
    elif target == "gpu":
        import cupy as xp
    return xp


def _mc(FC, target="cpu"):
    """
    Core fucntion for the meta-connectivity

    Parameters:
    ----------
    FC: array_like
        Functional connectivity time-series (edges, times)

    Returns:
    -------
    MC: array_like
        Meta-connectivity matrix (roi, roi, times)
    """
    assert target in ["cpu", "gpu"]
    xp = np_import(target)
    CC = xp.corrcoef(FC)
    return np.nan_to_num(CC, copy=False)


def meta_conn(FC, mask=None, n_jobs=1, dtype=np.float32, verbose=False):
    """
    Computes the meta-connectivity for a tensor of shape
    (roi, trials, time)

    Parameters:
    ----------
    FC: array_like
        Functional connectivity time-series (edges, trials, times).
    n_jobs: int | 1
        Number of jobs to use when parallelizing in observations.
    target: string | "cpu"
        Wheter to do computations on cpu or gpu
    Returns:
    -------
    MC: array_like
        Meta-connectivity matrix (roi, roi, freqs)
    """
    assert FC.ndim == 3

    # Get dimensions
    n_rois, n_trials, n_times = FC.shape
    masked = isinstance(mask, dict)

    def _for_trial(i):
        if masked:
            out = []
            for key in mask.keys():
                out += [_mc(FC[:, i, mask[key][i]]).astype(dtype)]
            out = np.stack(out, 0)
        else:
            out = _mc(FC[:, i, :]).astype(dtype)
        return out

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_trial, n_jobs=n_jobs, verbose=verbose, total=n_trials
    )
    # Compute the single trial coherence
    MC = parallel(p_fun(t) for t in range(n_trials))
    # Convert to numpy array
    MC = np.asarray(MC).T

    # If it an xarray get coords
    if isinstance(FC, xr.DataArray):
        np.testing.assert_equal(FC.dims[:2], ("roi", "trials"))
        trials, roi = FC.trials.data, FC.roi.data
        attrs = FC.attrs

        if masked:
            dims = ("sources", "targets", "times", "trials")
        else:
            dims = ("sources", "targets", "trials")

        MC = xr.DataArray(
            MC,
            dims=dims,
            coords={"sources": roi, "targets": roi, "trials": trials},
        )
        MC.attrs = attrs
    return MC
