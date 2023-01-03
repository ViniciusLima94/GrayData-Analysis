"""
Functions to treat the inputs
"""
import numpy as np
import xarray as xr

from .util import _is_binary


def get_weights_params(A):
    is_weighted = not _is_binary(A)
    if is_weighted:
        weights = "weight"
    else:
        weights = None
    return is_weighted, weights


def _check_inputs(array, dims):
    """
    Check the input type and size.

    Parameters
    ----------
    array: array_lie
        The data array.
    dims: int
        The number of dimensions the array should have.
    """
    assert isinstance(dims, int)
    assert isinstance(array, (np.ndarray, xr.DataArray))
    assert len(array.shape) == dims


def _unwrap_inputs(array, concat_trials=False):
    """
    Unwrap array and its dimensions for further manipulation.

    Parameters
    ----------
    array: array_like
        The data array (roi,roi,trials,time).
    concat_trials: bool | False
        Wheter to concatenate or not trials of the values in the array.

    Returns
    -------
    array values concatenated or not and the values for each of its dimensions.
    """
    if isinstance(array, xr.DataArray):
        # Concatenate trials and time axis
        try:
            roi = array.sources.values
            trials = array.trials.values
            time = array.time.values
        except AttributeError:
            roi = np.arange(0, array.shape[0])
            trials = np.arange(0, array.shape[2])
            time = np.arange(0, array.shape[3])
        if concat_trials:
            array = array.stack(observations=("trials", "times"))
        array = array.values
    else:
        roi = np.arange(0, array.shape[0])
        trials = np.arange(0, array.shape[2])
        time = np.arange(0, array.shape[3])
        if concat_trials:
            array = array.reshape((len(roi), len(roi), len(trials)*len(time)))
    return array, roi, trials, time
