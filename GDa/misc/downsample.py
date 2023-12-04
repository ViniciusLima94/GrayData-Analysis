import numpy as np
import xarray as xr
from frites.conn import define_windows

def downsample(data, slwin_len, min_events=0, freqs=False):
    """
    Downsample a 4D data array by sliding a window over it and setting it
    to one if any value is one inside the window.

    Parameters:
    ----------
    data : xarray.DataArray
        The input 4D data array with dimensions ("trials", "roi", "freqs", "times").
    slwin_len : float
        The length of the sliding window in seconds.
    min_events: int
        Minimum number of events inside the window required to consider it as active.
    freqs : bool, optional
        If True, the data array includes a "freqs" dimension; otherwise, it only
        has "trials" and "roi" dimensions.

    Returns:
    -------
    xarray.DataArray
        A downsampled data array with dimensions ("trials", "roi", "freqs", "times") or
        ("trials", "roi", "times") if 'freqs' is False. Each time point within the sliding
        window is set to one if any value is one within that window.

    Note:
    ----
    This function takes a 4D data array and a sliding window length as input. It slides
    the window over the time dimension and sets each time point within the window to one
    if any value in the original data array is one within that window.

    Example:
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> data = xr.DataArray(np.random.randint(0, 2, size=(10, 5, 3, 100)),
    ...                    dims=("trials", "roi", "freqs", "times"))
    >>> slwin_len = 1.0  # 1-second sliding window
    >>> downsampled_data = downsample(data, slwin_len)
    """

    if freqs:
        _dims = ("trials", "roi", "freqs", "times")
    else:
        _dims = ("trials", "roi", "times")

    assert isinstance(data, xr.DataArray)
    np.testing.assert_array_equal(data.dims, _dims)

    if freqs:
        ntrials, nrois, nfreqs, ntimes = data.shape
        freqs = data.freqs.values
    else:
        ntrials, nrois, ntimes = data.shape

    trials, rois, times = (
        data.trials.values,
        data.roi.values,
        data.times.values,
    )

    win, twin = define_windows(times, slwin_len=slwin_len)

    if freqs:
        _coords = (trials, rois, freqs, twin)
        _shape = (ntrials, nrois, nfreqs, len(twin))
    else:
        _coords = (trials, rois, twin)
        _shape = (ntrials, nrois, len(twin))

    data_array = data.values
    data_ds = np.zeros(_shape, dtype=int)

    for pos, (t_i, t_f) in enumerate(win):
        data_ds[..., pos] = data_array[..., t_i:t_f].sum(axis=-1) > min_events

    data_ds = xr.DataArray(data_ds, dims=_dims, coords=_coords)

    return data_ds
