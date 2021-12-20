import numpy  as np 
import xarray as xr
import numba  as nb

from   mne.filter                       import filter_data
from   frites.utils                     import parallel_func
from   frites.conn.conn_sliding_windows import define_windows
from   frites.dataset                   import SubjectEphy
from   frites.io                        import set_log_level, logger

def z_score(x):
    r'''
    Computes the z-score of data in the format ("trials","roi","times").
    - x: Input data-tensor.
    '''
    assert isinstance(x, (np.ndarray,xr.DataArray))

    if isinstance(x, np.ndarray):
        return (x-x.mean(-1)[...,None])/x.std(-1)[...,None]
    elif isinstance(x, xr.DataArray):
        assert "times" in x.dims
        return (x-x.mean(dim="times"))/x.std(dim="times")

def conn_correlation(data, times=None, roi=None, sfreq=None, f_low=None, f_high=None, pairs=None, win_sample=None, decim=None, 
        block_size=None, verbose=False, n_jobs=1):
    """ 
    Computes co-fluctuation time-series between rois (elementwise product of the z_score time-series) 
    for data in the format ("trials","roi","times").
    Parameters
    ----------
    data : array_like
        Electrophysiological data. Several input types are supported :
            * Standard NumPy arrays of shape (n_epochs, n_roi, n_times)
            * mne.Epochs
            * xarray.DataArray of shape (n_epochs, n_roi, n_times)
    roi : array_like | None
        ROI names of a single subject. If the input is an xarray, the
        name of the ROI dimension can be provided
    times : array_like | None
        Time vector array of shape (n_times,). If the input is an xarray, the
        name of the time dimension can be provided
    sfreq : float
        The sampling frequency
    f_low: float
        Lower frequency for filtering signal.
    f_high:
        Upper frequency for filtering the signal.
    pairs : array_like | None
        Pairs of contacts
    win_sample : array_like | None
        Array of shape (n_windows, 2) in order to take the mean of the
        estimated coherence inside sliding windows. You can use the function
        :func:`frites.conn.define_windows` to define either manually either
        sliding windows.
    decim : int | 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition. default 1 If int, returns tfr[…, ::decim]. If slice,
        returns tfr[…, decim].
    block_size : int | None
        Number of blocks of trials to process at once. This parameter can be
        use in order to decrease memory load. If None, all trials are used. If
        for example block_size=2, the number of trials are subdivided into two
        groups and each group is process one after the other.
    n_jobs : int | 1
        Number of jobs to use for parallel computing (use -1 to use all
        jobs). The parallel loop is set at the pair level.
    Returns
    -------
    dFC : xarray.DataArray
        DataArray of shape (n_trials, n_pairs, n_freqs, n_times)
    """ 

    set_log_level(verbose)

    # ___________________________________ I/O _________________________________
    if isinstance(data, xr.DataArray):
        trials, attrs = data[data.dims[0]].data, data.attrs
    else:
        trials, attrs = np.arange(data.shape[0]), {}

    # internal conversion
    data = SubjectEphy(data, y=trials, roi=roi, times=times, sfreq=sfreq)
    x, roi, times = data.data, data['roi'].data, data['times'].data
    trials, sfreq = data['y'].data, data.attrs['sfreq']
    n_trials, n_roi, n_times = data.shape

    # get the sorted non-directed pairs and build roi pairs names
    if (pairs is None):
        x_s, x_t = np.triu_indices(n_roi, k=1)
    else:
        assert isinstance(pairs, np.ndarray)
        assert (pairs.ndim == 2) and (pairs.shape[1] == 2)
        x_s, x_t = pairs[:, 0], pairs[:, 1]
    roi_s, roi_t = np.sort(np.c_[roi[x_s], roi[x_t]], axis=1).T
    roi_p   = [f"{s}-{t}" for s, t in zip(roi_s, roi_t)]
    n_pairs = len(roi_p)

    # build block size indices
    if isinstance(block_size, int) and (block_size > 1):
        indices = np.array_split(np.arange(n_trials), block_size)
    else:
        indices = [np.arange(n_trials)]

    # temporal mean
    need_mean = isinstance(win_sample, np.ndarray) and win_sample.shape[1] == 2
    if need_mean:
        if decim > 2:
            win_sample = np.floor(win_sample / decim).astype(int)
        times = times[win_sample].mean(1)

    # show info
    logger.info(f"Compute pairwise dFC (n_pairs={n_pairs}"
                f", decim={decim})")

    # Check if needs filtering
    if isinstance(f_low, (int,float)) or isinstance(f_high, (int,float)):
        x = filter_data(x, sfreq, f_low, f_high, method='iir', n_jobs=n_jobs)
    
    # z-score data
    x = z_score(x)

    # Decimate if needed
    if isinstance(decim, int):
        x     = x[...,::decim]
        times = times[::decim]

    # compute coherence on blocks of trials
    dfc = np.zeros((n_trials, n_pairs, len(times)))
    for tr in indices:
        # ______________________________ CORRELATION ____________________________
        def pairwise_correlation(w_x, w_y):
            # Elementwise product of z_scored time-series
            cc = x[tr,w_x]*x[tr,w_y]
            # mean inside temporal sliding window (if needed)
            if need_mean:
                cc = []
                for w_s, w_e in win_sample:
                    cc += [cc[..., w_s:w_e].mean(-1, keepdims=True)]
                cc = np.concatenate(cc, axis=-1)
            return cc

        # define the function to compute in parallel
        parallel, p_fun = parallel_func(
            pairwise_correlation, n_jobs=n_jobs, verbose=verbose,
            total=n_pairs)

        # compute the single trial coherence
        cc_tr        = parallel(p_fun(s, t) for s, t in zip(x_s, x_t))
        #  print(f"{np.shape(cc_tr)=}")
        dfc[tr, ...] = np.stack(cc_tr, axis=1)

    # ________________________________ DATAARRAY ______________________________
    # configuration
    cfg = dict(
        sfreq=sfreq, decim=decim
    )
    # conversion
    dfc = xr.DataArray(dfc, dims=('trials', 'roi', 'times'),
                      name='dfc', coords=(trials, roi_p, times),
                      attrs=cfg)
    return dfc
