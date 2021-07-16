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


def dFC(data, times=None, roi=None, sfreq=None, f_low=None, f_high=None, pairs=None, decim=None, win_args=None, block_size=None, verbose=False, n_jobs=1):
    r'''
    Computes dynamic FC for data in the format ("trials","roi","times")
    - x: Input tensor data with shape ("trials","roi","times").
    '''

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


    # show info
    logger.info(f"Compute pairwise dFC (n_pairs={n_pairs}"
                f", decim={decim})")

    # Check if needs filtering
    if isinstance(f_low, (int,float)) or isinstance(f_high, (int,float)):
        x = filter_data(x, sfreq, f_low, f_high, method='iir', n_jobs=n_jobs)
    
    # z-score data
    x = z_score(x)

    # If a window is specified then does windowed dFC else compute the co-fluctuations time-series
    is_windowed = False
    if isinstance(win_args, dict):
        is_windowed = True
        win, times = define_windows(times, **win_args)

    # compute coherence on blocks of trials
    dfc = np.zeros((n_trials, n_pairs, len(times)))
    for tr in indices:

        # ______________________________ CORRELATION ____________________________
        #  @nb.jit(nopython=True)
        def pairwise_correlation(w_x, w_y):
            cc = []
            for w_s,w_e in win:
                cc += [np.mean(x[tr,w_x,w_s:w_e]*x[tr,w_y,w_s:w_e],axis=-1)]
            cc = np.stack(cc,axis=1)
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
