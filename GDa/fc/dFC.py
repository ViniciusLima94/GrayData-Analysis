import numpy as np
import xarray as xr

from frites.utils import parallel_func
from frites.io import set_log_level, logger
from frites.conn import conn_io
from frites.conn.conn_tf import (_tf_decomp, _create_kernel,
                                 _smooth_spectra, _foi_average)


def z_score(x):
    """
    Computes the z-score of data in the format with the time
    axis as the last dimension.

    Parameters:
    ----------

    x: array_like
        Input data-tensor.
    """
    assert isinstance(x, (np.ndarray, xr.DataArray))

    if isinstance(x, np.ndarray):
        return (x-x.mean(-1)[..., np.newaxis])/x.std(-1)[..., np.newaxis]
    elif isinstance(x, xr.DataArray):
        assert "times" in x.dims
        return (x-x.mean(dim="times"))/x.std(dim="times")


def _pec(w, kernel, foi_idx, x_s, x_t, kw_para):
    """Power envelopes correlation"""
    # auto spectra (faster that w * w.conj())
    s_auto = w.real ** 2 + w.imag ** 2

    # smooth the auto spectra
    s_auto = _smooth_spectra(s_auto, kernel)
    # demean spectra
    s_auto = z_score(s_auto)

    # define the pairwise coherence
    def pairwise_pec(w_x, w_y):
        # computes the pec
        out = s_auto[:, w_x, :, :] * s_auto[:, w_y, :, :]
        # mean inside frequency sliding window (if needed)
        if isinstance(foi_idx, np.ndarray):
            return _foi_average(out, foi_idx)
        else:
            return out

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(pairwise_pec, **kw_para)

    # compute the single trial coherence
    return parallel(p_fun(s, t) for s, t in zip(x_s, x_t))


def conn_power_corr(
        data, freqs=None, roi=None, times=None, pairs=None,
        sfreq=None, foi=None, sm_times=.5, sm_freqs=1, sm_kernel='hanning',
        mode='morlet', n_cycles=7., mt_bandwidth=None, decim=1, kw_cwt={},
        kw_mt={}, block_size=None, n_jobs=-1, verbose=None, dtype=np.float32):

    """Wavelet-based single-trial time-resolved spectral connectivity.
    Parameters
    ----------
    data : array_like
        Electrophysiological data. Several input types are supported :
            * Standard NumPy arrays of shape (n_epochs, n_roi, n_times)
            * mne.Epochs
            * xarray.DataArray of shape (n_epochs, n_roi, n_times)
    metric : str | "coh"
        Which connectivity metric. Use either :
            * 'coh' : Coherence
            * 'plv' : Phase-Locking Value (PLV)
            * 'sxy' : Cross-spectrum
        By default, the coherenc is used.
    freqs : array_like
        Array of central frequencies of shape (n_freqs,).
    roi : array_like | None
        ROI names of a single subject. If the input is an xarray, the
        name of the ROI dimension can be provided
    times : array_like | None
        Time vector array of shape (n_times,). If the input is an xarray, the
        name of the time dimension can be provided
    pairs : array_like | None
        Pairs of links of shape (n_pairs, 2) to compute. If None, all pairs are
        computed
    sfreq : float | None
        Sampling frequency
    foi : array_like | None
        Extract frequencies of interest. This parameters should be an array of
        shapes (n_foi, 2) defining where each band of interest start and
        finish.
    sm_times : float | .5
        Number of points to consider for the temporal smoothing in seconds. By
        default, a 500ms smoothing is used.
    sm_freqs : int | 1
        Number of points for frequency smoothing. By default, 1 is used which
        is equivalent to no smoothing
    kernel : {'square', 'hanning'}
        Kernel type to use. Choose either 'square' or 'hanning'
    mode : {'morlet', 'multitaper'}
        Spectrum estimation mode can be either: 'multitaper' or 'morlet'.
    n_cycles : array_like | 7.
        Number of cycles to use for each frequency. If a float or an integer is
        used, the same number of cycles is going to be used for all frequencies
    mt_bandwidth : array_like | None
        The bandwidth of the multitaper windowing function in Hz. Only used in
        'multitaper' mode.
    decim : int | 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition. default 1 If int, returns tfr[…, ::decim]. If slice,
        returns tfr[…, decim].
    kw_cwt : dict | {}
        Additional arguments sent to the mne-function
        :py:`mne.time_frequency.tfr_array_morlet`
    kw_mt : dict | {}
        Additional arguments sent to the mne-function
        :py:`mne.time_frequency.tfr_array_multitaper`
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
    conn : xarray.DataArray
        DataArray of shape (n_trials, n_pairs, n_freqs, n_times)
    """
    set_log_level(verbose)

    if isinstance(sm_times, np.ndarray):
        raise NotImplementedError("Frequency dependent kernel in development"
                                  f"only first {sm_times[0]} will be used")

    # _________________________________ METHODS _______________________________
    conn_f, f_name = {
        'pec': (_pec, "Power correlation")
    }['pec']

    # _________________________________ INPUTS ________________________________
    # inputs conversion
    data, cfg = conn_io(
        data, times=times, roi=roi, agg_ch=False, win_sample=None, pairs=pairs,
        sort=True, block_size=block_size, sfreq=sfreq, freqs=freqs, foi=foi,
        sm_times=sm_times, sm_freqs=sm_freqs, verbose=verbose,
        name=f'Sepctral connectivity (metric = {f_name}, mode={mode})',
    )

    # extract variables
    x, trials, attrs = data.data, data['y'].data, cfg['attrs']
    times, n_trials = data['times'].data, len(trials)
    x_s, x_t, roi_p = cfg['x_s'], cfg['x_t'], cfg['roi_p']
    indices, sfreq = cfg['blocks'], cfg['sfreq']
    freqs, _, foi_idx = cfg['freqs'], cfg['need_foi'], cfg['foi_idx']
    f_vec, sm_times, sm_freqs = cfg['f_vec'], cfg['sm_times'], cfg['sm_freqs']
    n_pairs, n_freqs = len(x_s), len(freqs)

    # temporal decimation
    if isinstance(decim, int):
        times = times[::decim]
        sm_times = int(np.round(sm_times / decim))
        sm_times = max(sm_times, 1)

    # Create smoothing kernel
    kernel = _create_kernel(sm_times, sm_freqs, kernel=sm_kernel)

    # define arguments for parallel computing
    mesg = f'Estimating pairwise {f_name} for trials %s'
    kw_para = dict(n_jobs=n_jobs, verbose=verbose, total=n_pairs)

    # show info
    logger.info(f"Computing pairwise {f_name} (n_pairs={n_pairs}, "
                f"n_freqs={n_freqs}, decim={decim}, sm_times={sm_times}, "
                f"sm_freqs={sm_freqs})")

    # ______________________ CONTAINER FOR CONNECTIVITY _______________________
    # compute coherence on blocks of trials
    conn = np.zeros((n_trials, n_pairs, len(f_vec), len(times)), dtype=dtype)
    for tr in indices:
        # --------------------------- TIME-FREQUENCY --------------------------
        # time-frequency decomposition
        w = _tf_decomp(
            x[tr, ...], sfreq, freqs, n_cycles=n_cycles, decim=decim,
            mode=mode, mt_bandwidth=mt_bandwidth, kw_cwt=kw_cwt, kw_mt=kw_mt,
            n_jobs=n_jobs)

        # ----------------------------- CONN TRIALS ---------------------------
        # give indication about computed trials
        kw_para['mesg'] = mesg % f"{tr[0]}...{tr[-1]}"

        # computes conn across trials
        conn_tr = conn_f(w, kernel, foi_idx, x_s, x_t, kw_para)

        # merge results
        conn[tr, ...] = np.stack(conn_tr, axis=1)

        # Call GC
        del conn_tr, w

    # _________________________________ OUTPUTS _______________________________
    # configuration
    cfg = dict(
        sfreq=sfreq, sm_times=sm_times, sm_freqs=sm_freqs, sm_kernel=sm_kernel,
        mode=mode, n_cycles=n_cycles, mt_bandwidth=mt_bandwidth, decim=decim,
        type=metric
    )

    # conversion
    conn = xr.DataArray(conn, dims=('trials', 'roi', 'freqs', 'times'),
                        name=metric, coords=(trials, roi_p, f_vec, times),
                        attrs=check_attrs({**attrs, **cfg}))
    return conn
