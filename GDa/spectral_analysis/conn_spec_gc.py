
"""Compute the epoch-averaged spectral Wiener-Granger Causality (Dhamala et. al. (2008)).

1. A method that computes GC in time-frequency
"""
# Authors : Vinicius Lima <vinicius.lima.cordeiro@gmail.com >
#           Etienne Combrisson <e.combrisson@gmail.com>
#
# License : BSD (3-clause)

import numpy  as np
import xarray as xr

from scipy.signal       import fftconvolve, coherence
from mne.time_frequency import tfr_array_morlet, tfr_array_multitaper

from frites.io          import set_log_level, logger
from frites.utils       import parallel_func
from frites.dataset     import SubjectEphy
from frites.config      import CONFIG


###############################################################################
###############################################################################
#                           NON-PARAMETRIC SPECTRAL GC
###############################################################################
###############################################################################

##################### DUPLICATED FROM CONN_COH ##################################
def _tf_decomp(data, sf, freqs, mode='morlet', n_cycles=7.0, mt_bandwidth=None,
               decim=1, kw_cwt={}, kw_mt={}, n_jobs=1):
    """Time-frequency decomposition using MNE-Python.

    Parameters
    ----------
    data : array_like
        Electrophysiological data of shape (n_trials, n_chans, n_times)
    sf : float
        Sampling frequency
    freqs : array_like
        Central frequency vector.
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

    Returns
    -------
    out : array_like
        Time-frequency transform of shape (n_epochs, n_chans, n_freqs, n_times)
    """
    if mode == 'morlet':
        out = tfr_array_morlet(
            data, sf, freqs, n_cycles=n_cycles, output='complex', decim=decim,
            n_jobs=n_jobs, **kw_cwt)
    elif mode == 'multitaper':
        # In case multiple values are provided for mt_bandwidth
        # the MT decomposition is done separatedly for each 
        # Frequency center
        if isinstance(mt_bandwidth, (list, tuple, np.ndarray)):
            # Arrays freqs, n_cycles, mt_bandwidth should have the same size
            assert len(freqs)==len(n_cycles)==len(mt_bandwidth)
            out = []
            for f_c, n_c, mt in zip(freqs, n_cycles, mt_bandwidth):
                out += [tfr_array_multitaper(
                    data, sf, [f_c], n_cycles=float(n_c), time_bandwidth=mt,
                    output='complex', decim=decim, n_jobs=n_jobs, **kw_mt)]
            out = np.stack(out, axis=2).squeeze()
        elif isinstance(mt_bandwidth, (type(None), int, float)):
            out = tfr_array_multitaper(
                data, sf, freqs, n_cycles=n_cycles, time_bandwidth=mt_bandwidth,
                output='complex', decim=decim, n_jobs=n_jobs, **kw_mt)
    else:
        raise ValueError('Method should be either "morlet" or "multitaper"')

    return out

def conn_spec_gc(
    data, freqs=None, roi=None, times=None, sfreq=None, pairs=None,
    win_sample=None, foi=None, mode='morlet', n_cycles=7., n_iter=50, tol=1e-8, 
    mt_bandwidth=None, decim=1, kw_cwt={}, kw_mt={}, block_size=None, 
    n_jobs=-1, verbose=None):
    """Non-parametric spectral GC using tf decomposition from Dhamala et. al (2008)

    Parameters
    ----------
    data : array_like
        Electrophysiological data. Several input types are supported :

            * Standard NumPy arrays of shape (n_epochs, n_roi, n_times)
            * mne.Epochs
            * xarray.DataArray of shape (n_epochs, n_roi, n_times)

    freqs : array_like
        Array of frequencies.
    roi : array_like | None
        ROI names of a single subject. If the input is an xarray, the
        name of the ROI dimension can be provided
    times : array_like | None
        Time vector array of shape (n_times,). If the input is an xarray, the
        name of the time dimension can be provided
    sfreq : float
        The sampling frequency
    pairs : array_like | None
        Pairs of contacts
    win_sample : array_like | None
        Array of shape (n_windows, 2) in order to take the mean of the
        estimated coherence inside sliding windows. You can use the function
        :func:`frites.conn.define_windows` to define either manually either
        sliding windows.
    foi : array_like | None
        Extract frequencies of interest. This parameters should be an array of
        shapes (n_freqs, 2) defining where each band of interest start and
        finish.
    mode : {'morlet', 'multitaper'}
        Spectrum estimation mode can be either: 'multitaper' or 'morlet'.
    n_cycles : array_like | 7.
        Number of cycles to use for each frequency. If a float or an integer is
        used, the same number of cycles is going to be used for all frequencies
    n_iter : int | 30
        Number of iterations to be used in the Wilson spectral-matrix factorization.
    tol : float | 1e-12
        Tolerance of the Wilson factorization measure as the distance between the
        spectral matrix recontructed by the algorithm and the original one.
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
    coh : xarray.DataArray
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

    # inputs checking
    assert isinstance(freqs, (np.ndarray, list, tuple))
    assert isinstance(n_iter, int)
    assert isinstance(tol, float)
    freqs = np.array(freqs)
    n_freqs = len(freqs)

    # get the sorted directed pairs and build roi pairs names
    if (pairs is None):
        x_s, x_t = np.where(~np.eye(n_roi,dtype=bool))
    else:
        assert isinstance(pairs, np.ndarray)
        assert (pairs.ndim == 2) and (pairs.shape[1] == 2)
        x_s, x_t = pairs[:, 0], pairs[:, 1]
    #roi_s, roi_t = np.sort(np.c_[roi[x_s], roi[x_t]], axis=1).T
    roi_s, roi_t = (np.c_[roi[x_s], roi[x_t]]).T
    roi_p = [f"{s}->{t}" for s, t in zip(roi_s, roi_t)]
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

    # temporal decimation
    if isinstance(decim, int): times = times[::decim]  # noqa

    # frequency mean
    need_foi = isinstance(foi, np.ndarray) and foi.shape[1] == 2
    if need_foi:
        _f = xr.DataArray(np.arange(len(freqs)), dims=('freqs',),
                          coords=(freqs,))
        foi_s = _f.sel(freqs=foi[:, 0], method='nearest').data
        foi_e = _f.sel(freqs=foi[:, 1], method='nearest').data
        foi_idx = np.c_[foi_s, foi_e]
        f_vec = freqs[foi_idx].mean(1)
    else:
        f_vec = freqs

    # show info
    logger.info(f"Compute pairwise spectral Granger=causality (n_pairs={n_pairs}, "
                f"n_freqs={n_freqs}, decim={decim})")

    # compute coherence on blocks of trials
    gc = np.zeros((n_pairs, len(f_vec), len(times)))
    # ___________________________ TIME-FREQUENCY __________________________
    # time-frequency decomposition
    w = _tf_decomp(
        x, sfreq, freqs, n_cycles=n_cycles, decim=decim,
        mode=mode, mt_bandwidth=mt_bandwidth, kw_cwt=kw_cwt, kw_mt=kw_mt,
        n_jobs=n_jobs)

    # At this point the sampling frequency should be reescaled with decim
    if isinstance(decim, int): sfreq = sfreq/decim

    # ______________________________ COHERENCY ____________________________
    # auto spectra (faster that w * w.conj())
    s_auto = (w.real ** 2 + w.imag ** 2).mean(0) # Average over epochs

    def pairwise_gc(w_x, w_y):
        # computes the cross-spectrum
        s_xy = ( w[:, w_x, :, :] * np.conj(w[:, w_y, :, :]) ).mean(0) # Average over epochs
        s_yx = ( w[:, w_y, :, :] * np.conj(w[:, w_x, :, :]) ).mean(0) # Average over epochs
        # Spectral matrix
        S = np.array([[s_auto[w_x, :, :], s_xy],
                      [s_yx, s_auto[w_y, :, :]]])
        # Factorize spectral matrix for each time stamp (probabily very slow)
        Ix2y = np.zeros((len(f_vec),len(times)))
        for ts in range(S.shape[-1]):
            _, H, Z    = _ein_wilson_factorization(S[:,:,:,ts], freqs, sfreq, Niterations=n_iter, tol=tol, verbose=False)
            Ix2y[:,ts] = _granger_causality(S[:,:,:,ts], H, Z)

        return Ix2y 

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        pairwise_gc, n_jobs=n_jobs, verbose=verbose,
        total=n_pairs)

    # compute the single trial coherence
    gc = parallel(p_fun(s, t) for s, t in zip(x_s, x_t))

    # ________________________________ DATAARRAY ______________________________
    # configuration
    cfg = dict(
        sfreq=sfreq, mode=mode, n_cycles=n_cycles, mt_bandwidth=mt_bandwidth, decim=decim
    )
    # conversion
    gc = xr.DataArray(gc, dims=('roi', 'freqs', 'times'),
                       name='gc', coords=(roi_p, f_vec, times),
                       attrs=cfg)
    return gc

def _granger_causality(S, H, Z):
    '''
        This equations can be checkd in Ding 2009
    '''
    N = H.shape[2]
    Hxx = H[0,0,:]
    Hxy = H[0,1,:]
    Hyx = H[1,0,:]
    Hyy = H[1,1,:]

    #Hxx_tilda = Hxx + (Z[0,1]/Z[0,0]) * Hxy
    #Hyx_tilda = Hyx + (Z[0,1]/Z[0,0]) * Hxx
    Hyy_circf = Hyy + (Z[1,0]/Z[1,1]) * Hyx

    Syy = Hyy_circf*Z[1,1]*np.conj(Hyy_circf) + Hyx*(Z[0,0]-Z[1,0]*Z[1,0]/Z[1,1]) * np.conj(Hyx)
    #Sxx = Hxx_tilda*Z[0,0]*np.conj(Hxx_tilda) + Hxy*(Z[1,1]-Z[0,1]*Z[0,1]/Z[0,0]) * np.conj(Hxy)
    
    Ix2y = np.log( Syy/(Hyy_circf*Z[1,1]*np.conj(Hyy_circf)) )
    #Iy2x = np.log( Sxx/(Hxx_tilda*Z[0,0]*np.conj(Hxx_tilda)) )
    #Ixy  = np.zeros(N)

    # For now ignoring intantaneous GC
    #for i in range(N):
    #    Ixy[i]  = np.log( (Hxx_tilda[i]*Z[0,0]*np.conj(Hxx_tilda[i]))*(Hyy_circf[i]*Z[1,1]*np.conj(Hyy_circf[i])/np.linalg.det(S[:,:,i])) ).real
    
    return Ix2y.real#, Iy2x.real, Ixy.real

def _PlusOperator(g,m,fs,freq):
    
    N = freq.shape[0]-1

    gam = np.fft.ifft(g, axis=-1)

    gamp = gam.copy()
    beta0 = 0.5*gam[:,:,0]
    gamp[:,:,0] = np.triu(beta0)
    gamp[:,:,len(freq):] = 0 
    
    gp = np.fft.fft(gamp, axis=-1)

    return gp

def _ein_wilson_factorization(S, freq, fs, Niterations=100, tol=1e-12, verbose=True):
    '''
                Algorithm for the Wilson Factorization of the spectral matrix.
    '''

    m = S.shape[0]       # Number of variables
    N = freq.shape[0]-1  # Number of frequencies

    f_ind = 0
    Sarr  = np.zeros([m,m,2*N]) * (1+1j)
    for f in freq:
        Sarr[:,:,f_ind] = S[:,:,f_ind]
        if(f_ind>0):
                Sarr[:,:,2*N-f_ind] = S[:,:,f_ind].T
        f_ind += 1

    gam  = np.fft.ifft(Sarr, axis=-1).real
    h    = np.linalg.cholesky(gam[:,:,0]).T

    psi  = np.repeat(h[:,:,None].real, Sarr.shape[2], axis=2) * (1+0*1j)

    I = np.eye(m)

    def _tensormul(A,B):
        return np.einsum('ijz,jkz->ikz', A, B, casting='no')

    def _tensormatmul(A,B):
        return np.einsum('ijz,jk->ikz',  A, B, casting='no')

    g = np.zeros([m,m,2*N]) * (1+1j)
    for iteration in range(Niterations):
            psi_inv = np.linalg.inv(psi.T).T
            g = _tensormul( _tensormul(psi_inv,Sarr), np.conj(psi_inv).transpose(1,0,2) ) + I[:,:,None]

            gp = _PlusOperator(g, m, fs, freq)
            psiold = psi.copy()

            psi    = _tensormul(psi,gp)
            psierr = np.linalg.norm(psi-psiold, 1, axis=(0,1)).mean()
            if(psierr<tol):
                break
            if verbose == True:
                print('Err = ' + str(psierr))

    Snew = _tensormul(psi[:,:,:N+1], np.conj(psi[:,:,:N+1]).transpose(1,0,2))

    gamtmp = np.fft.ifft(psi,axis=-1)

    A0    = gamtmp[:,:,0]
    A0inv = np.linalg.inv(A0)
    Znew  = np.matmul(A0, A0.T).real

    Hnew = _tensormatmul(psi[:,:,:N+1], A0inv)

    return Snew, Hnew, Znew

def _wilson_factorization(S, freq, fs, Niterations=100, tol=1e-12, verbose=True):
    '''
        Algorithm for the Wilson Factorization of the spectral matrix.
    '''

    m = S.shape[0]    
    N = freq.shape[0]-1

    Sarr  = np.zeros([m,m,2*N]) * (1+1j)

    f_ind = 0

    for f in freq:
        Sarr[:,:,f_ind] = S[:,:,f_ind]
        if(f_ind>0):
            Sarr[:,:,2*N-f_ind] = S[:,:,f_ind].T
        f_ind += 1

    gam = np.zeros([m,m,2*N])

    for i in range(m):
        for j in range(m):
            gam[i,j,:] = (np.fft.ifft(Sarr[i,j,:])).real

    gam0 = gam[:,:,0]
    h    = np.linalg.cholesky(gam0).T

    psi = np.ones([m,m,2*N]) * (1+1j)

    for i in range(0,Sarr.shape[2]):
        psi[:,:,i] = h

    I = np.eye(m)

    g = np.zeros([m,m,2*N]) * (1+1j)
    for iteration in range(Niterations):

        for i in range(Sarr.shape[2]):
            g[:,:,i] = np.matmul(np.matmul(np.linalg.inv(psi[:,:,i]),Sarr[:,:,i]),np.conj(np.linalg.inv(psi[:,:,i])).T)+I

        gp = _PlusOperator(g, m, fs, freq)
        psiold = psi.copy()
        psierr = 0
        for i in range(Sarr.shape[2]):
            psi[:,:,i] =np.matmul(psi[:,:,i], gp[:,:,i])
            psierr    += np.linalg.norm(psi[:,:,i]-psiold[:,:,i],1) / Sarr.shape[2]

        if(psierr<tol):
            break

        if verbose == True:
            print('Err = ' + str(psierr))

    Snew = np.zeros([m,m,N+1]) * (1 + 1j)

    for i in range(N+1):
        Snew[:,:,i] = np.matmul(psi[:,:,i], np.conj(psi[:,:,i]).T)

    gamtmp = np.zeros([m,m,2*N]) * (1 + 1j)

    for i in range(m):
        for j in range(m):
            gamtmp[i,j,:] = np.fft.ifft(psi[i,j,:]).real

    A0    = gamtmp[:,:,0]
    A0inv = np.linalg.inv(A0)
    Znew  = np.matmul(A0, A0.T).real

    Hnew = np.zeros([m,m,N+1]) * (1 + 1j)

    for i in range(N+1):
        Hnew[:,:,i] = np.matmul(psi[:,:,i], A0inv)

    return Snew, Hnew, Znew

if __name__ == '__main__':

    pass
