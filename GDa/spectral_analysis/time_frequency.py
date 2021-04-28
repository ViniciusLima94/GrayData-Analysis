import numpy            as     np
import mne
import os
import h5py
import multiprocessing
from   joblib           import Parallel, delayed
from   .util            import smooth_spectra
from   ..util           import downsample

def wavelet_transform(data = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0, 
                      time_bandwidth = None, delta = 1, method = 'morlet', baseline_correction = False, n_jobs = 1):
    r'''
    Compute the time-frequency decompostion of the data using either morlet or multitatper transform.
    > INPUTS:
    - data: Data with dimensions ("trials","roi","time")
    - fs: sampling frequency of the data
    - freqs: Array containing the frequencies used in the tf decomposition
    - n_cycles: Int or array specifying the number of cycles of the waveletes/tapers for each frequency in freq
    - time_bandwidth: Temporal smmothing parameter for the multitaper transform (temp. smoothing proportional to 1/time_bandwidth)
    - delta: Delta for downsampling the time axis of the data
    - method: Which method to use to perform the tf decomposition (morlet or multitaper)
    - baseline_correction: Wheter to baseline correct the data or not
    - n_jobs: Number of jobs to use
    > OUTPUTS:
    - out: The data decomposed in tf domain with dimensions  ("trials","roi","freq","time")
    '''

    if method not in ['morlet', 'multitaper']:
        raise ValueError('Method should be either "morlet" or "multitaper"')
    if method == 'morlet' and time_bandwidth is not None:
        print('For method equals "morlet" time_bandwidth is not used')
    if method == 'morlet':
        out = mne.time_frequency.tfr_array_morlet(data, fs, freqs, n_cycles = n_cycles, zero_mean=False,
                                                  output='complex', decim = delta, n_jobs=n_jobs)
    if method == 'multitaper':
        out = mne.time_frequency.tfr_array_multitaper(data, fs, freqs, n_cycles = n_cycles, zero_mean=False,
                                                      time_bandwidth = time_bandwidth, output='complex', 
                                                      decim = delta, n_jobs=n_jobs)
    # Baseline correcting tf signal
    if baseline_correction:
        a = out.mean(axis=-1)
        b = out.std(axis=-1)
        out = (out-np.expand_dims(a,-1))/np.expand_dims(b,-1)

    return out

def wavelet_coherence(data = None, pairs = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0, 
                      time_bandwidth = None, delta = 1, method = 'morlet', win_time = 1, win_freq = 1, 
                      kernel='hann', dir_out = None, baseline_correction = False, n_jobs = 1):
    r'''
    Compute the tf coherence for the given pairs of channels in the data using either morlet or multitatper transform.
    > INPUTS:
    - data: Data with dimensions ("trials","roi","time")
    - pairs: Matrix (2xNpairs) with the index of the pairs to which compute coherence
    - fs: sampling frequency of the data
    - freqs: Array containing the frequencies used in the tf decomposition
    - n_cycles: Int or array specifying the number of cycles of the waveletes/tapers for each frequency in freq
    - time_bandwidth: Temporal smmothing parameter for the multitaper transform (temp. imoothing proportional to 1/time_bandwidth)
    - delta: Delta for downsampling the time axis of the data
    - method: Which method to use to perform the tf decomposition (morlet or multitaper)
    - win_time: Length of the smoothing kernel in the time axis
    - win_freq: Length of the smoothing kernel in the frequency axis
    - kernel: Which kernel to use ('square' or 'hann')
    - dir_out: Path where the coherence data will be saved (if None, files won't be saved)
    - baseline_correction: Wheter to baseline correct the data or not
    - n_jobs: Number of jobs to use
    > OUTPUTS:
    - out: The coherence tensor with dimensions ("links","trials","freq","time")
    '''

    # Data dimension
    T, C, L = data.shape

    # Computing wavelets
    W = wavelet_transform(data = data, fs = fs, freqs = freqs, n_cycles = n_cycles, 
                          time_bandwidth = time_bandwidth, delta = 1, 
                          method = method, baseline_correction=baseline_correction, n_jobs = n_jobs)
    # Auto spectra
    S_auto = W * np.conj(W)

    def pairwise_coherence(index_pair, win_time, win_freq):
        channel1, channel2 = pairs[index_pair, 0], pairs[index_pair, 1]
        Sxy = W[:,channel1,:,:] * np.conj(W[:,channel2,:,:])
        if win_time > 1 or win_freq > 1:
            Sxx = smooth_spectra(S_auto[:,channel1, :, :], win_time, win_freq, kernel=kernel, fft=True, axes = (1,2))
            Syy = smooth_spectra(S_auto[:,channel2, :, :], win_time, win_freq, kernel=kernel, fft=True, axes = (1,2))
            Sxy = smooth_spectra(Sxy, win_time, win_freq, fft=True, kernel=kernel, axes = (1,2))
            coh = np.abs(Sxy[:,:,::delta])**2 / (Sxx[:,:,::delta] * Syy[:,:,::delta])
        else:
            coh = np.abs(Sxy[:,:,::delta])**2 / (S_auto[:,channel1,:,::delta]*S_auto[:,channel2,:,::delta])
        if dir_out is not None:
            file_name = os.path.join( dir_out, 'ch1_' + str(channel1) + '_ch2_' + str(channel2) +'.h5')
            with h5py.File(file_name, 'w') as hf:
                hf.create_dataset('coherence', data=np.abs(coh).astype(np.float32))
        return coh

    out = Parallel(n_jobs=n_jobs, backend='loky', timeout=1e6)(delayed(pairwise_coherence)(i, win_time, win_freq) for i in range(pairs.shape[0]) )

    return np.squeeze(out).real

# BELLOW WE HAVE THE WAVLET TRANSFORMS THAT I USED ONLY FOR TESTING BUT NOW ARE DEPRECATED
# def gabor_transform(signal = None, fs = 20, freqs = np.arange(6,60,1), n_cycles = 7.0):
#     n      = len(signal)
#     sigma2 = 1
#     if n%2 == 0:
#         omega  = np.concatenate( (np.arange(0, n/2), np.arange(-np.ceil(n/2), 0) ) ) * fs/n
#     else:
#         omega  = np.concatenate( (np.arange(0, n/2), np.arange(-np.ceil(n/2)+1, 0) ) ) * fs/n
# 
#     fftx   = np.fft.fft(signal)
# 
#     tolerance = 0.5
# 
#     mincenterfreq = 2*tolerance*np.sqrt(sigma2)*fs*n_cycles/n
#     maxcenterfreq = fs*n_cycles/(n_cycles+tolerance/np.sqrt(sigma2))
# 
#     s_array  = n_cycles/freqs
#     minscale = n_cycles/maxcenterfreq
#     maxscale = n_cycles/mincenterfreq
# 
#     nscale = len(freqs)
#     wt     = np.zeros([n,nscale]) * (1+1j)
#     scaleindices = np.arange(0,len(s_array))[(s_array>=minscale)*(s_array<=maxscale)]
#     psi_array = np.zeros([n, nscale]) * (1+1j)
# 
#     for kscale in scaleindices:
#         s    = s_array[kscale]
#         freq = (s*omega - n_cycles)
#         Psi  = (4*np.pi*sigma2)**(1/4) * np.sqrt(s) * np.exp(-sigma2/2*freq**2)
#         wt[:,kscale] = np.fft.ifft(fftx*Psi)
#         psi_array[:,kscale]=np.fft.ifft(Psi)
# 
#     return wt
# 
# def gabor_spectrum(signal1 = None, signal2 = None, fs = 20, freqs = np.arange(6,60,1),  
#                    kernel='hann', win_time = 1, win_freq = 1, n_cycles = 7.0):
#     if type(signal2) != np.ndarray:
#         wt1    = gabor_transform(signal=signal1,fs=fs,freqs=freqs,n_cycles=n_cycles)
#         Sxx    = wt1*np.conj(wt1)
#         return smooth_spectra(Sxx.T, win_time, win_freq, kernel=kernel, fft=True, axes=(0,1))
#     else:
#         wt1 = gabor_transform(signal=signal1,fs=fs,freqs=freqs,n_cycles=n_cycles)
#         wt2 = gabor_transform(signal=signal2,fs=fs,freqs=freqs,n_cycles=n_cycles)
# 
#         npts  = wt1.shape[0]
#         nfreq = len(freqs)
# 
#         Sxy    = wt1*np.conj(wt2)
#         Sxx    = wt1*np.conj(wt1)
#         Syy    = wt2*np.conj(wt2)
# 
#         # Smoothing spectra
#         Sxx = smooth_spectra(Sxx.T, win_time, win_freq, kernel=kernel, fft=True, axes=(0,1))
#         Syy = smooth_spectra(Syy.T, win_time, win_freq, kernel=kernel, fft=True, axes=(0,1))
#         Sxy = smooth_spectra(Sxy.T, win_time, win_freq, kernel=kernel, fft=True, axes=(0,1))
#         return Sxx, Syy, Sxy
# 
# def gabor_coherence(signal1 = None, signal2 = None, fs = 20, freqs = np.arange(6,60,1),  
#                     kernel='hann', win_time = 1, win_freq = 1, n_cycles = 7.0):
#     Sxx, Syy, Sxy = gabor_spectrum(signal1 = signal1, signal2 = signal2, fs = fs, freqs = freqs,  
#                     kernel=kernel, win_time = win_time, win_freq = win_freq, n_cycles = n_cycles)
# 
#     return np.abs(Sxy)**2 / (Sxx * Syy)
