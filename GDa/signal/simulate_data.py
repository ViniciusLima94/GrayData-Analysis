'''
Translated from Mike X Cohen MATLAB codes.
'''
import numpy  as np 
import xarray as xr

def white_noise(trials=1, nvars=1, n=1000, fs=1000, amp=1, ntype='uniform'):
    r'''
    Generate a white noise signal.
    > INPUTS:
    - trials: number of trials
    - nvars: number of channles or variables
    - n: number of points in the signal
    - fs: sampling frequency of the signal
    - amp: amplitude of the noise
    - ntype: wheter the noise is uniform or normal distributed
    > OUTPUTS:
    - signal: generated signal with dimensions [trials,channels,time]
    '''
    assert ntype in ['uniform', 'normal'], 'ntype should be either uniform or normal'
    # Which method to use 
    if ntype=='uniform': noise = np.random.rand(trials, nvars, n)
    if ntype=='normal':  noise = np.random.randn(trials, nvars, n)

    signal = xr.DataArray( amp*noise, dims=("trials", "roi", "time"),
                           coords = {"time": np.arange(n)/fs})
    return signal

def pink_noise(trials=1, nvars=1, n=1000, fs=1000, tau=10):
    r'''
    Generate pink noise signal.
    > INPUTS:
    - trials: number of trials
    - nvars: number of channles or variables
    - n: number of points in the signal
    - fs: sampling frequency of the signal
    - tau: decay time of the power spectrum
    > OUTPUTS:
    - signal: generated signal with dimensions [trials,channels,time]
    '''

    # Generate 1/f spectrum
    a_s = np.random.rand(trials, nvars, n) * np.exp(-np.arange(n)/tau)
    # Randomized phases
    fc  = a_s * np.exp( 1j*2*np.pi*np.random.rand(trials, nvars, n) )
    # Obtaining signal via inverse FFT
    signal = np.fft.ifft(fc,axis=-1).real

    signal = xr.DataArray( signal, dims=("trials", "roi", "time"),
                           coords = {"time": np.arange(n)/fs})
    return signal

def ongoing_non_stationary(trials=1, nvars=1, n=1000, fs=1000, peakfreq=10, fwhm=1):
    r'''
    Generate an ongoing non-statinary signal with gaussian spectra.
    > INPUTS:
    - trials: number of trials
    - nvars: number of channles or variables
    - n: number of points in the signal
    - fs: sampling frequency of the signal
    - peakfreqs: the peak frequencies in the spectra
    - fwhm: fullwidth at half maximum
    > OUTPUTS:
    - signal: generated signal with dimensions [trials,channels,time]
    '''
    
    # frequency axis
    freqs = np.linspace(0, fs, n)
    # gaussian in the frequency domain
    s     = fwhm*(2*np.pi-1)/(4*np.pi)                     # Normalized width
    x     = (freqs-peakfreq[:,np.newaxis]).T/s             # Shifted frequencies
    fg    = np.sum( np.exp( -0.5 * x**2 ), axis=1)         # Gaussian

    # Fourier coefficients of random spectrum
    fc    = np.random.rand(trials, nvars, n) * np.exp( 1j*2*np.pi*np.random.rand(trials, nvars, n) )
    # Multiply by the gaussian
    fc    = fg * fc

    # Generated signal
    signal = np.fft.ifft(fc, axis=-1).real 
    # Convert to xarray
    signal = xr.DataArray( signal, dims=("trials", "roi", "time"),
                           coords = {"time": np.arange(n)/fs} )

    return signal

def transiesnt_oscillation_gauss(trials=1, nvars=1, n=1000, fs=1000, sin_freq=10, peaktime=1, width=0.1, phaselocked=True):
    r'''
    Generate a transient oscilation combining a gaussian in time domain and a sin.
    > INPUTS:
    - trials: number of trials
    - nvars: number of channles or variables
    - n: number of points in the signal
    - fs: sampling frequency of the signal
    - sin_freq: frequency of the sin oscillation
    - peaktime: time of the gaussian peak
    - width: width of the gaussian
    > OUTPUTS:
    - signal: generated signal with dimensions [trials,channels,time]
    '''

    # time array
    times = np.arange(n)/fs

    # generate time-domain gaussian
    gaus  = np.sum( np.exp( -(np.arange(n)/fs-peaktime[:,np.newaxis]).T**2 / (2*width**2) ), axis=1)

    # generate sine function
    sw = np.sin( 2*np.pi*sin_freq*np.arange(n)/fs  + int(True)*np.random.rand(trials, nvars, 1)*2 *np.pi )

    # generate the signal
    signal = sw*gaus

    # convert to xarray
    signal = xr.DataArray( signal, dims=("trials", "roi", "time"),
                           coords = {"time": np.arange(n)/fs} )

    return signal

