'''
Translated from Mike X Cohen MATLAB codes.
'''
import numpy  as np 
import xarray as xr

def white_noise(trials, nvars, n, fs, amp):
    r'''
    Generate a white noise signal.
    > INPUTS:
    - trials: number of trials
    - nvars: number of channles or variables
    - n: number of points in the signal
    - fs: sampling frequency of the signal
    - amp: amplitude of the noise
    > OUTPUTS:
    - signal: generated signal with dimensions [trials,channels,time]
    '''
    signal = xr.DataArray( amp*np.random.rand(trials, nvars, n), dims=("trials", "roi", "time"),
                           coords = {"time": np.arange(n)/fs})
    return signal

def pink_noise(trials, nvars, n, fs, tau):
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

def ongoing_non_stationary(trials, nvars, n, fs, peakfreq, fwhm ):
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
    s     = fwhm*(2*np.pi-1)/(4*np.pi)                              # Normalized width
    x     = np.prod( (freqs-peakfreq[:,np.newaxis]).T/s, axis=1 )   # Shifted frequencies
    fg    = np.exp( -0.5 * x**2 )                                   # Gaussian

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

def transiesnt_oscillation_gauss():
    pass
