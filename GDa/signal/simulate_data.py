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

def transient_oscillation_gauss(trials=1, nvars=1, n=1000, fs=1000, sin_freq=10, peaktime=1, width=0.1, phaselocked=True):
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
    sw = np.sin( 2*np.pi*sin_freq*np.arange(n)/fs  + (1-int(phaselocked))*np.random.rand(trials, nvars, 1)*2 *np.pi )

    # generate the signal
    signal = sw*gaus

    # convert to xarray
    signal = xr.DataArray( signal, dims=("trials", "roi", "time"),
                           coords = {"time": np.arange(n)/fs} )

    return signal

def transient_oscillation_gauss_non_stationary(trials=1, nvars=1, n=1000, fs=1000, peakfreq=10, fwhm=1, peaktime=1, width=0.1):
    r'''
    Generate a transient oscilation combining a gaussian in time domain and frequency domain
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

    # frequency axis
    freqs = np.linspace(0, fs, n)
    # gaussian in the frequency domain
    s     = fwhm*(2*np.pi-1)/(4*np.pi)                     # Normalized width
    x     = (freqs-peakfreq[:,np.newaxis]).T/s             # Shifted frequencies
    fg   = np.sum( np.exp( -0.5 * x**2 ), axis=1)          # Gaussian

    # Fourier coefficients of random spectrum
    fc    = np.random.rand(trials, nvars, n) * np.exp( 1j*2*np.pi*np.random.rand(trials, nvars, n) )
    # Multiply by the gaussian
    fc    = fg * fc

    # time array
    times = np.arange(n)/fs
    # generate time-domain gaussian
    gaus  = np.sum( np.exp( -(times-peaktime[:,np.newaxis]).T**2 / (2*width**2) ), axis=1)

    # generate the signal
    signal = np.fft.ifft(fc, axis=-1).real*gaus

    # convert to xarray
    signal = xr.DataArray( signal, dims=("trials", "roi", "time"),
                           coords = {"time": np.arange(n)/fs} )

    return signal

def ar_model_dhamala(trials = 10, n=5000, fs = 200, C=0.2, t_start=0, t_stop=None, cov = None, verbose=False):
    r'''
    Coupled auto-regressive model from Dhamala et. al. (2008)
    X_{1}(t) = 0.55X_{1}(t-1)-0.8X_{1}(t-2)+C(t)X_{2}(t-1)+\epsilon (t)
    X_{2}(t) = 0.55X_{2}(t-1)-0.8X_{2}(t-2)+\xi (t)
    Here, X_1(t) and X_2(t) are AR(2). The variable t is the time step index,
    such that the actual time is t'=t\,\Delta t=t/f_{\rm s}. Besides, we know by construction that
    X_2(t) influences X_1(t) through the coupling constant C
    (although the opposite does not happen). In the simulation C(t)=0.25 for t<15s, and zero otherwise. 
    > INPUTS:
    - trials: number of trials
    - n: number of points in the signal
    - fs: sampling frequency of the signal
    - t_start: starting time of coupling
    - t_stop:  ending time of coupling
    - cov: normal noise covariance matrix
    > OUTPUTS:
    - signal: generated signal with dimensions [trials,channels,time]
    '''

    from tqdm import tqdm 

    T = n / fs

    time = np.linspace(0, T, n)

    X = np.random.random([trials, n])
    Y = np.random.random([trials, n])

    def interval(t, t_start, t_stop):
        if t_stop==None:
            return (t>=t_start)
        else:
            return (t>=t_start)*(t<=t_stop)

    # Iterator
    itr = range(trials) 
    for i in ( tqdm( itr ) if verbose else itr ):
        E = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=(n,))
        for t in range(2, n):
            X[i,t] = 0.55*X[i,t-1] - 0.8*X[i,t-2] + interval(time[t],t_start,t_stop)*C*Y[i,t-1] + E[t,0]
            Y[i,t] = 0.55*Y[i,t-1] - 0.8*Y[i,t-2] + E[t,1]

    Z = np.zeros([trials,2,n])

    Z[:,0,:] = X
    Z[:,1,:] = Y

    Z = xr.DataArray(Z, dims=('trials', 'roi', 'time'),
                     coords=(np.arange(trials), ['X_1', 'X_2'], np.arange(n) / fs))

    return Z
