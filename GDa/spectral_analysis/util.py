import scipy.signal
import numpy         as np

def hann_window(win_size):
    r'''
    Create a Hann window of win_size.
    > INPUTS:
    win_size: Size of hanning window
    > OUTPUTS:
    > Hann window with size win_size
    '''
    if win_size < 3: return np.ones(win_size)
    else: return 0.5 - np.cos(2*np.pi*np.linspace(0,1,win_size))/2

def create_kernel(win_time, win_freq, kernel='hann'):
    assert kernel in ['square', 'hann'], "The kernel shape should be either square or hann."

    if kernel == 'square':
        return np.ones([win_freq, win_time])/(win_time*win_freq)
    else:
        # One hann window for time and one for frequency axis
        hann_t, hann_f = hann_window(win_time), hann_window(win_freq)
        hann           = (np.tile(hann_t, (win_freq,1)).T * hann_f).T
        return hann / np.sum(hann)

def smooth_spectra(spectra, win_time, win_freq, kernel='hann', fft=False, axes = 0):
    r'''
    Smooth multidimensional spectra.
    > INPUTS:
    - spectra: Multidim. array containing the spectra either [freq,time,trials] or [freq,time]
    - win_time: Filter length in time axis
    - win_freq: Filter length in freq axis
    - fft: Wheter to use fft or not to convolve
    - axes: Axes to perform the convolution
    > OUTPUTS:
    - smoothed spectra
    '''

    if len(spectra.shape) == 2:
        k = create_kernel(win_time, win_freq, kernel=kernel)
        ##kernel = hann_time*np.ones([win_freq, win_time])*hann_freq[:,None]/(win_freq*win_time)
    elif len(spectra.shape) == 3:
        k = create_kernel(win_time, win_freq, kernel=kernel)[np.newaxis,:,:]
        #kernel = hann_rime*np.ones([1, win_freq, win_time])/(win_freq*win_time)

    if fft == True:
        return scipy.signal.fftconvolve(spectra, k, mode='same', axes= axes)
    else:
        return scipy.signal.convolve(spectra, k, mode='same') 
