######################################################################################
# SET THE PARAMETERS TO DO ALL THE ANALYSIS READ AND SAVE DATA
######################################################################################
import numpy as np

#####################################################################################################
# Directories
#####################################################################################################

# Directories
dirs = {
        'rawdata':'GrayLab/',
        'results':'Results/',
        'monkey' :['lucy', 'ethyl'],
        'session':'session01',
        'date'   :[['141014', '141015', '141205', '150128', '150211', '150304'], []]
        }
 
#####################################################################################################
# Spectral analysis
#####################################################################################################
# Bands
#bands = {
#        'lucy':  [[0.1,6],[6,14],[14,26],[26,42],[42,80]],
#        'ethyl': [[0.1,8],[8,21],[21,33],[33,80]]
#}

# Smoothing windows
sm_times = 300
sm_freqs = 1

# Defining parameters
f_start, f_end, n_freqs, sfreq = .1, 80, 50, 1000
freqs = np.linspace(f_start, f_end, n_freqs, endpoint=True)
delta = 15       # Downsampling factor
mode  = 'morlet' # ("morlet", "mt_1", "mt_2")
if mode in ["morlet", "mt_1"]:
    foi   = np.array([
            [0.1, 6.],
            [6., 14.],
            [14., 26.],
            [26., 42.],
            [42., 80.]
                ])
    n_cycles     = freqs/2
    mt_bandwidth = None
    decim_at='tfd'
elif mode is "mt_2":
    foi   = np.array([
            [0.1, 6.],
            [6., 14.],
            [14., 26.],
            [26., 42.],
            [42., 80.]
                ])
    freqs = foi.mean(axis=1)
    W     = np.ceil( foi[:,1]-foi[:,0] )   # Bandwidth
    foi   = None
    n_cycles     = np.array([3, 5, 9, 12, 16])
    mt_bandwidth = np.array([2, 4, 4.28, 5.647, 9.65])
    decim_at     = 'coh'

#if method == 'morlet':
#    # Range of frequencies to be analyzed
#    freqs = np.linspace(0.1, 80, 50) 
#    # Number of cycles for the wavelet
#    n_cycles = freqs/2
#    # Time bandwidth for the multitaper
#    time_bandwidth = None
#if method == 'multitaper':
#    # Time bandwidth for the multitaper
#    time_bandwidth = 8
#    # Frequncy centers
#    freqs = np.mean( bands['lucy'], axis = 1)
#    # Bandwidth
#    d_f = np.ceil( (np.array(bands['lucy'])[:,1]-np.array(bands['lucy'])[:,0])/2 )
#    # Number of cycles
#    n_cycles = time_bandwidth * freqs / d_f
