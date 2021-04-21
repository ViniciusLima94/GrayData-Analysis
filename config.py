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
bands = {
        'lucy':  [[0.1,6],[6,14],[14,26],[26,42],[42,80]],
        'ethyl': [[0.1,8],[8,21],[21,33],[33,80]]
        }

# Smoothing windows
win_freq = 1   # Freq
win_time = 70  # Time
# Delta for downsampling
delta = 15
# Method
method   = 'multitaper'

if method == 'morlet':
    # Range of frequencies to be analyzed
    freqs = np.linspace(0.1, 80, 50) 
    # Number of cycles for the wavelet
    n_cycles = freqs/2
    # Time bandwidth for the multitaper
    time_bandwidth = None
if method == 'multitaper':
    # Time bandwidth for the multitaper
    time_bandwidth = 8
    # Frequncy centers
    freqs = np.mean( bands['lucy'], axis = 1)
    # Bandwidth
    d_f = np.ceil( (np.array(bands['lucy'])[:,1]-np.array(bands['lucy'])[:,0])/2 )
    # Number of cycles
    n_cycles = time_bandwidth * freqs / d_f
