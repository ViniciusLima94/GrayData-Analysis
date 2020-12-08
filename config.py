######################################################################################
# SET THE PARAMETERS TO DO ALL THE ANALYSIS READ AND SAVE DATA
######################################################################################
import numpy as np

#####################################################################################################
# Directories
#####################################################################################################

# Directories
dirs = {'rawdata':'GrayLab/',
        'results':'Results/',
        'monkey' :['lucy', 'ethyl'],
        'session':'session01',
        'date'   :[['141014', '141015', '141205', '150128', '150211', '150304'], []]
        }
 
#####################################################################################################
# Spectral analysis
#####################################################################################################

# Range of frequencies to be analyzed
freqs = np.arange(4,60,2)
# Delta for downsampling
delta = 15
# Number of cycles for the wavelet
n_cycles = 5
# Time bandwidth for the multitaper
time_bandwidth = None
# Smoothing windows
win_freq = 1
win_time = 34
# Method
method   = 'morlet'
# Bands
bands = np.array([[4,8],[8,15],[15,30],[30,60]])
