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
# Bands
delta = [4,5,6,7]
alpha = [8,10,12,14]
beta  = [16,20,24,28,30,34] 
low_gamma  = np.arange(40, 70, 4) 
high_gamma = np.arange(70, 100, 4)
# Range of frequencies to be analyzed
freqs = np.concatenate((delta, alpha, beta, low_gamma, high_gamma)) #np.arange(4,102,2)
# Delta for downsampling
delta = 15
# Number of cycles for the wavelet
n_cycles = freqs/2
# Time bandwidth for the multitaper
time_bandwidth = 8
# Smoothing windows
win_freq = 1
win_time = 34
# Method
method   = 'morlet'
# Bands
bands = np.array([[4,8],[8,15],[15,40],[40,70], [70,100]])
