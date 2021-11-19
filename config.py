###############################################################################
# SET THE PARAMETERS TO DO ALL THE ANALYSIS READ AND SAVE DATA
###############################################################################
import numpy as np

###############################################################################
# Directories
###############################################################################
dates = np.loadtxt('GrayLab/lucy/sessions.txt', dtype=str)

# Directories
dirs = {
        'rawdata': 'GrayLab/',
        'results': 'Results/',
        'monkey': ['lucy', 'ethyl'],
        'session': 'session01',
        'date': [dates, []]
        }

###############################################################################
# Spectral analysis
###############################################################################

# Smoothing windows
sm_times = 0.3  # In seconds
sm_freqs = 1
sm_kernel = "square"

# Defining parameters
delta = 20       # Downsampling factor
mode = 'morlet'  # ("morlet", "mt_1", "mt_2")
foi = np.array([
        [0, 6.],
        [6., 14.],
        [14., 26.],
        [26., 42.],
        [42., 80.]
            ])

# if mode in ["morlet", "mt_1"]:
#  n_freqs = 15
freqs = np.linspace(3, 75, 10)
# Frequency resolution
#  s_f   = (foi[:,1]-foi[:,0])/4
n_cycles = freqs/4
mt_bandwidth = None
# elif mode == "multitaper":
# freqs = foi.mean(axis=1)
# W     = np.ceil( foi[:,1]-foi[:,0] ) # Bandwidth
# foi   = None
# n_cycles     = np.array([3, 5, 9, 12, 16])
# mt_bandwidth = np.array([2, 4, 4.28, 5.647, 9.65])
