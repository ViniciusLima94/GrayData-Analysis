######################################################################################
# SET THE PARAMETERS TO DO ALL THE ANALYSIS READ AND SAVE DATA
######################################################################################
import numpy as np

#####################################################################################################
# Directories
#####################################################################################################

dates = np.loadtxt('GrayLab/lucy/sessions.txt', dtype=str)

# Directories
dirs = {
        'rawdata':'GrayLab/',
        'results':'Results/',
        'monkey' :['lucy', 'ethyl'],
        'session':'session01',
        'date'   :[dates, []]
        }

#  dirs = {
#          'rawdata':'GrayLab/',
#          'results':'Results/',
#          'monkey' :['lucy', 'ethyl'],
#          'session':'session01',
#          'date'   :[['150128', '150211', '150304'], []]
#          }
 
#####################################################################################################
# Spectral analysis
#####################################################################################################

# Smoothing windows
sm_times = 300
sm_freqs = 1

# Defining parameters
delta = 15       # Downsampling factor
mode  = 'morlet' # ("morlet", "mt_1", "mt_2")
foi   = np.array([
        [0.1, 6.],
        [6., 14.],
        [14., 26.],
        [26., 42.],
        [42., 80.]
            ])

if mode in ["morlet", "mt_1"]:
    n_freqs = 50
    freqs = np.linspace(foi[0,0], foi[-1,1], n_freqs, endpoint=True)
    n_cycles     = freqs/2
    mt_bandwidth = None
    decim_at='tfd'
elif mode == "mt_2":
    freqs = foi.mean(axis=1)
    W     = np.ceil( foi[:,1]-foi[:,0] )   # Bandwidth
    foi   = None
    n_cycles     = np.array([3, 5, 9, 12, 16])
    mt_bandwidth = np.array([2, 4, 4.28, 5.647, 9.65])
    decim_at     = 'coh'
