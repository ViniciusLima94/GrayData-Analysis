###############################################################################
# SET THE PARAMETERS TO DO ALL THE ANALYSIS READ AND SAVE DATA
###############################################################################
import numpy as np

###############################################################################
# Set up directories to read the data
###############################################################################


def get_dates(monkey):
    sessions = []
    if monkey == "lucy":
        sessions = np.array(
            ['141017', '141014', '141015', '141016', '141023', '141024', '141029',
             '141103', '141112', '141113', '141125', '141126', '141127', '141128',
             '141202', '141203', '141205', '141208', '141209', '141211', '141212',
             '141215', '141216', '141217', '141218', '150114', '150126', '150128',
             '150129', '150205', '150210', '150211', '150212', '150213', '150217',
             '150219', '150223', '150224', '150226', '150227', '150302', '150303',
             '150304', '150305', '150403', '150407', '150408', '150413', '150414',
             '150415', '150416', '150427', '150428', '150429', '150430', '150504',
             '150511', '150512', '150527', '150528', '150529', '150608']
        )
    elif monkey == "ethyl":
        sessions = np.array(["110704", "110707", "110714", "110720", "110728",
                             "110803", "110811", "110906", "111028", "110705",
                             "110708", "110718", "110725", "110729", "110808",
                             "110823", "110920", "110706", "110711", "110719",
                             "110726", "110802", "110810", "110830", "110927",
                             ])
    return sessions


# Directories
dirs = {
    'rawdata': 'GrayLab/',
    'results': 'Results/',
    'monkey': ['lucy', 'ethyl'],
    'session': 'session01',
    'date': [get_dates("lucy"), get_dates("ethyl")]
}

###############################################################################
# Spectral analysis parameters
###############################################################################

# Smoothing windows
sm_times = 0.3  # In seconds
sm_freqs = 1
sm_kernel = "square"

# Defining parameters
decim = 20  # Downsampling factor
mode = 'multitaper'  # Wheter to use Morlet or Multitaper

n_freqs = 10  # How many frequencies to use
freqs = np.linspace(3, 75, n_freqs)  # Frequency array
n_cycles = freqs/4  # Number of cycles
mt_bandwidth = None


def return_evt_dt(align_at, monkey="lucy"):
    """ Return the window in which the data will be loaded
    depending on the alignment """
    assert align_at in ["cue", "match"]
    assert monkey in ["lucy", "ethyl"]
    if monkey == "lucy":
        if align_at == "cue":
            return [-0.65, 3.00]
        return [-2.2, 0.65]
    if align_at == "cue":
        return [-0.5, 2.7]
    return [-2.2, 0.65]
