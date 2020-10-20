#####################################################################################################
# Read and save the LFP data and information for each trial in numpy format
#####################################################################################################
import mne
import numpy                as np 
import matplotlib.animation as animation
import matplotlib.pyplot    as plt

#####################################################################################################
# Loading session data
#####################################################################################################
session_data = np.load('raw_lfp/lucy_session01_150128.npy', allow_pickle=True).item()
LFP          = session_data['data']
fsample      = int(session_data['info']['fsample'])
T,C,N        = LFP.shape

freqs = np.arange(4,60,1)

W_ml = mne.time_frequency.tfr_array_morlet(LFP, fsample, freqs, n_cycles=5.0, zero_mean=False, 
                                           use_fft=True, decim=15, output='complex', n_jobs=-1, verbose=None)

W_mt = mne.time_frequency.tfr_array_multitaper(LFP, fsample, freqs, n_cycles=5.0, zero_mean=False, 
                                               time_bandwidth=None, use_fft=True, decim=15, output='complex', 
                                               n_jobs=-1, verbose=None)

#####################################################################################################
# Computing spectra
#####################################################################################################
trial    = 0
ch1, ch2 = 10, 30
# Morlet
Sxx_ml = W_ml[trial,ch1,:,:] * np.conj(W_ml[trial,ch1,:,:])
Syy_ml = W_ml[trial,ch2,:,:] * np.conj(W_ml[trial,ch2,:,:]) 
Sxy_ml = W_ml[trial,ch1,:,:] * np.conj(W_ml[trial,ch2,:,:]) 
# Multitaper
Sxx_mt = W_mt[trial,ch1,:,:] * np.conj(W_mt[trial,ch1,:,:])
Syy_mt = W_mt[trial,ch2,:,:] * np.conj(W_mt[trial,ch2,:,:]) 
Sxy_mt = W_mt[trial,ch1,:,:] * np.conj(W_mt[trial,ch2,:,:]) 