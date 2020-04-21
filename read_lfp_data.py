#!/usr/bin/env python
# coding: utf-8

'''
    Read and prepare data for Spectral analysis.
'''
import numpy as np 
import glob 
import scipy.signal as sig
import scipy.io as scio
import scipy.special as spe
import h5py
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, 'params/')
from set_params_base import *

# Find zero time wrt to selected event
t0   = trial_info['sample_on']
t0_f = trial_info['sample_off']
t1   = trial_info['match_on']
# Printing info
print('Number of channels: ' + str(recording_info['channel_count']))
print('Sample frequency: ' + str(recording_info['fsample']) + ' Hz')
print('Number of trials: ' + str(trial_info['num_trials']))
#--------------------------------------------------------------------------
# Reading data
#--------------------------------------------------------------------------
files   = sorted(glob.glob(session['dir']+'/'+dirs['date'][nmonkey][nses]+'*'))

# Get only LFP channels does not contain slvr and ms_mod
indch  = (recording_info['slvr'] == 0) & (recording_info['ms_mod'] == 0)
indch  = np.arange(1, recording_info['channel_count']+1, 1)[indch]
# Trial type
if  session['evt_names'][ntype] == 'samplecor':
	# Use only completed trials and correct
	indt = (trial_info['trial_type'] == 1) & (trial_info['behavioral_response'] == 1)
	indt = np.arange(1, trial_info['num_trials']+1, 1)[indt]
elif session['evt_names'][ntype] == 'sampleinc':
	# Use only completed trials and incorrect
	indt = (trial_info['trial_type'] == 1) & (trial_info['behavioral_response'] == 0)
	indt = np.arange(1, trial_info['num_trials']+1, 1)[indt]
elif session['evt_names'][ntype] == 'samplecorinc':
	# Use all completed correct and incorrect trials 
	indt = (trial_info['trial_type'] == 1) 
	indt = np.arange(1, trial_info['num_trials']+1, 1)[indt]

# Stimulus presented
stimulus = trial_info['sample_image'][indt-1]-1

# Duration of the cue (trial dependent)
dcue = (t0_f - t0)[indt-1]
# Distance sample on match on (trial dependent)
dsm  = (t1 - t0)[indt-1]

# Record choices, i.e. find which oculomotor choice was performed
choice = np.nan*np.ones(trial_info['sample_image'].shape[0])
# Incorrect response means the monkey chose the nonmatch image
ind = trial_info['behavioral_response'] == 0
choice[ind] = trial_info['nonmatch_image'][ind]
# Correct response means the monkey chose the match image
ind = trial_info['behavioral_response'] == 1
choice[ind] = trial_info['match_image'][ind]
trial_info['choice'] = choice

# Data matrix dimensions
L = int( (1000*session['evt_dt'][1] - 1000*session['evt_dt'][0]) + 1 ) # Number of time points
T = len(indt)  # Number of trials
C = len(indch) # Number of channels

data = np.empty([T, C, L])   # LFP data
time = np.empty([T, L])      # Time vector for each trial
trialinfo = np.empty([T, 5]) # Info about each trial

# Loop over trials
print('Reading data...')
i = 0
delta_t   = 1.0 / recording_info['fsample']
for nt in indt:
    # Reading file with LFP data
    f     = h5py.File(files[nt-1], "r")
    lfp_data = np.transpose( f['lfp_data'] )
    f.close()
    # Beginning and ending index
    indb = int(t0[nt-1] + 1000*session['evt_dt'][0])
    inde = int(t0[nt-1] + 1000*session['evt_dt'][1])
    # Find time index
    ind = np.arange(indb, inde+1).astype(int)
    # Super tensor containing LFP data, dimension NtrialsxNchannelsxTime
    data[i] = lfp_data[indch-1, indb:inde+1]
    # Time vector, one for each trial, dimension NtrialsxTime
    time[i] = np.arange(session['evt_dt'][0], session['evt_dt'][1]+delta_t, delta_t)
    # Keep track of [ real trial number, sample image, choice, outcome (correct or incorrect), reaction time ]
    trialinfo[i] = np.array([nt, trial_info['sample_image'][nt-1], trial_info['choice'][nt-1], trial_info['behavioral_response'][nt-1], trial_info['reaction_time'][nt-1]/1000.0])
    i = i + 1

print('Saving LFP matrices...')
np.save('raw_lfp/'+dirs['session']+'_'+dirs['date'][nmonkey][nses]+'.npy', data)

# Plot LFP signals
'''
trials   = [0, 100, 250, 500]
channels = [0, 5, 20, 30, 45] 
colors   = {0:'b', 5:'g', 20:'m', 30:'r', 45:'y'}
count    = 1
plt.figure(figsize=(13,10))
for t in trials:
    for c in channels:
        plt.subplot(len(trials), len(channels), count)
        plt.subplots_adjust(wspace=.1, hspace=.1)
        plt.box(False)
        plt.plot(data[t,c,1000:2000], c=colors[c])
        plt.xticks([])
        plt.yticks([])
        if c == 0:
            plt.ylabel('Trial ' + str(t))
        if t == 0:
            plt.title('Ch. ' + str(c))
        count += 1
plt.savefig('lfp_sample.pdf', dpi = 600)
'''

# Original channel's labels
labels = recording_info['channel_numbers'][indch-1].astype(str)
# Parameters
nP = int( spe.comb(C, 2) )
pairs = np.zeros([nP, 2], dtype=int)
count = 0
for i in range(C):
	for j in range(i+1, C):
		pairs[count, 0] = i 
		pairs[count, 1] = j
		count += 1
# Number of trials
nT = T

#indt = np.linspace(0, data.shape[2]-1, 200, dtype=int)
indt = np.arange(session['dt'], data.shape[2]-session['dt']+session['step']-session['step'], session['step'])
taxs = time[0][indt]
readinfo = {'nP':nP, 'nT':nT, 'pairs':pairs, 'trialinfo':trialinfo, 'tarray': time[0],'time':taxs, 'channels_labels': labels, 'indt':indt, 'dcue': dcue, 'dsm': dsm, 'stim':stimulus}
scio.savemat(session['dir_out']+'readinfo.mat', readinfo)