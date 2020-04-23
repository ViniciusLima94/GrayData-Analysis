#!/usr/bin/env python
# coding: utf-8

'''
    Code to compute the spectograms for each pair of gray's data, 
    based on Andrea's matlab code.
'''
import numpy as np 
import glob 
import scipy.signal as sig
import scipy.special as spe
import os
import sys
import mne.filter
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing

idx=3


nmonkey = 0
nses    = 6
ntype   = 0
#####################################################################################################
# Directories
#####################################################################################################
dirs = {'rawdata':'GrayLab/',
        'results':'Results/',
        'monkey' :['lucy', 'ethyl'],
        'session':'session01',
        'date'   :[['141014', '141015', '141205', '150128', '150211', '150304'], []]
        }
# Specify which trial to use
#trial_number = int( sys.argv[-1] ) 

#--------------------------------------------------------------------------
# Read output from read_lfp_data.py
#--------------------------------------------------------------------------
'''
readinfo = scio.loadmat(session['dir_out']+'readinfo.mat')
nP     = readinfo['nP'][0,0]
pairs  = readinfo['pairs']
indt   = readinfo['indt'][0]
tarray = readinfo['tarray'][0]
taxs   = readinfo['time'][0]
#data   = np.load('raw_lfp/'+dirs['session']+'_'+dirs['date'][nmonkey][nses]+'.npy')[trial_number,:,:]
'''

path = 'raw_lfp/'+dirs['monkey'][nmonkey]+'_'+'session01'+'_'+dirs['date'][nmonkey][idx]+'.npy'
LFP  = np.load(path, allow_pickle=True).item()


step = 25 
dt   = 250
fc   = np.arange(6, 62, 2)
df   = 4

nP      = LFP['info']['nP']
nT      = LFP['info']['nT']
pairs   = LFP['info']['pairs']
tarray  = LFP['info']['tarray']
indt    = np.arange(dt, LFP['data'].shape[2]-dt, step)
taxs    = LFP['info']['tarray'][indt]
fsample = LFP['info']['fsample']
data    = LFP['data']
dir_out = LFP['path']['dir_out']


def coh_matrix(data, nT, nP, session, time, L, as_mat=False, return_coh=False):
    '''
        Computes the coherence matrix (spectrogram) between a given pair
        numbered nP, and trial numbered nT.
        Inputs:
        data: Super tensor with LFP data for all trials and pairs.
        nT:   Trial number.
        nP:   Pair number.
        session: Session dictionary with info.
        time: Time array.
        T:    Number of time points.
        as_mat: If True will save as .mat.
        Outputs:
        File with spectogram for pair nP of trial nT.
    '''
    # Coherence matris time vs frequency
    print('Trial: '+str(nT)+', Pair: '+str(nP))
    coh = np.empty( [len(time), len(fc)] )
    # First LFP
    sig1 = data[pairs[nP, 0], :].copy()
    # Second LFP
    sig2 = data[pairs[nP, 1], :].copy()
    del data
    S = (1+1j)*np.zeros([3, L])
    for nf in range( fc.shape[0] ):
        bpfreq = np.array( [ fc[nf]-df, fc[nf]+df ] )
        f_low, f_high  = bpfreq[0], bpfreq[1]
        sig1f  = mne.filter.filter_data(sig1, fsample, f_low, f_high, method = 'iir', verbose=False, n_jobs=1)
        sig2f  = mne.filter.filter_data(sig2, fsample, f_low, f_high, method = 'iir', verbose=False, n_jobs=1)
        Sx     = sig.hilbert(sig1f)
        Sy     = sig.hilbert(sig2f)
        S[0, :] = np.multiply( Sx, np.conj(Sy) )
        S[1, :] = np.multiply( Sx, np.conj(Sx) )
        S[2, :] = np.multiply( Sy, np.conj(Sy) )
        Sm = sig.convolve2d(S.T, np.ones([dt, 1]), mode='same') 
        Sm = Sm[indt,:]
        coh[:, nf] = ( Sm[:, 0]*np.conj(Sm[:, 0]) /(Sm[:, 1]*Sm[:,2]) ).real
    # Saving data matrix for each trial and pair
    if return_coh==True:
        return coh
    '''
    else:
        if as_mat==False:
            file_name = session['dir_out']+dirs['session']+'_'+dirs['date'][nmonkey][nses]+'_trial_'+str(nT)+'_pair_'+str(pairs[nP, 0])+'_'+str(pairs[nP, 1])+'.dat'
            np.savetxt(file_name, coh.T)
        elif as_mat==True:
            file_name = session['dir_out']+dirs['session']+'_'+dirs['date'][nmonkey][nses]+'_trial_'+str(nT)+'_pair_'+str(pairs[nP, 0])+'_'+str(pairs[nP, 1])+'.mat'
            coh = {'trial':coh.T}
            scio.savemat(file_name, coh )
    '''
#--------------------------------------------------------------------------
# Compute in parallel the spectograms for each pair and trial
#--------------------------------------------------------------------------
#print('Trial = ' + str(trial_number))
#for ip in range(nP):
#    print('Pair = ' + str(ip))
#    coh_matrix(data, trial_number, ip, session, taxs, data.shape[1], as_mat=True)

for trial_number in range(540):
    print('Trial = ' + str(trial_number))
    #data   = np.load('raw_lfp/'+dirs['session']+'_'+dirs['date'][nmonkey][nses]+'.npy')[trial_number,:,:]
    Parallel(n_jobs=40)(delayed(coh_matrix)(data, trial_number, ip, None, taxs, data.shape[1], as_mat=True) for ip in range(nP))