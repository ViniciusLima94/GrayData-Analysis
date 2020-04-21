#!/usr/bin/env python
# coding: utf-8

'''
    Set which parameters to use. Which monkey, which session and which type os trials to use.
    Set folders and recording info.
'''
import numpy as np
import h5py
import scipy.io as scio
import scipy.ndimage as simg

# Choosing which monkey to use, which session, which event and type.
nmonkey = 0
nses    = 0
ntype   = 0

#--------------------------------------------------------------------------
# Directories
#--------------------------------------------------------------------------
dirs = {'rawdata':'GrayLab/',
		'results':'Results/',
		'monkey' :['lucy', 'ethyl'],
		'session':'session01',
		'date'   :[['150128'], [],]}

#--------------------------------------------------------------------------
# Create session dicitionary
#--------------------------------------------------------------------------
session = {'dir'        :dirs['rawdata']+dirs['monkey'][nmonkey]+'/'+dirs['date'][nmonkey][nses]+'/'+str(dirs['session']+'/'),
		   'dir_out'    :dirs['results']+dirs['monkey'][nmonkey]+'/'+dirs['date'][nmonkey][nses]+'/'+str(dirs['session']+'/'),
		   'fname_base' :dirs['date'][nmonkey][nses],
		   'evt_names'  :['samplecor','sampleinc','samplecorinc'],
		   'evt_trinfo' :['sample_on','match_on'],
		   'evt_dt'     :[-0.65,3.00 ],
		   'fc'         :np.arange(6, 62, 2),
		   'df'         :4,
       'f_bands'    :np.array( [np.arange(0,2), np.arange(2,5), np.arange(5,13), np.arange(13, 28)] ),
		   'dt'         :250,
		   'step'       :25,}
           
# Create out folder
try:
    os.makedirs(session['dir_out'])
except:
    None
#--------------------------------------------------------------------------
# Recording and trials info dicitionaries
#--------------------------------------------------------------------------
info = ['recording_info.mat', 'trial_info.mat']
ri = scio.loadmat(session['dir']+info[0])['recording_info']
with h5py.File(session['dir']+info[1], 'r') as ti:
    ti = ti['trial_info']
    trial_info     = {'num_trials': int(ti['num_trials'][0,0]),
                  'trial_type': ti['trial_type'][:].T[0],
                  'behavioral_response': ti['behavioral_response'][:].T[0],
                  'sample_image': ti['sample_image'][:].T[0],
                  'nonmatch_image': ti['nonmatch_image'][:].T[0],
                  'match_image': ti['match_image'][:].T[0],
                  'reaction_time': ti['reaction_time'][:].T[0],
                  'sample_on': ti['sample_on'][:].T[0], #1th image is shown
                  'match_on': ti['match_on'][:].T[0],   #2nd image is shown
                  'sample_off': ti['sample_off'][:].T[0],}

recording_info = {'channel_count': ri['channel_count'].astype(int)[0][0],
                  'channel_numbers':ri['channel_numbers'][0,0][0],
                  'area':  ri['area'][0,0][0],
                  'fsample': ri['lfp_sampling_rate'].astype(int)[0][0],
                  'ms_mod': ri['ms_mod'][0,0][0],                        #
                  'slvr': ri['slvr'][0,0][0],}                           #