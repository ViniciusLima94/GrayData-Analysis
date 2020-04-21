# Data Analysis

Description: Codes to perform spectral analysis on the Gray data. In particular, using the script "spectral_analysis.py" we will compute the pairwise coherence in the time-frequency domain for the data available.

## Dependencies

- python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
- pip install --user mne
- pip install --user h5py

## The data

The data should be transfered to the /data directory with the path organized in the following structure: **data/monkey_name/date/session/mat_files**

As an example, the data I'm using is under the path: **data/lucy/150128/session1/**

## Setting up parameters

The parameters should be set on the scripts "params/set_params_base.py", and "params/set_params.py". In general the parameter values are set as python dictionaries. First you should set the directiories we will work on, such as the data directiories cited above, and the directiores were the results will be saved, this can be done in "params/set_params_base.py".

```
# Choosing which monkey to use, which session, which event and type.
nmonkey = 0
nses    = 0
ntype   = 0

#--------------------------------------------------------------------------
# Directories
#--------------------------------------------------------------------------
dirs = {'rawdata':'data/',
	'results':'Results/',
	'monkey' :['lucy', 'ethyl'],
	'session':'session01',
	'date'   :[['150128'], [],]}
```

For now the variables indicating indicating the number of the monkey (nmonkey), the number of the session array (nses) and the type (ntype) should all be zero because I was working with only one session, for more sessions where the keys of the dirs dictionary has length greather than 1 this may vary. Finally, in "params/set_params_base.py" we wil read and store the MATLAB structures 'trial_info.mat', and 'recording_info.mat', those files have information about the experiment such as the number of trials, frequency sample, the behavioral response of the monkey and so on, all those informations will also be stored in python dictionaries.

```
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
```

All the paramters on the dictionaries above such as number of trials used, number of frequency and time points, channels labels for the data session used will be kept in variable on "params/set_params.py" for later use. All those vairable will be save on the output directory **Results/....**

## Reading the data

After setting up the parameters in "params/set_params_base.py" you should be ready to read the data with the "read_lfp_data.py", for only one session of the data you can run it with: **ipython read_lfp_data.py** (for more session you might need to paralelize the code and run it on a cluster using a bash script).

This script will print those informations on the screen:

```
Number of channels: 56
Sample frequency: 1000 Hz
Number of trials: 1006
Number of used channels: 49
Number of used trials: 540
Reading data...
Saving LFP matrices...
```

After running this code it will save the dictionary bellow:

```
readinfo = {'nP':nP, 'nT':nT, 'pairs':pairs, 'trialinfo':trialinfo, 'tarray': time[0],'time':taxs, 'channels_labels': labels, 'indt':indt, 'dcue': dcue, 'dsm': dsm, 'stim':stimulus}
```

Where,

```
nP = Number of pairs of channels;
nT = Number of trials used;
pairs = Index of each pair of channel (not actual label or anatomical position)
trialinfp = Array containing the real trial number, sample image, choice, outcome (correct or incorrect), reaction time
tarray = Array contained all the time points 
taxs = tarray = Array contained only the time points sampled (in relation to the cue onset or match onset)
channels_labels = Contain the real channel labels
dcue = Array containing the duration of the cue for each used trial in this session
dsm = Array containing the time difference between sample and match onset
stim = Array containing the stimulus used in each trial in this session
```

Finally, it will store a tensor in the numpy format ".npy" in the "raw_lfp/" directory. This tensor will have dimensions [Number of used trials, Number of used channels, Time].

The tensor file will be named **'raw_lfp/session_number_date.npy'**, so in the data I'm using it would be: **raw_lfp/session01_150128.npy**, this can be read with numpy using the following code:

```
import numpy as np
np.load('raw_lfp/session01_150128.npy', allow_pickle=True).item()
```

This data is save aligned in relation to the cue onset.

## Computing pairwise time-frequency coherence

After the steps above we can run the script "spectral_analysis.py" to compute the pairwise time-frequency coherence, it will generate a bunch of files on the output directory (created when we first runed the "read_lfp_data.py" script). Those files will be named as: **sessNumber_date_trialNumber_pair_index1_index2.mat**, note that index1 and index2 are the same indexes we saved on the variable "pairs" and not the actual channel labels (those can be retrieved using the labels array). Just for an example, and output file for the data I'm using has the name: **session01_150128_trial_0_pair_0_3.mat** (you can save it in npy if you want).

You will want to run this script on a cluster, to compute the coherence matrices independently run the bash script "run_coh.sh" with **sbatch run_coh.sh**
