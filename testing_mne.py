#####################################################################################################
# Compare the coherence matrices
#####################################################################################################
import os
import numpy                as np 
import matplotlib.pyplot    as plt

#####################################################################################################
# Loading session data
#####################################################################################################
session_data = np.load('raw_lfp/lucy_session01_150128.npy', allow_pickle=True).item()
pairs        = session_data['info']['pairs']
dir_out      = session_data['path']['dir_out']

trial        = np.random.randint(0, 540)
pair         = np.random.randint(0, 1176)
ch1, ch2     = pairs[pair,0], pairs[pair,1]

path1        = os.path.join(dir_out, 'trial_' +str(trial) + '_ch1_' + str(ch1) + '_ch2_' + str(ch2) +'.npy')
path2        = os.path.join(dir_out, 'trial_' +str(trial) + '_pair_'+ str(pair) +'.npy')

coh1         = np.load(path1, allow_pickle=True).item()['coherence']
coh2         = np.load(path2, allow_pickle=True).item()['coherence']

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(coh1.real, aspect='auto',cmap='jet',origin='lower')
plt.subplot(1,2,2)
plt.imshow(coh2.real, aspect='auto',cmap='jet',origin='lower')
plt.show()