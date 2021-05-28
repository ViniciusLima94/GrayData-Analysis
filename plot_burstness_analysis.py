r'''
Script to plot burstness statistics computed in _burstness_analysis.py_ .
'''
import matplotlib.pyplot        as       plt
import seaborn                  as       sns

from   GDa.temporal_network     import   temporal_network
from   GDa.util                 import   smooth

import sys
import os
import h5py

from   config                   import   *
from   tqdm                     import   tqdm

###############################################################################
# Setting parameters to read and save data as usual
###############################################################################

idx     = int(sys.argv[-1]) # Index to acess the desired session
nmonkey = 0
nses    = 1
ntype   = 0

# Path in which to save coherence data
path_st = os.path.join('figures', str(dirs['monkey'][nmonkey]), str(dirs['date'][nmonkey][idx]), f'session0{nses}')
if not os.path.exists(path_st):
    os.makedirs(path_st)

# Bands names
band_names  = [r'band 1', r'band 2', r'band 3', r'band 4', r'band 5']
stages      = ['baseline', 'cue', 'delay', 'match']

###############################################################################
# Defining parameters to instantiate temporal network
###############################################################################

def set_net_params(trial_type=None, behavioral_response=None, relative=None, q=None):
    r'''
    Return dict. with parameters to instantiate temporal network object.
    '''

    # Default parameters plus params passed to method trial type 1
    return dict( data_raw_path='GrayLab/', tensor_raw_path='Results/', monkey=dirs['monkey'][nmonkey],
                 session=1, date=dirs['date'][nmonkey][idx], trial_type=trial_type,
                 behavioral_response=behavioral_response, relative=relative, q=q, wt=(30,30), verbose=False )

###############################################################################
# 1. Trial averaged super-tensor for each band 
###############################################################################

# Instantiating a temporal network object without thresholding the data
net =  temporal_network( **set_net_params([1], [1]) )

plt.figure(figsize=(20,30))
aux = scipy.stats.zscore(net.super_tensor.mean(dim='trials'), axis=-1)
for i in range(len(band_names)):
    plt.subplot(len(band_names), 1, i+1)
    plt.imshow(aux[:,i,:], 
               aspect = 'auto', cmap = 'RdBu_r', origin = 'lower',
               extent = [0, net.tarray[-1], 1, net.session_info['nP']], 
               vmin=-4, vmax=4)
    plt.colorbar()
    plt.ylabel('Pair Number', fontsize=15)
    plt.title(band_names[i])
plt.xlabel('Time [a.u.]', fontsize=15)
plt.tight_layout()
plt.savefig(
    os.path.join(path_st, f"trial_averaged_super_tensor_{dirs['date'][nmonkey][idx]}.png"),
    dpi=300)
plt.close()

###############################################################################
# 2. Evoked-response potentials
###############################################################################

plt.figure(figsize=(20,30))
# Average activation sequences over links
mu_filtered_super_tensor = net.super_tensor.mean(dim='links')
for i in range(len(band_names)):
    plt.subplot(len(band_names), 1, i+1)
    for t in range(net.super_tensor.shape[2]):
        plt.plot(net.tarray, 
        mu_filtered_super_tensor.isel(trials=t, bands=i).values, 
        color='b', lw=.1)
    plt.plot(net.tarray, 
             mu_filtered_super_tensor.isel(bands=i).median(dim='trials'),
            color='k', lw=3)
    plt.plot(net.tarray, 
         mu_filtered_super_tensor.isel(bands=i).quantile(q=5/100,dim='trials'),
        color='r', lw=3)
    plt.plot(net.tarray, 
    plt.title(band_names[i])
    mu_filtered_super_tensor.isel(bands=i).quantile(q=95/100,dim='trials'),
    color='r', lw=3)
    plt.xlim([net.tarray[0],net.tarray[-1]])
plt.xlabel('Time [s]', fontsize=15)
plt.tight_layout()
plt.savefig(
    os.path.join(path_st, f"evoked_reponse_potential_{dirs['date'][nmonkey][idx]}.png"),
    dpi=300)
plt.close()

###############################################################################
# 3. Burstness statistics dependence on the threshold
###############################################################################

###############################################################################
# 4. Burstness statistics distributions
###############################################################################

q_list = np.array([0.3, 0.5, 0.8, 0.9]) # Overwriting q_list

for q in tqdm( q_list ):

