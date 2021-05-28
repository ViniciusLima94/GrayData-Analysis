r'''
Script to plot burstness statistics computed in _burstness_analysis.py_ .
'''
import matplotlib.pyplot        as       plt
import seaborn                  as       sns

from   GDa.temporal_network     import   temporal_network
from   GDa.util                 import   smooth

import scipy
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


# Bands names
band_names  = [r'band 1', r'band 2', r'band 3', r'band 4', r'band 5']
stages      = ['baseline', 'cue', 'delay', 'match']

###############################################################################
# Reading file with the computed statistics (burstness_stats.h5)
###############################################################################
# Path in which to save coherence data
path_st = os.path.join('figures', str(dirs['monkey'][nmonkey]), str(dirs['date'][nmonkey][idx]), f'session0{nses}')
if not os.path.exists(path_st):
    os.makedirs(path_st)

path_stats = os.path.join('Results', str(dirs['monkey'][nmonkey]), str(dirs['date'][nmonkey][idx]), f'session0{nses}')
hf = h5py.File(os.path.join(path_stats, 'burstness_stats.h5'), 'r')

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
for i in tqdm( range(len(band_names)) ):
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
for i in tqdm( range(len(band_names)) ):
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
             mu_filtered_super_tensor.isel(bands=i).quantile(q=95/100,dim='trials'),
             color='r', lw=3)
    plt.title(band_names[i])
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
cv = hf['q_dependence'][:]

q_list = np.arange(0.2, 1.0, 0.1)

titles = ['Mean burst duration', 'Norm. total active time', 'CV']
plt.figure(figsize=(12,15))
count = 1
for i in tqdm( range(len(net.bands)) ):
    for k in range(3):
        plt.subplot(5,3,count)
        for s in range(len(stages)):
            v_median = np.median(  cv[:,i,k,s,:],axis=0)
            v_q05    = np.quantile(cv[:,i,k,s,:], 5/100, axis=0)
            v_q95    = np.quantile(cv[:,i,k,s,:], 95/100, axis=0)
            diq      = (v_q95-v_q05)/2
            plt.plot(q_list, v_median, label=stages[s])
            plt.fill_between(q_list, v_median-diq, v_median+diq, alpha=0.2)
        count +=1
        plt.xlim([0.2,0.9])
        if k == 0: plt.ylabel(f'Band {i}', fontsize=15)
        if i == 0: plt.title(titles[k], fontsize=15)
        if i < 4:  plt.xticks([])
        if i == 0 and k==0: plt.legend()
        if i == 4: plt.xlabel('q', fontsize=15)
plt.tight_layout()
plt.savefig(
    os.path.join(path_st, f"q_dependence_{dirs['date'][nmonkey][idx]}.png"),
    dpi=300)
plt.close()

###############################################################################
# 4. Burstness statistics distributions
###############################################################################
cv = hf['bs_stats'][:]

q_list = np.array([0.3, 0.5, 0.8, 0.9]) # Overwriting q_list

for idx, q in tqdm( enumerate(q_list) ):
    titles = ['Mean burst duration', 'Norm. total active time', 'CV']
    bins   = [np.linspace(0.07,0.14,50), np.linspace(0.12,0.30,50), np.linspace(0.55,0.85,50) ]
    plt.figure(figsize=(20,20))
    count = 1
    for i in range(len(net.bands)):
        for k in range(3):
            plt.subplot(5,3,count)
            for j in range(len(stages)):
                sns.kdeplot(data=bs_stats[idx][:,i,j,k], shade=True)
                if i==0: plt.title(titles[k], fontsize=15)
                if count in [1,4,7,10,13]: plt.ylabel(f'Band {i}', fontsize=15)
            plt.legend(['baseline','cue','delay','match'])
            count+=1
    plt.tight_layout()
    plt.savefig(
        os.path.join(path_st, f"stats_dists_q_{int(100*q)}_{dirs['date'][nmonkey][idx]}.png"),
        dpi=300)
    plt.close()
