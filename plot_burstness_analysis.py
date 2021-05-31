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

import matplotlib
matplotlib.use('Agg')

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

xy   = scipy.io.loadmat('Brain Areas/lucy_brainsketch_xy.mat')['xy'] # Channels coordinates
d_eu = np.zeros(net.session_info['pairs'].shape[0])
for i in range(net.session_info['pairs'].shape[0]):
    c1, c2 = net.session_info['channels_labels'].astype(int)[net.session_info['pairs'][i,0]], net.session_info['channels_labels'].astype(int)[net.session_info['pairs'][i,1]]
    dx = xy[c1-1,0] - xy[c2-1,0]
    dy = xy[c1-1,1] - xy[c2-1,1]
    d_eu[i] = np.sqrt(dx**2 + dy**2)

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
bs_stats = hf['bs_stats'][:]

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

###############################################################################
# 5. Burstness statistics averaged for all links
###############################################################################

titles = ['Mean burst duration', 'Norm. total active time', 'CV']
for idx, q in tqdm( enumerate(q_list) ):
    plt.figure(figsize=(20,5))
    for k in range(3):
        plt.subplot(1,3,k+1)
        for j in range(len(net.bands)):
            p=bs_stats[idx][:,j,0,k]
            c=bs_stats[idx][:,j,1,k]
            d=bs_stats[idx][:,j,2,k]
            m=bs_stats[idx][:,j,3,k]
            nf=np.sqrt(net.super_tensor.shape[0])
            plt.errorbar(range(4), [p.mean(), c.mean(), d.mean(), m.mean()], 
                         [p.std()/nf, c.std()/nf, d.std()/nf, m.std()/nf])
            plt.xticks(range(4), ['baseline', 'cue', 'delay', 'match'], fontsize=15)
            plt.title(titles[k], fontsize=15)
        plt.legend(band_names)
    plt.savefig(
        os.path.join(path_st, f"stats_link_avg_{int(100*q)}_{dirs['date'][nmonkey][idx]}.png"),
        dpi=300)
    plt.close()

###############################################################################
# 6. Burstness statistics ploted as matrix Nodes x Nodes 
###############################################################################

nC = ses.data.attrs['nC'] # Number of channels

for idx, q in tqdm( enumerate(q_list) ):
    # Converting stats to matrix
    mu     = np.zeros([nC, nC, len(net.bands), len(stages)])  
    mu_tot = np.zeros([nC, nC, len(net.bands), len(stages)]) 
    CV     = np.zeros([nC, nC, len(net.bands), len(stages)]) 
    for j in range( net.session_info['pairs'].shape[0]):
        mu[net.session_info['pairs'][j,0], net.session_info['pairs'][j,1], :, :]     = bs_stats[idx][j,:,:,0]
        mu_tot[net.session_info['pairs'][j,0], net.session_info['pairs'][j,1], :, :] = bs_stats[idx][j,:,:,1]
        CV[net.session_info['pairs'][j,0], net.session_info['pairs'][j,1], :, :]     = bs_stats[idx][j,:,:,2]

    # Plotting
    ################################################ MU ################################################
    plt.figure(figsize=(15,15))
    count = 1
    for k in tqdm( range(len(net.bands)) ):
        for i in range(len(stages)):
            plt.subplot(len(net.bands),len(stages),count)
            aux = (mu[:,:,k,i]+mu[:,:,k,i].T)
            plt.imshow(aux, aspect='auto',
                       cmap='jet', origin='lower',
                       vmin=0, vmax=np.median(mu[:,:,k,:]+mu[:,:,k,:].transpose(1,0,2)*4) )
            plt.colorbar()
            if stages[i] == 'baseline': plt.yticks(range(nC), ses.data.roi.values)
            else: plt.yticks([])
            if k == 4: plt.xticks(range(nC), ses.data.roi.values, rotation=90)
            else: plt.xticks([])
            if k == 0: plt.title(stages[i], fontsize=15)
            if stages[i] == 'baseline': plt.ylabel(band_names[k], fontsize=15)
            #plt.colorbar()
            count+=1
    plt.tight_layout()
    plt.savefig(
        os.path.join(path_st, f"matrix_mu_{int(100*q)}_{dirs['date'][nmonkey][idx]}.png"),
        dpi=300)
    plt.close()
    ############################################## MU_tot ##############################################
    plt.figure(figsize=(15,15))
    count = 1
    for k in tqdm( range(len(net.bands)) ):
        for i in range(len(stages)):
            plt.subplot(len(net.bands),len(stages),count)
            aux = (mu_tot[:,:,k,i]+mu_tot[:,:,k,i].T)
            plt.imshow(aux, aspect='auto', cmap='jet', origin='lower', vmin=0, vmax=0.3)
            plt.colorbar()
            if stages[i] == 'baseline': plt.yticks(range(49), ses.data.roi.values)
            else: plt.yticks([])
            if k == 4: plt.xticks(range(49), ses.data.roi.values, rotation=90)
            else: plt.xticks([])
            if k == 0: plt.title(stages[i], fontsize=15)
            if stages[i] == 'baseline': plt.ylabel(band_names[k], fontsize=15)
            #plt.colorbar()
            count+=1
    plt.tight_layout()
    plt.savefig(
        os.path.join(path_st, f"matrix_mu_tot_{int(100*q)}_{dirs['date'][nmonkey][idx]}.png"),
        dpi=300)
    plt.close()
    ################################################ CV ################################################
    plt.figure(figsize=(15,15))
    count = 1
    for k in tqdm( range(len(net.bands)) ):
        for i in range(len(stages)):
            plt.subplot(len(net.bands),len(stages),count)
            aux = (CV[:,:,k,i]+CV[:,:,k,i].T)
            plt.imshow(aux**6, aspect='auto', cmap='jet', origin='lower', vmin=0, vmax=0.3)
            plt.colorbar()
            if stages[i] == 'baseline': plt.yticks(range(49), ses.data.roi.values)
            else: plt.yticks([])
            if k == 4: plt.xticks(range(49), ses.data.roi.values, rotation=90)
            else: plt.xticks([])
            if k == 0: plt.title(stages[i], fontsize=15)
            if stages[i] == 'baseline': plt.ylabel(band_names[k], fontsize=15)
            #plt.colorbar()
            count+=1
    plt.tight_layout()
    plt.savefig(
        os.path.join(path_st, f"matrix_cv_{int(100*q)}_{dirs['date'][nmonkey][idx]}.png"),
        dpi=300)
    plt.close()

###############################################################################
# 7. Burstness vs mean active time 
###############################################################################

for idx, q in tqdm( enumerate(q_list) ):
    plt.figure(figsize=(15,20))
    count = 1
    x_min, x_max = bs_stats[idx][...,0].min(), bs_stats[idx][...,0].max()
    y_min, y_max = bs_stats[idx][...,2].min(), bs_stats[idx][...,2].max()
    bins         = [np.linspace(x_min,x_max,20),np.linspace(y_min, y_max,20)]
    for j in tqdm( range(len(net.bands)) ):
        Hb, xb, yb = np.histogram2d(bs_stats[idx][:,j,0,0], bs_stats[idx][:,j,0,2],
                                    bins=bins, density = True  )
        for i in range(1,len(stages)):
            # Plotting top links
            plt.subplot(len(net.bands), len(stages)-1, count)
            H, xb, yb = np.histogram2d(bs_stats[idx][:,j,i,0], bs_stats[idx][:,j,i,2],
                                   bins=bins, density = True )
            plt.imshow(H-Hb, aspect='auto', cmap='RdBu_r', origin='lower',
                       extent=[1000*xb[0],1000*xb[-1],yb[0],yb[-1]],
                       interpolation='gaussian', vmin=-1000, vmax=1000)
            plt.colorbar()
            if j < 4 : plt.xticks([])
            if i > 1 : plt.yticks([])
            if j == 4: plt.xlabel('Mean burst dur. [ms]', fontsize=15)
            if j == 0: plt.title(f'{stages[i]}-baseline', fontsize=15)
            if i == 1: plt.ylabel('CV', fontsize=15)
            count += 1
    plt.tight_layout()
    plt.savefig(
        os.path.join(path_st, f"bst_mu_dist_{int(100*q)}_{dirs['date'][nmonkey][idx]}.png"),
        dpi=300)
    plt.close()

###############################################################################
# 8. Burstness vs euclidean distance 
###############################################################################

for idx, q in tqdm( enumerate(q_list) ):
    plt.figure(figsize=(15,20))
    count = 1
    x_min, x_max = d_eu.min(), d_eu.max()
    y_min, y_max = bs_stats[idx][...,2].min(), bs_stats[idx][...,2].max()
    bins         = [np.linspace(x_min,x_max,20),np.linspace(y_min, y_max,20)]
    for j in tqdm( range(len(net.bands)) ):
        Hb, xb, yb = np.histogram2d(d_eu, bs_stats[idx][:,j,0,2],
                                    bins=bins, density = True  )
        for i in range(1,len(stages)):
            # Plotting top links
            plt.subplot(len(net.bands), len(stages)-1, count)
            H, xb, yb = np.histogram2d(d_eu, bs_stats[idx][:,j,i,2],
                                   bins=bins, density = True )
            plt.imshow(H-Hb, aspect='auto', cmap='RdBu_r', origin='lower',
                       extent=[xb[0],xb[-1],yb[0],yb[-1]],
                       interpolation='gaussian', vmin=-0.02, vmax=0.02)
            plt.colorbar()
            if j < 4 : plt.xticks([])
            if i > 1 : plt.yticks([])
            if j == 4: plt.xlabel('Euclidian distance', fontsize=15)
            if j == 0: plt.title(f'{stages[i]}-baseline', fontsize=15)
            if i == 1: plt.ylabel('CV', fontsize=15)
            count += 1
    plt.tight_layout()
    plt.savefig(
        os.path.join(path_st, f"bst_eucl_dist_{int(100*q)}_{dirs['date'][nmonkey][idx]}.png"),
        dpi=300)
    plt.close()


