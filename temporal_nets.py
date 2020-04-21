#!/usr/bin/env python
# coding: utf-8

'''
Notebook with the main results of the analysis found in:
1. spectral_analysis_notebook;
2. clustering_analysis_notebook;
3. temporal_nets_analysis_notebook
Containing figures used in the report (discussion in report file)
'''
# Importing modules 
import numpy as np 
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.io as scio
import scipy.ndimage as simg
import scipy
import seaborn as sns
from scipy import stats
import scipy.signal as sig
import mne.filter
import h5py
import sys
sys.path.insert(0, 'params/')
from set_params import *
import pycwt as wavelet
from pycwt.helpers import find

#Auxiliar functions to perform analysis
from tools import *

import seaborn as sns

from MulticoreTSNE import MulticoreTSNE as TSNE

import networkx as nx

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler

#--------------------------------------------------------------------------
# Building super tensor
#--------------------------------------------------------------------------
''' 
Super tensor, dimensions [number of pairs, number of frequency points, number of observations (time*trials)]
'''
#nT=10
ST = np.zeros([nP, nF, nt*nT]) 
'''
To build the super tensor first we read all the spectograms for each pair, and each trial
'''
for i in range(nP):
    for trial in range(nT):
        coh = loadcohm(dirs['session'], dirs['date'][nmonkey][nses], trial, pairs[i,0], pairs[i,1], session['dir_out'])
        ST[i, :, trial*nt:(trial+1)*nt] = coh
# Save super tensor into npy file
np.save('super_tensor/'+dirs['session']+'_'+dirs['date'][nmonkey][nses]+'.npy', data)
'''
Read saved super tensor instead of building it
'''
#np.save('super_tensor/'+dirs['session']+'_'+dirs['date'][nmonkey][nses]+'.npy', data)
#--------------------------------------------------------------------------
#  Averaging frequency bands of interest in the super tensor
#--------------------------------------------------------------------------
ST_averaged = np.zeros([nP, Nbands, ST.shape[2]])
plt.figure(figsize=(10,8))
for i in range(Nbands):
    ST_averaged[:, i, :] = ST[:, session['f_bands'][i]].mean(axis=1)
    plt.subplot(2,2,i+1)
    plt.imshow(ST_averaged[:,i,500:1000], aspect='auto', cmap='jet', extent=[0, 500, 1176, 0], vmin=0, vmax=1)
    plt.xlabel('Observations')
    plt.ylabel('Pair number')
    plt.title(fbands[i])
    plt.colorbar()
plt.tight_layout()
plt.savefig('img/super_tensor.pdf', dpi = 600)
plt.close()
#--------------------------------------------------------------------------
# Determining coherence threshold for each frequency band
#--------------------------------------------------------------------------
# Pooled histogram for each frequency band
bin_values = np.linspace(0, 1, 100) # Bin values
thr        = [] # Threshold do be used in the temporal network analysis
q          = 85
for n in range(Nbands):
    pool = ST_averaged[:, n, :].flatten(order='C')
    plt.subplot(2,2,n+1)
    qrt  = stats.mstats.mquantiles(pool, prob=q/100) # 3rd quartile of the data
    thr.append(qrt) # Threshold equals the 3rd quartier
    hist = np.histogram( pool, bins=bin_values)
    plt.plot(hist[1][1:], hist[0])
    plt.vlines(qrt, 0, hist[0].max()+0.001, lw=2, color='r', linestyles='--')
    plt.xlim([0,1])
    plt.ylim([0, hist[0].max()+0.001])
    plt.title('Pooled histograms:' + fbands[n])
    plt.ylabel('#')
    plt.xlabel('Coherence')
plt.legend(['Histogram', 'Threshold: ' +  str(q) + '%'])
plt.tight_layout()
plt.savefig('img/coherence_threshold.pdf', dpi = 600)
plt.close()
#--------------------------------------------------------------------------
# Analysis of the temporal networks
#--------------------------------------------------------------------------

# Task stages
# time < 0                               : Before cue
# 0 <= time < dcue(T)                    : Cue
# dcue(T) <= time < (1-f)*dsm(T)+f*dc(T) : Early delay
# (1-f)*dsm(T)+fdc(T) <= time < dsm(T)   : Late delay
# time >= dsm(T)+dc(T)                   : Match

def index_stage(stage, f):
	if stage == 'before_cue':
		use = time < 0
	elif stage == 'cue':
		use = (0 <= time)*(time < dcue)
	elif stage == 'early_delay':
		use = (dcue <= time)*(time < dcue + f*(dsm-dcue))
	elif stage == 'late_delay':
		use = (dcue + f*(dsm-dcue) <= time)*(time < dsm)
	elif stage == 'match':
		use = (time >= dsm)*(time <= dsm+dcue)
	else:
		print('Invalid stage')
	return use

def separate_tensor_by_task_stage(super_tensor, stage, dcue, dsm, f):
	'''
		Separates the super tensor by task stage.
		Inputs:
		super_tensor: Multidimensional array containing the super tensor
		stage: Which stage to separate
		dcue: Trial dependent cue duration
		trial: Which trial to use
	'''	

	use = index_stage(stage, f)

	# Keeping only observation in a given task stage
	idxs = np.array(list(use)*nT)           # Index of observations in the chosen stage
	new_time_axis = time[use]               # New time axis for the chosen stage      
	new_nt        = new_time_axis.shape[0]  # New number of time points
	C    = super_tensor[:,:,idxs]
	# Print info
	print('Super tensor with ' + str(C.shape[0]) + ' links, ' + str(C.shape[1]) + ', and ' + str(C.shape[2]) + ' observations.')
	return C, new_time_axis, new_nt

print('Separating super tensor for each task stage...')
C = {}
new_time_axis = {}
new_nt = {}
for stage in stages:
	print(stage + ': ')
	C[stage], new_time_axis[stage], new_nt[stage] = separate_tensor_by_task_stage(ST_averaged, stage, dcue, dsm, f)
#--------------------------------------------------------------------------
# Constructing weighted and binary networks using threshold
#--------------------------------------------------------------------------
print('Generationg weighted and binary networks for each task stage')
W = {}
B = {}
for stage in stages:
	print(stage+'...')
	W[stage] = np.zeros([nP, Nbands, C[stage].shape[2]]) # Weighted network
	B[stage] = np.zeros([nP, Nbands, C[stage].shape[2]]) # Binary network
	for n in range(Nbands):
	    W[stage][:,n,:] = np.multiply( (C[stage][:,n,:]>thr[n]), C[stage][:,n,:] ) 
	    B[stage][:,n,:] = (C[stage][:,n,:]>thr[n]).astype(int)
	    
	# Plot binary and weighted network
	'''
	plt.figure(figsize=(15,15))
	count = 1
	for n in range(Nbands):
	    plt.subplot(4,2,count)
	    plt.imshow(B[stage][:,n,:], aspect='auto', cmap='gray', extent=[0, B[stage].shape[2], 1176, 0])
	    plt.xlabel('Observations')
	    plt.ylabel('Pair number')
	    plt.title(fbands[n])
	    plt.colorbar()
	    count += 1
	    plt.subplot(4,2,count)
	    plt.imshow(W[stage][:,n,:], aspect='auto', cmap='jet', extent=[0, W[stage].shape[2], 1176, 0], vmin=0, vmax=1)
	    plt.xlabel('Observations')
	    plt.ylabel('Pair number')
	    plt.title(fbands[n])
	    plt.colorbar()
	    count += 1
	plt.tight_layout()
	plt.savefig('img/raster_plots_'+stage+'.pdf', dpi = 600)
	plt.close()count = 1
for stage in stages:
	for n in range(Nbands):
		plt.subplot(len(stages), Nbands, count)
		plt.imshow(B[stage][:,n,:], aspect='auto', cmap='gray')
		plt.xlim([0, 500])
		if stage == 'before_cue':
			plt.title(fbands[n])
		if stage!='match':
			plt.xticks([])
		if n > 0:
			plt.yticks([])
		count += 1
plt.savefig('img/raster_plots.pdf', dpi = 600)
	'''
'''
c = ['r-o', 'm-o', 'g-o', 'b-o', 'k-o']
names = ['Baseline', 'Cue', 'Early Delay', 'Late Delay', 'Match']
i = 0
cont=0
for stage in stages:
	for t in range(new_nt[stage]):
		plt.figure()
		plt.imshow(ethyl_brainsketch)
		idx = B[stage][:,3,t].astype(bool)
		ch1 = labels[pairs[idx, 0]].astype(int)
		ch2 = labels[pairs[idx, 1]].astype(int)
		p1  = [xy[ch1-1,0], xy[ch2-1,0]]
		p2  = [xy[ch1-1,1], xy[ch2-1,1]]
		plt.plot(p1, p2, c[i], lw=.5)
		plt.title(names[i])
		plt.xticks([])
		plt.yticks([])
		plt.savefig('/home/vinicius/Documentos/net_movie/'+str(cont)+'.png')
		plt.close()
		cont += 1
	i+=1 
'''
count = 1
for stage in stages:
	for n in range(Nbands):
		plt.subplot(len(stages), Nbands, count)
		plt.imshow(B[stage][:,n,:], aspect='auto', cmap='gray')
		plt.xlim([0, 500])
		if stage == 'before_cue':
			plt.title(fbands[n])
		if stage!='match':
			plt.xticks([])
		if n > 0:
			plt.yticks([])
		count += 1
plt.savefig('img/raster_plots.pdf', dpi = 600)
plt.close()
#--------------------------------------------------------------------------
# Raster plot for nodes
#--------------------------------------------------------------------------

# Creating nodes raster plots, if a given link (i,j) where i and j are both
# channels is on then i and j are considered on
B_node = {}
W_node = {}
for stage in stages:
	
	B_node[stage] = np.zeros([nC, Nbands, C[stage].shape[2]]) # Binary network for nodes
	W_node[stage] = np.zeros([nC, Nbands, C[stage].shape[2]]) # Strength network for nodes

	for n in range(Nbands):
		for c in range(nC):
			# Find index of pairs in which channel c participates
			idx  = np.logical_or(pairs[:,0]==c, pairs[:,1]==c)
			# Find index where those pairs are active
			binary   = B[stage][idx,n,:].sum(axis = 0)
			weighted = W[stage][idx,n,:].sum(axis = 0) 
			B_node[stage][c,n,:] = (binary>0).astype(int)
			W_node[stage][c,n,:] = weighted

	# Plot binary and weighted network
	'''
	plt.figure(figsize=(15,15))
	count = 1
	for n in range(Nbands):
		plt.subplot(4,2,count)
		plt.imshow(B_node[stage][:,n,:], aspect='auto', cmap='gray', extent=[0, B_node[stage].shape[2], nC, 0])
		plt.xlim([500,1000])
		plt.xlabel('Observations')
		plt.ylabel('Node number')
		plt.title(fbands[n])
		plt.colorbar()
		count += 1
		plt.subplot(4,2,count)
		plt.imshow(W_node[stage][:,n,:], aspect='auto', cmap='jet', extent=[0, B_node[stage].shape[2], nC, 0])
		plt.xlim([500,1000])
		plt.colorbar()
		count += 1
	plt.tight_layout()
	plt.savefig('img/raster_plots_nodes_'+stage+'.png')
	plt.close()
	'''
count = 1
for stage in stages:
	for n in range(Nbands):
		plt.subplot(len(stages), Nbands, count)
		plt.imshow(B_node[stage][:,n,:], aspect='auto', cmap='gray')
		plt.xlim([0, 500])
		if stage == 'before_cue':
			plt.title(fbands[n])
		if stage!='match':
			plt.xticks([])
		if n > 0:
			plt.yticks([])
		count += 1
plt.savefig('img/raster_plots_node.pdf', dpi = 600)
plt.close()
#--------------------------------------------------------------------------
# Measuring activation time for each link (within each trial and band)
#--------------------------------------------------------------------------
def count_transitions(seq):
    '''
    Count total of activations within the sequence, also return the duration of each sequence
    '''
    count = 0
    count_len = 0
    seq_len = []
    if seq[0] == 1:
        count += 1
        count_len += 1
    for i in range(1, seq.shape[0]):
        if seq[i]==1 and seq[i-1]==1:
            count_len += 1
            if i == seq.shape[0]-1:
                seq_len.append(count_len) 
        elif seq[i]==1 and seq[i-1]==0:
            count_len += 1
            count += 1
        elif seq[i]==0 and seq[i-1]==1:
            seq_len.append(count_len)
            count_len = 0  
    if seq[-1] == 1 and seq[-2] == 0:
        count += 1
        seq_len.append(count_len) 

    return float(count), np.array(seq_len)

act_time = {}
for stage in stages:
	
	# Store link average active time, std of active time and active time CV  and total active fraction and mean strength per trial, std strength per trial, cv strength per trial
	act_time[stage] = np.zeros([Nbands, nT, nP,7]) 

	for n in range(Nbands):
	    for trial in range(nT):
	        for i in range(nP):
	            # Index indicating wheter link i is "on" (1) or "off" (0)
	            idx      = B[stage][i,n,trial*new_nt[stage]:(trial+1)*new_nt[stage]]
	            _, seq_len  = count_transitions(idx)
	            if seq_len.shape[0] > 0:
	                act_time[stage][n, trial, i, 0] = seq_len.mean()/ float(new_nt[stage])
	                act_time[stage][n, trial, i, 1] = seq_len.std()
	                act_time[stage][n, trial, i, 2] = seq_len.std() / seq_len.mean()
	                act_time[stage][n, trial, i, 3] = idx.sum()     / float(new_nt[stage])
	            else:
	                act_time[stage][n, trial, i, 0] = 0
	                act_time[stage][n, trial, i, 1] = 0
	                act_time[stage][n, trial, i, 2] = 0
	                act_time[stage][n, trial, i, 3] = 0
	            act_time[stage][n, trial, i, 4] = W[stage][i,n,trial*new_nt[stage]:(trial+1)*new_nt[stage]].mean()
	            act_time[stage][n, trial, i, 5] = W[stage][i,n,trial*new_nt[stage]:(trial+1)*new_nt[stage]].std()
	            if act_time[stage][n, trial, i, 4] > 0:
	            	act_time[stage][n, trial, i, 6] = act_time[stage][n, trial, i, 5]/act_time[stage][n, trial, i, 4] 
	            else:
	            	act_time[stage][n, trial, i, 6] = 0

	np.save(stage+'_act_time.npy', act_time[stage])

#--------------------------------------------------------------------------
# Measuring activation time for each node (within each trial and band)
#--------------------------------------------------------------------------

act_time_node = {}
for stage in stages:
	
	# Store link average active time, std of active time and active time CV  and total active fraction 
	act_time_node[stage] = np.zeros([Nbands, nT, nC,4]) 

	for n in range(Nbands):
	    for trial in range(nT):
	        for i in range(nC):
	            # Index indicating wheter link i is "on" (1) or "off" (0)
	            idx      = B_node[stage][i,n,trial*new_nt[stage]:(trial+1)*new_nt[stage]]
	            _, seq_len  = count_transitions(idx)
	            if seq_len.shape[0] > 0:
	                act_time_node[stage][n, trial, i, 0] = seq_len.mean()/ float(new_nt[stage])
	                act_time_node[stage][n, trial, i, 1] = seq_len.std()
	                act_time_node[stage][n, trial, i, 2] = seq_len.std() / seq_len.mean()
	                act_time_node[stage][n, trial, i, 3] = idx.sum()     / float(new_nt[stage])
	            else:
	                act_time_node[stage][n, trial, i, 0] = 0
	                act_time_node[stage][n, trial, i, 1] = 0
	                act_time_node[stage][n, trial, i, 2] = 0
	                act_time_node[stage][n, trial, i, 3] = 0

	np.save(stage+'_act_time_node.npy', act_time_node[stage])

#--------------------------------------------------------------------------
# Building adjacency matrices for each task stage, FB and observation
#--------------------------------------------------------------------------
# B[stage] = np.zeros([nP, Nbands, C[stage].shape[2]])
# Adjacency matrices for each task stage, frequency band and observation
Madj   = {}
for stage in stages:
	Madj[stage] = np.zeros([nC, nC, Nbands, C[stage].shape[2]])
	for n in range(Nbands):
		for t in range(C[stage].shape[2]):
			idx = B[stage][:,n,t]==1 # All active pairs for band n and observation t indexes
			ch1 = pairs[idx, 0]
			ch2 = pairs[idx, 1]
			Madj[stage][ch1, ch2, n, t] = 1
			Madj[stage][:, :, n, t] += Madj[stage][:, :, n, t].T

# Weighted
Madjw = {}
for stage in stages:
	Madjw[stage] = np.zeros([nC, nC, Nbands, C[stage].shape[2]])
	for n in range(Nbands):
		for t in range(C[stage].shape[2]):
			idx = W[stage][:,n,t]>0 # All active pairs for band n and observation t indexes
			ch1 = pairs[idx, 0]
			ch2 = pairs[idx, 1]
			Madjw[stage][ch1, ch2, n, t] = W[stage][idx,n,t]
			Madjw[stage][:, :, n, t] += Madjw[stage][:, :, n, t].T

#--------------------------------------------------------------------------
# Computing coreness
#--------------------------------------------------------------------------
largest_k = {}
# Plot each with size propotional to the largest k-core in which it participates
for stage in stages:
	largest_k[stage] = np.zeros([nC, Nbands, C[stage].shape[2]])
	for n in range(Nbands):
	    for t in range(C[stage].shape[2]):
	        G  = nx.Graph(Madj[stage][:,:,n,t])
	        largest_k[stage][:,n,t] =  list(nx.core_number(G).values())
np.save('coreness.npy', largest_k)

#--------------------------------------------------------------------------
# Modularity
#--------------------------------------------------------------------------
modularity = {}
# Plot each with size propotional to the largest k-core in which it participates
for stage in stages:
	modularity[stage] = np.zeros([Nbands, C[stage].shape[2]])
	for n in range(Nbands):
	    for t in range(C[stage].shape[2]):
	        G  = nx.Graph(Madj[stage][:,:,n,t])
	        modularity[stage][n,t] =  np.shape(nx.algorithms.community.greedy_modularity_communities(G))[0] 
m_mean = []
m_std  = []
for stage in stages:
	m_mean.append(modularity[stage][0,:].mean())
	m_std.append(modularity[stage][0,:].std() / float(modularity[stage][0,:].shape[0])  )
#--------------------------------------------------------------------------
# Betweeness centrality
#--------------------------------------------------------------------------
betweeness = {}
# Plot each with size propotional to the largest k-core in which it participates
for stage in stages:
	betweeness[stage] = np.zeros([nC, Nbands, C[stage].shape[2]])
	for n in range(Nbands):
	    for t in range(C[stage].shape[2]):
	        G  = nx.Graph(Madj[stage][:,:,n,t])
	        betweeness[stage][:,n,t] =  list(nx.betweenness_centrality(G).values())

#--------------------------------------------------------------------------
# Plotting data for links
#--------------------------------------------------------------------------
# Average active time
plt.figure(figsize=(12,10))
for n in range(Nbands):
    plt.subplot(2,2,n+1)
    for s in stages:
        data = np.load(s+'_act_time.npy')[n,:,:,0].reshape(nP*nT)
        cont, x = np.histogram(data, np.linspace(0,1,10))
        cont = cont / float(cont.sum() * np.diff(x)[-1])
        plt.loglog(x[1:], cont[:], 'o-',label=s)
        #sns.distplot(data[data>0], kde=False, bins=np.linspace(0,40,30), norm_hist=True)
        plt.ylabel('#')
        plt.xlabel('Average "on" time [s]')
        plt.title('Mean "on" time for ' + fbands[n])
plt.legend(stages)
plt.tight_layout()
plt.savefig('img/avg_active_time.pdf', dpi=600)
plt.close()
# Coefficient of variation
plt.figure(figsize=(12,10))
for n in range(Nbands):
    plt.subplot(2,2,n+1)
    for s in stages:
        data = np.load(s+'_act_time.npy')[n,:,:,2].reshape(nP*nT)
        cont, x = np.histogram(data, np.linspace(0,1,10))
        cont    = cont / float(cont.sum() * np.diff(x)[-1])
        plt.loglog(x[1:], cont[0:], 'o-', label=s)
        #sns.distplot(data[data>0], kde=False, bins=np.linspace(0,2,30), norm_hist=True)
        plt.ylabel('#')
        plt.xlabel('CV')
        plt.title('Coefficient of variation for ' + fbands[n])
plt.legend(stages)
plt.tight_layout()
plt.savefig('img/coefficiente_variation.pdf', dpi=600)
plt.close()
# Active time fraction
plt.figure(figsize=(12,10))
for n in range(Nbands):
    plt.subplot(2,2,n+1)
    for s in stages:
        data = np.load(s+'_act_time.npy')[n,:,:,3].reshape(nP*nT)
        cont, x = np.histogram(data, np.linspace(0,1,10))
        cont    = cont / float(cont.sum() * np.diff(x)[-1])
        plt.loglog(x[1:], cont[0:], 'o-', label=s)
        #sns.distplot(data[data>0], kde=False, bins=np.linspace(0,1,20), norm_hist=True)
        plt.ylabel('#')
        plt.xlabel('Total active time fraction')
        plt.title('Fraction of total active time for ' + fbands[n])
plt.legend(stages)
plt.tight_layout()
plt.savefig('img/fraction_on_time.pdf', dpi=600)
plt.close()
# CV  and average on time histograms
plt.figure(figsize=(20,20))
count = 1
for s in stages:
	for n in range(Nbands):
		plt.subplot(len(stages),Nbands,count)
		on_time = np.load(s + '_act_time.npy')[n,:,:,3].reshape(nP*nT)
		cv      = np.load(s + '_act_time.npy')[n,:,:,2].reshape(nP*nT)
		H,x,y=np.histogram2d(on_time, cv, 20, normed=True)
		plt.imshow(np.log(H+1), aspect='auto', cmap='jet', interpolation='gaussian',extent=[x[0], x[1], y[1], y[0]])
		plt.colorbar()
		plt.title(fbands[n])
		if n == 0:
			plt.ylabel(s)
		count += 1
plt.tight_layout()
plt.savefig('img/2d_his.png')
plt.close()

#--------------------------------------------------------------------------
# Plotting data for nodes
#--------------------------------------------------------------------------
# Average active time
plt.figure(figsize=(12,10))
for n in range(Nbands):
    plt.subplot(2,2,n+1)
    for s in stages:
        data = np.load(s+'_act_time_node.npy')[n,:,:,0].reshape(nC*nT)
        cont, x = np.histogram(data, np.linspace(0,1,10))
        cont = cont / float(cont.sum() * np.diff(x)[-1])
        plt.plot(x[1:], cont[:], 'o-',label=s)
        #sns.distplot(data[data>0], kde=False, bins=np.linspace(0,40,30), norm_hist=True)
        plt.ylabel('#')
        plt.xlabel('Average "on" time [s]')
        plt.title('Mean "on" time for ' + fbands[n])
plt.legend(stages)
plt.tight_layout()
plt.savefig('img/avg_active_time_node.pdf', dpi = 600)
plt.close()
# Coefficient of variation
plt.figure(figsize=(12,10))
for n in range(Nbands):
    plt.subplot(2,2,n+1)
    for s in stages:
        data = np.load(s+'_act_time_node.npy')[n,:,:,2].reshape(nC*nT)
        cont, x = np.histogram(data, np.linspace(0,1,10))
        cont    = cont / float(cont.sum() * np.diff(x)[-1])
        plt.loglog(x[1:], cont[0:], 'o-', label=s)
        #sns.distplot(data[data>0], kde=False, bins=np.linspace(0,2,30), norm_hist=True)
        plt.ylabel('#')
        plt.xlabel('CV')
        plt.title('Coefficient of variation for ' + fbands[n])
plt.legend(stages)
plt.tight_layout()
plt.savefig('img/coefficiente_variation_node.pdf', dpi=600)
plt.close()
# Active time fraction
plt.figure(figsize=(12,10))
for n in range(Nbands):
    plt.subplot(2,2,n+1)
    for s in stages:
        data = np.load(s+'_act_time_node.npy')[n,:,:,3].reshape(nC*nT)
        cont, x = np.histogram(data, np.linspace(0,1,10))
        cont    = cont / float(cont.sum() * np.diff(x)[-1])
        plt.loglog(x[1:], cont[0:], 'o-', label=s)
        #sns.distplot(data[data>0], kde=False, bins=np.linspace(0,1,20), norm_hist=True)
        plt.ylabel('#')
        plt.xlabel('Total active time fraction')
        plt.title('Fraction of total active time for ' + fbands[n])
plt.legend(stages)
plt.tight_layout()
plt.savefig('img/fraction_on_time_node.pdf', dpi=600)
plt.close()
# CV  and average on time histograms
plt.figure(figsize=(20,20))
count = 1
for s in stages:
	for n in range(Nbands):
		plt.subplot(len(stages),Nbands,count)
		on_time = np.load(s + '_act_time_node.npy')[n,:,:,3].reshape(nC*nT)
		cv      = np.load(s + '_act_time_node.npy')[n,:,:,2].reshape(nC*nT)
		H,x,y=np.histogram2d(on_time, cv, 20, normed=True)
		plt.imshow(np.log(H+1), aspect='auto', cmap='jet', interpolation='gaussian',extent=[x[0], x[1], y[1], y[0]])
		plt.colorbar()
		plt.title(fbands[n])
		if n == 0:
			plt.ylabel(s)
		count += 1
plt.tight_layout()
plt.savefig('img/2d_his_node.png', dpi=600)
plt.close()

#--------------------------------------------------------------------------
# Brain maps for links
#-------------------------------------------------------------------------
def define_color_links(factor):
	if factor>=0 and factor<0.2:
		col = 'y'
		wid = 0.1
		ms  = 0.5
	elif factor>=0.2 and factor<0.4:
		col = 'g'
		wid = 0.1
		ms  = 0.5
	elif factor>=0.4 and factor<0.6:
		col = 'm'
		wid = 0.6
		ms  = 0.5
	elif factor>=0.6 and factor<0.9:
		col = 'b'
		wid = 1.0
		ms  = 0.5
	elif factor>= 0.9:
		col = 'k'
		wid = 1.0
		ms  = 0.5
	return col, wid, ms

# Average time on map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate((all_stages,act_time[s][n,:,:,0].mean(axis=0)))
	for stage in stages:
		scale      = act_time[stage][n,:,:,0].mean(axis=0)
		max_scale  = all_stages.max()
		min_scale  = all_stages.min()
		factor = ( (scale-min_scale)/(max_scale-min_scale) )
		plt.subplot(Nbands, len(stages),count)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.imshow(ethyl_brainsketch)
		for p in np.argsort(factor):
			ch1 = int(labels[pairs[p, 0]])
			ch2 = int(labels[pairs[p, 1]])
			p1  = [xy[ch1-1,0], xy[ch2-1,0]]
			p2  = [xy[ch1-1,1], xy[ch2-1,1]]
			col, wid, ms = define_color_links(factor[p])
			plt.plot(p1, p2, '.-', ms = ms, lw = wid, c = col)
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/ontime_brain_map.pdf', dpi = 600)
plt.close()

# CV map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate((all_stages,act_time[s][n,:,:,2].mean(axis=0)))
	for stage in stages:
		scale      = act_time[stage][n,:,:,2].mean(axis=0)
		max_scale  = all_stages.max()
		min_scale  = all_stages.min()
		factor = ( (scale-min_scale)/(max_scale-min_scale) )
		plt.subplot(Nbands, len(stages),count)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.imshow(ethyl_brainsketch)
		for p in np.argsort(factor):
			ch1 = int(labels[pairs[p, 0]])
			ch2 = int(labels[pairs[p, 1]])
			p1  = [xy[ch1-1,0], xy[ch2-1,0]]
			p2  = [xy[ch1-1,1], xy[ch2-1,1]]
			col, wid, ms = define_color_links(factor[p])
			plt.plot(p1, p2, '.-', ms = ms, lw = wid, c = col)
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/cv_brain_map.pdf', dpi = 600)
plt.close()

# Fraction on time
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate( (all_stages, act_time[s][n,:,:,3].mean(axis=0)) )
	for stage in stages:
		scale     = act_time[stage][n,:,:,3].mean(axis=0)
		max_scale = all_stages.max()
		min_scale = all_stages.min()
		factor = ( (scale-min_scale)/(max_scale-min_scale) )
		plt.subplot(Nbands, len(stages),count)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.imshow(ethyl_brainsketch)
		for p in np.argsort(factor):
			ch1 = int(labels[pairs[p, 0]])
			ch2 = int(labels[pairs[p, 1]])
			p1  = [xy[ch1-1,0], xy[ch2-1,0]]
			p2  = [xy[ch1-1,1], xy[ch2-1,1]]
			col, wid, ms = define_color_links(factor[p])
			plt.plot(p1, p2, '.-', ms = ms, lw = wid, c = col)
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/ontimefrac_brain_map.pdf', dpi = 600)
plt.close()

# Average strength over trial map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate( (all_stages, act_time[s][n,:,:,4].mean(axis=0)) )
	for stage in stages:
		scale     = act_time[stage][n,:,:,4].mean(axis=0)
		max_scale = all_stages.max()
		min_scale = all_stages.min()
		factor = ( (scale-min_scale)/(max_scale-min_scale) )
		plt.subplot(Nbands, len(stages),count)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.imshow(ethyl_brainsketch)
		for p in np.argsort(factor):
			ch1 = int(labels[pairs[p, 0]])
			ch2 = int(labels[pairs[p, 1]])
			p1  = [xy[ch1-1,0], xy[ch2-1,0]]
			p2  = [xy[ch1-1,1], xy[ch2-1,1]]
			col, wid, ms = define_color_links(factor[p])
			plt.plot(p1, p2, '.-', ms = ms, lw = wid, c = col)
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.tight_layout()
plt.savefig('img/strength_brain_map.pdf', dpi=600)
plt.close()

# Average strength CV over trial map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate( (all_stages, act_time[s][n,:,:,6].mean(axis=0)) )
	for stage in stages:
		scale     = act_time[stage][n,:,:,6].mean(axis=0)
		max_scale = all_stages.max()
		min_scale = all_stages.min()
		factor = ( (scale-min_scale)/(max_scale-min_scale) )
		plt.subplot(Nbands, len(stages),count)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.imshow(ethyl_brainsketch)
		for p in np.argsort(factor):
			ch1 = int(labels[pairs[p, 0]])
			ch2 = int(labels[pairs[p, 1]])
			p1  = [xy[ch1-1,0], xy[ch2-1,0]]
			p2  = [xy[ch1-1,1], xy[ch2-1,1]]
			col, wid, ms = define_color_links(factor[p])
			plt.plot(p1, p2, '.-', ms = ms, lw = wid, c = col)
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/strengthCV_brain_map.pdf', dpi = 600)
plt.close()

#--------------------------------------------------------------------------
# Brain maps for nodes
#--------------------------------------------------------------------------
# Average time on map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate( (all_stages, act_time_node[s][n,:,:,0].mean(axis=0)) )
	for stage in stages:
		scale = act_time_node[stage][n,:,:,0].mean(axis=0)
		max_scale  =  all_stages.max()
		min_scale  =  all_stages.min()
		factor = 5*( (scale-min_scale)/(max_scale-min_scale) )**3
		plt.subplot(Nbands, len(stages),count)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.imshow(ethyl_brainsketch)
		for c in range(nC):
			ch = labels[c].astype(int)
			plt.plot(xy[ch-1,0], xy[ch-1,1], 'mo', ms = factor[c] )
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/ontime_brain_map_node.pdf', dpi = 600)
plt.close()

# CV map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate( (all_stages, act_time_node[s][n,:,:,2].mean(axis=0)) )
	for stage in stages:
		scale = act_time_node[stage][n,:,:,2].mean(axis=0)
		max_scale  =  all_stages.max()
		min_scale  =  all_stages.min()
		factor = 7*( (scale-min_scale)/(max_scale-min_scale) )**1
		plt.subplot(Nbands, len(stages),count)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.imshow(ethyl_brainsketch)
		for c in range(nC):
			ch = labels[c].astype(int)
			plt.plot(xy[ch-1,0], xy[ch-1,1], 'mo', ms = factor[c] )
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/cv_brain_map_node.pdf', dpi = 600)
plt.close()

# Fration time on map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate( (all_stages, act_time_node[s][n,:,:,3].mean(axis=0)) )
	for stage in stages:
		scale = act_time_node[stage][n,:,:,3].mean(axis=0)
		max_scale  =  all_stages.max()
		min_scale  =  all_stages.min()
		factor = 5*( (scale-min_scale)/(max_scale-min_scale) )**3
		plt.subplot(Nbands, len(stages),count)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.imshow(ethyl_brainsketch)
		for c in range(nC):
			ch = labels[c].astype(int)
			plt.plot(xy[ch-1,0], xy[ch-1,1], 'mo', ms = factor[c])
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/ontimefrac_brain_map_node.pdf', dpi = 600)
plt.close()

# Strength map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate( (all_stages, W_node[s][:,n,:].mean(axis=1)) )
	for stage in stages:
		scale = W_node[stage][:,n,:].mean(axis=1)
		max_scale  =  all_stages.max()
		min_scale  =  all_stages.min()
		factor = 6*( (scale-min_scale)/(max_scale-min_scale) )**2
		plt.subplot(Nbands, len(stages),count)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.imshow(ethyl_brainsketch)
		for c in range(nC):
			ch = labels[c].astype(int)
			plt.plot(xy[ch-1,0], xy[ch-1,1], 'mo', ms = factor[c] )
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/strength_brain_map_node.pdf', dpi = 600)
plt.close()

# CV strength map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate( (all_stages, W_node[s][:,n,:].std(axis=1)/W_node[s][:,n,:].mean(axis=1)) )
	for stage in stages:
		scale = W_node[stage][:,n,:].std(axis=1) / W_node[stage][:,n,:].mean(axis=1)
		max_scale  =  all_stages.max()
		min_scale  =  all_stages.min()
		factor = 5*( (scale-min_scale)/(max_scale-min_scale) )**2
		plt.subplot(Nbands, len(stages),count)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.imshow(ethyl_brainsketch)
		for c in range(nC):
			ch = labels[c].astype(int)
			plt.plot(xy[ch-1,0], xy[ch-1,1], 'mo', ms = factor[c])
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/strengthCV_brain_map_node.pdf', dpi = 600)
plt.close()

# Plot coreness series for each channel
plt.figure(figsize=(10,10))
count = 1
for n in range(Nbands):
	for stage in stages:
		plt.subplot(Nbands, len(stages),count)
		plt.imshow(largest_k[stage][:,n,:], aspect='auto', cmap='jet', vmin=0, vmax=18)
		plt.colorbar()
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
			plt.yticks(np.arange(nC)[::8], labels[::8])
			plt.ylabel('Channel')
		else:
			plt.yticks([])
		if n == 0:
			plt.title(stage)
		if n < 3:
			plt.xticks([])
		if n == 3:
			plt.xlabel('Observations')
		count += 1
		plt.xlim([500,1000])
plt.tight_layout()
plt.savefig('img/coreness.pdf', dpi = 600)
plt.close()

# Plot betweeness series for each channel
plt.figure()
count = 1
for n in range(Nbands):
	for stage in stages:
		plt.subplot(Nbands, len(stages),count)
		plt.imshow(betweeness[stage][:,n,:], aspect='auto', cmap='jet', vmax=0.4)
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
			plt.yticks(np.arange(nC)[::8], labels[::8])
			plt.ylabel('Channel')
		else:
			plt.yticks([])
		if stage == 'early_delay':
			plt.colorbar()
		if n == 0:
			plt.title(stage)
		if n < 3:
			plt.xticks([])
		if n == 3:
			plt.xlabel('Observations')
		count += 1
plt.tight_layout()
plt.savefig('img/betweeness.png', dpi = 600)
plt.close()

# Coreness map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate( (all_stages, largest_k[s][:,n,:].mean(axis = 1)) )
	for stage in stages:
		scale = largest_k[stage][:,n,:].mean(axis = 1)
		max_scale  =  all_stages.max()
		min_scale  =  all_stages.min()
		factor = 8*( (scale-min_scale)/(max_scale-min_scale) )**2
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.subplot(Nbands, len(stages),count)
		plt.imshow(ethyl_brainsketch)
		for c in range(nC):
			ch = labels[c].astype(int)
			plt.plot(xy[ch-1,0], xy[ch-1,1], 'mo', ms = factor[c] )
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/coreness_brain_map.pdf', dpi = 600)
plt.close()

# Coreness CV map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate( (all_stages, largest_k[s][:,n,:].std(axis = 1)/largest_k[s][:,n,:].mean(axis = 1)) )
	for stage in stages:
		scale = largest_k[stage][:,n,:].std(axis = 1) / largest_k[stage][:,n,:].mean(axis = 1)
		max_scale  =  all_stages.max()
		min_scale  =  all_stages.min()
		factor = 8*( (scale-min_scale)/(max_scale-min_scale) )**2
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.subplot(Nbands, len(stages),count)
		plt.imshow(ethyl_brainsketch)
		for c in range(nC):
			ch = labels[c].astype(int)
			plt.plot(xy[ch-1,0], xy[ch-1,1], 'mo', ms = factor[c] )
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/corenessCV_brain_map.pdf', dpi = 600)
plt.close()

#--------------------------------------------------------------------------
# Ploting map with average power in each band for each node
#--------------------------------------------------------------------------
raw_lfp    = np.load(session['dir_out']+'raw_lfp_data.npy') # Loading LFP data
mean_power = {}
std_power  = {}

t0 = 0.0
dt = 1.0 / recording_info['fsample']

# Using wavelet transform
for stage in stages:

	mean_power[stage] = np.zeros([nC, nT, Nbands])
	std_power[stage]  = np.zeros([nC, nT, Nbands])

	idx  = indt[index_stage(stage,f)]
	didx = np.diff(idx)[0]
	use  = np.arange(idx.min(), idx.max()+1, 1)

	data = raw_lfp[:,:,use]

	for trial in range(nT):
		for c in range(nC):
			dat = data[trial, c, :]

			mother = wavelet.Morlet(6)
			s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
			dj = 1 / 12  # Twelve sub-octaves per octaves
			J = 7 / dj  # Seven powers of two with dj sub-octaves

			dat_norm = (dat-dat.mean()) / dat.std()
			wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J, mother)
			iwave = wavelet.icwt(wave, scales, dt, dj, mother) * dat.std()
			power = (np.abs(wave)) ** 2 
			power = power.mean(axis=-1) 
			#power = power / scipy.integrate.simps(freqs, power)
			# Averaging over frequency bands
			idx = (freqs>=6)*(freqs<=8)
			mean_power[stage][c, trial, 0] = power[idx].mean()
			std_power[stage][c, trial, 0]  = power[idx].std()
			idx = (freqs>=10)*(freqs<=14)
			mean_power[stage][c, trial, 1] = power[idx].mean()
			std_power[stage][c, trial, 1]  = power[idx].std()
			idx = (freqs>=16)*(freqs<30)
			mean_power[stage][c, trial, 2] = power[idx].mean()
			std_power[stage][c, trial, 2]  = power[idx].std()
			idx = (freqs>=30)*(freqs<=60)
			mean_power[stage][c, trial, 3] = power[idx].mean()
			std_power[stage][c, trial, 3]  = power[idx].std()

mean_power_avg = {}
std_power_avg  = {}
for stage in stages:
	mean_power_avg[stage] = np.zeros([nC, Nbands])
	std_power_avg[stage]  = np.zeros([nC, Nbands])
	for nf in range(Nbands):
		for c in range(nC):
			mean_power_avg[stage][c,nf] = mean_power[stage][c, :, nf].mean()
			std_power_avg[stage][c,nf]  = std_power[stage][c, :, nf].std()

# Average power map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate( (all_stages, mean_power_avg[s][:,n]) )
	for stage in stages:
		scale = mean_power_avg[stage][:,n]
		max_scale  =  all_stages.max()
		min_scale  =  all_stages.min()
		factor = 8*( (scale-min_scale)/(max_scale-min_scale) )**2
		plt.subplot(Nbands, len(stages),count)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.imshow(ethyl_brainsketch)
		for c in range(nC):
			ch = labels[c].astype(int)
			plt.plot(xy[ch-1,0], xy[ch-1,1], 'mo', ms = factor[c] )
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/avg_power_morlet.pdf', dpi = 600)
plt.close()

# CV power map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate( (all_stages, std_power_avg[s][:,n]/mean_power_avg[s][:,n]) )
	for stage in stages:
		scale = std_power_avg[stage][:,n] / mean_power_avg[stage][:,n]
		max_scale  =  all_stages.max()
		min_scale  =  all_stages.min()
		factor = 8*( (scale-min_scale)/(max_scale-min_scale) )**2
		plt.subplot(Nbands, len(stages),count)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.imshow(ethyl_brainsketch)
		for c in range(nC):
			ch = labels[c].astype(int)
			plt.plot(xy[ch-1,0], xy[ch-1,1], 'mo', ms = factor[c] )
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/avg_powerCV_morlet.pdf', dpi = 600)
plt.close()

#--------------------------------------------------------------------------
# Jaccard index nodes
#--------------------------------------------------------------------------
def jaccard(list_con_t1, list_con_t2):
    '''
        Measure the Jaccard index for the same node, for times t1 and t2, t2 > t1.
        list_con_t1: List of nodes that a given channel is connected to at time t1.
        list_con_t2: List of nodes that a given channel is connected to at time t2.
        output:
        Jaccard index
    '''
    if np.union1d(list_con_t1, list_con_t2).shape[0] > 0:
        return 1.0*np.intersect1d(list_con_t1, list_con_t2).shape[0] / np.union1d(list_con_t1, list_con_t2).shape[0]
    else:
        return 1


jac_matrix = {}
for stage in stages:
	c_idx = np.arange(nC)
	jac_matrix[stage] = np.zeros([nC, Nbands, new_nt[stage]*nT-nT])
	for n in range(Nbands):
		for c in range(nC):
			count = 0
			for t in range(nT):
				t1 = 0 + new_nt[stage]*t
				t2 = 1 + new_nt[stage]*t
				while t2 < new_nt[stage]*(t+1):
					l1 = c_idx[Madj[stage][c,:,n,t1].astype(bool)]
					l2 = c_idx[Madj[stage][c,:,n,t2].astype(bool)]
					jac_matrix[stage][c, n, count] = jaccard(l1, l2)
					t1 +=1
					t2 +=1
					count += 1

# Ploting time series
plt.figure()
count = 1
for n in range(Nbands):
	for stage in stages:
		plt.subplot(Nbands, len(stages),count)
		plt.imshow(jac_matrix[stage][:,n,:], aspect='auto', cmap='jet')
		plt.colorbar()
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
			plt.yticks(np.arange(nC)[::8], labels[::8])
			plt.ylabel('Channel')
		else:
			plt.yticks([])
		if n == 0:
			plt.title(stage)
		if n < 3:
			plt.xticks([])
		if n == 3:
			plt.xlabel('Observations')
		count += 1
plt.tight_layout()
plt.savefig('img/jaccard.pdf', dpi = 600)
plt.close()

# Jaccard brain map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate( (all_stages, jac_matrix[s][:,n,:].mean(axis=1)) )
	for stage in stages:
		scale = jac_matrix[stage][:,n,:].mean(axis=1)
		max_scale  =  all_stages.max()
		min_scale  =  all_stages.min()
		factor = 10*( (scale-min_scale)/(max_scale-min_scale) )**3
		plt.subplot(Nbands, len(stages),count)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.1, hspace=.05)
		plt.imshow(ethyl_brainsketch)
		for c in range(nC):
			ch = labels[c].astype(int)
			plt.plot(xy[ch-1,0], xy[ch-1,1], 'mo', ms = factor[c] )
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/jaccard_map.pdf', dpi=600)
plt.close()

# Jaccard CV brain map
plt.figure()
count = 1
for n in range(Nbands):
	all_stages = []
	for s in stages:
		all_stages = np.concatenate( (all_stages, jac_matrix[s][:,n,:].std(axis=1)/jac_matrix[s][:,n,:].mean(axis=1)) )
	for stage in stages:
		scale = jac_matrix[stage][:,n,:].std(axis=1)/jac_matrix[stage][:,n,:].mean(axis=1)
		max_scale  =  all_stages.max()
		min_scale  =  all_stages.min()
		factor = 10*( (scale-min_scale)/(max_scale-min_scale) )**3
		plt.subplot(Nbands, len(stages),count)
		plt.imshow(ethyl_brainsketch)
		for c in range(nC):
			ch = labels[c].astype(int)
			plt.plot(xy[ch-1,0], xy[ch-1,1], 'mo', ms = factor[c] )
			plt.xticks([])
			plt.yticks([])
		if stage == 'before_cue':
			plt.ylabel(fbands[n])
		if n == 0:
			plt.title(stage)
		count += 1
plt.savefig('img/jaccardCV_map.pdf', dpi=600)
plt.close()