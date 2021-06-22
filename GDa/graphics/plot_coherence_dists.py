import numpy             as np 
import matplotlib.pyplot as plt

def plot_pooled_coherence_dists(super_tensor, bins = 100, normed = False, thrs=None, titles = None, figsize=(10,10)):
    plt.figure(figsize=figsize)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        n,x = np.histogram(super_tensor[:,i,:].flatten(), bins=bins)
        if normed == True:
            delta = 1e5 / n.max()
            n = n / n.sum()
        else:
            delta = 1e5
        if i == 0 or i == 2:
            plt.ylabel('#Count')
        if i == 2 or i == 3:
            plt.xlabel('Coherence')
        plt.title(titles[i])
        plt.plot(x[1:], n)
        plt.xlim([0,1])
        plt.ylim([0, n.max() + delta])
        if type(thrs) != type(None):
            plt.vlines(thrs[i], 0, n.max() + delta, linestyle = '--', color = 'r')
            if i == 0:
                plt.legend(['Distribution', 'Threshold'])

def plot_pooled_coherence_dists_per_stage(super_tensor, time_masks, bins = 100, normed = False,  titles = None, figsize=(10,10)):
    plt.figure(figsize=figsize)
    for i in range(4):
        plt.subplot(2,2,i+1)
        nb, xb = np.histogram(super_tensor[:,i,time_masks[0]].flatten(), bins=bins)
        nc, xc = np.histogram(super_tensor[:,i,time_masks[1]].flatten(), bins=bins)
        nd, xd = np.histogram(super_tensor[:,i,time_masks[2]].flatten(), bins=bins)
        nm, xm = np.histogram(super_tensor[:,i,time_masks[3]].flatten(), bins=bins)
        # Normalizing histograms
        nb = nb / nb.sum()
        nc = nc / nc.sum()
        nd = nd / nd.sum()
        nm = nm / nm.sum()
        plt.semilogy(xb[1:], nb)
        plt.semilogy(xc[1:], nc)
        plt.semilogy(xd[1:], nd)
        plt.semilogy(xm[1:], nm)
        plt.title(titles[i])
        if i == 0:
            plt.legend(['Baseline', 'Cue', 'Delay', 'Match'])
        if i == 2 or i == 3:
            plt.xlabel('Coherence')
        if i == 0 or i == 2:
            plt.ylabel('#Counts')

def plot_pooled_coherence_dists_per_stim(super_tensor, stim_masks, bins = 100, normed = False, thrs=None, titles = None, figsize=(10,10)):
    plt.figure(figsize=figsize)
    for j in range(4):
        plt.subplot(2,2,j+1)
        for i in range(stim_masks.shape[0]):
            #  Computing histograms
            nb, xb = np.histogram(super_tensor[:,j,stim_masks[i].astype(bool)].flatten(), bins=bins)
            #  Normalizing histograms
            nb     = nb / nb.sum()
            plt.semilogy(xb[1:], nb)
        if j == 0:
            plt.legend(['Stim 1', 'Stim 2', 'Stim 3', 'Stim 4', 'Stim 5'])
        if j == 2 or j == 3:
            plt.xlabel('Coherence')
        if j == 0 or j == 2:
            plt.ylabel('#Counts')







