r'''
Characterizing bursting dynamics of links
In order to charaterize the bursting dynamics of the links in the temporal network the methods in GDa.stats.bursting module can be used.

In particular, we do so for binary arrays of activations that, similar to spike-trains, are binary with the 
value one used to represent the activation of a certain link. For each activation sequence we characterize 
the following statistics: link avg. activation time (mu), total act. time relative to task stage time (mu_tot), 
CV (mean activation time over its std).

This procedure requires that we threshold the values of coherence in the temporal network. The thresholds are 
defined for each link (relative) or commonly for all links (absolute) based on a arbritary quartile of its 
coherence distribution q. Since this thresholding procedure can affect the results we also study how the threshold 
level influnces the statistics measured. It is also important to note that since we may compare different 
trial types (ODRT, int. fixation, blocked fixation) the threshold is computed commonly for all the trials.
'''
import numpy                    as       np
import xarray                   as       xr
import matplotlib.pyplot        as       plt
import seaborn                  as       sns
import scipy.signal

import GDa.stats.bursting       as       bst
from   GDa.temporal_network     import   temporal_network

import sys
import os

from   config                   import   *
from   GDa.util                 import   smooth

###############################################################################
# Distribution of the average coherence value per task-stage and band
###############################################################################

idx     = int(sys.argv[-1]) # Index to acess the desired session
nmonkey = 0
nses    = 1
ntype   = 0

###############################################################################
# Defining parameters to instantiate temporal network
###############################################################################

# Path to coherence data 
path_st = os.path.join('Results', str(dirs['monkey'][nmonkey]), str(dirs['date'][nmonkey][idx]), f'session0{nses}')

def set_net_params(path_st, trial_type=None, behavioral_response=None, relative=None, q=None):
    r'''
    Return dict. with parameters to instantiate temporal network object.
    '''

    # Default parameters plus params passed to method trial type 1
    return dict( data_raw_path='GrayLab/', tensor_raw_path=path, monkey=dirs['monkey'][nmonkey],
                 session=1, date=dirs['date'][nmonkey][idx], trial_type=trial_type,
                 behavioral_response=behavioral_response, relative=relative, q=q, wt=(30,30) )

###############################################################################
# Distribution of the average coherence value per task-stage and band
###############################################################################

# Instantiating a temporal network object without thresholding the data
net =  temporal_network( **set_net_params(path_st, [1], [1]) )

avg_coh = np.zeros((net.super_tensor.sizes['links'], net.super_tensor.sizes['bands'], len(stages)))
for j in tqdm( range( len(stages) ) ):
    avg_coh[:,:,j] = net.get_data_from(stage=stages[j], pad=False).mean(dim='observations')

plt.figure(figsize=(12,8))
for j in tqdm( range( len(stages) ) ):
    plt.subplot(2,2,j+1)
    sns.violinplot(data=avg_coh[:,:,j], color='orange')
    plt.title(stages[j], fontsize=15)
    if j%2==0: plt.ylabel('Coherence', fontsize=15)
    if j==2 or j==3: plt.xticks(range(len(band_names)),band_names, fontsize=15)
    else: plt.xticks([])
plt.tight_layout()

