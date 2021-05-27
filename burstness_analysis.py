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
from   GDa.util                 import   smooth

import sys
import os

from   config                   import   *
from   tqdm                     import   tqdm

###############################################################################
# Define methods
###############################################################################
def _compute_stats(q, relative=True):
    r''' 
    Method to compute the statistics of interest for a given threshold value q. The statistics are described bellow:

    Next we compute the three quantities of interest: the mean burst duration ($\mu$), the normalized total active time ($\mu_\rm{tot}$), and (CV). 
    Bellow we briefly describe how each of those measurements in an example scenario. 

    Consider the activation series for the edge $p$ composed by nodes $i$ and $j$ ($p=\{i,j\}$) and trial $T$: $A_{p}^{T}=\{00011100001100111111\}$.

    1. the mean burst duration ($\mu$): 
    In the example above $A_{p}^{T}$ has three activation bursts of sizes $3$, $2$, and $6$, therefore $\mu$ = mean(3,2,6) ~ 3.7 a.u. and standard deviantion $\sigma_{\mu}$ = std(3,2,6) ~ 1.7 a.u.;

    2. the normalized total active time ($\mu_\rm{tot}$): The total active time is given by: $len(A_{p}^{T})^{-1}\sum A_{p}^{T}$. 
    If a specif stage $s$ of the experiment is analysed for all trials we consider the concatenated activations series: 
    $A_{p}(s)$, if $n(s)^T$ is the number of observations in stage $s$ at trial $T$ then: $\mu_\rm{tot} = 
    (\sum_{T}n(s)^T)^{-1}\sum A_{p}(s)$;

    3. Burstness CV: The burstness CV is computed from step one as: CV = $\sigma_{\mu}/\mu$.

    > INPUTS:
    - q: The quqartile to use to threshold the data.
    '''

    net =  temporal_network( **set_net_params([1], [1], relative=relative, q=q) )

    # Burstiness analysis statistics
    bs_stats = np.zeros((net.super_tensor.sizes['links'],net.super_tensor.sizes['bands'],len(stages), 3))
    for j in tqdm( range(len(stages)) ):
        bs_stats[:,:,j,:]=np.apply_along_axis(bst.compute_burstness_stats, -1,
                          net.get_data_from(stage=stages[j], pad=True),
                          samples = net.get_number_of_samples(stage=stages[j]),
                          dt=delta/net.super_tensor.attrs['fsample'])

###############################################################################
# Distribution of the average coherence value per task-stage and band
###############################################################################

idx     = int(sys.argv[-1]) # Index to acess the desired session
nmonkey = 0
nses    = 1
ntype   = 0

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
                 behavioral_response=behavioral_response, relative=relative, q=q, wt=(30,30) )

###############################################################################
# 1. Distribution of the average coherence value per task-stage and band
###############################################################################

# Instantiating a temporal network object without thresholding the data
net =  temporal_network( **set_net_params([1], [1]) )

avg_coh = np.zeros((net.super_tensor.sizes['links'], net.super_tensor.sizes['bands'], len(stages)))
for j in tqdm( range( len(stages) ) ):
    avg_coh[:,:,j] = net.get_data_from(stage=stages[j], pad=False).mean(dim='observations')

###############################################################################
# 2. Effect of threshold variation  
###############################################################################

# Here we compute the mean and interquartile distances of the measures of interest for
# different thereshod values

#  q_list = np.arange(0.2, 1.0, 0.1)
#  cv     = np.zeros([net.super_tensor.shape[0], len(net.bands), 3, len(stages), len(q_list)])

#  for i in tqdm( range(len(q_list)) ):
#      # Instantiating a temporal network object without thresholding the data
#      net =  temporal_network(**set_net_params([1], [1], relative=True, q=q_list[i]) )

#      for j,s in zip(range(len(stages)),stages):
#          cv[...,j,i]  = np.apply_along_axis(bst.compute_burstness_stats, -1,
#                         net.get_data_from(stage=s,pad=True),
#                         samples = net.get_number_of_samples(stage=s),
#                         dt      = delta/net.super_tensor.attrs['fsample'])

###############################################################################
# 3. Compute statistics for three different thresholds
###############################################################################
q_list = np.array([0.3, 0.5, 0.8, 0.9]) # Overwriting q_list

bs_stats = [] 

for q in tqdm( q_list ):
    bs_stats += [_compute_stats(q, relative=True)]

