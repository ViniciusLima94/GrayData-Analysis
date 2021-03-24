##################################################################################
# Perform network measurements on the super tensor (coherence data)
##################################################################################
import sys
import numpy                             as     np
from   GDa.temporal_network              import temporal_network
from   GDa.graphics.plot_raster          import plot_nodes_raster_all_bands 
from   GDa.graphics.plot_adjacency       import plot_adjacency 
from   GDa.misc.create_grids             import create_stages_time_grid
from   GDa.net.layerwise                 import * 
from   GDa.net.null_models               import * 
from   GDa.net.temporal                  import * 
import scipy.signal 
from   tqdm                              import tqdm 
from   joblib                            import Parallel, delayed 
from   scipy                             import stats

##################################################################################
# GLOBAL PARAMETER 
##################################################################################
q_thr               = 0.8 # Percentile to define coherence threshold
monkey              = 'lucy'
trial_type          = 3
behavioral_response = None

##################################################################################
# INSTATIATING TEMPORAL NETWORK
# Monkey = "lucy" or "ethyl
# trial_type  = 1 (drt), 2 (intervealed fixation), 3 (blocked fixation) or 4 (blank trials)
# align_to = "cue" or "match" 
# behavioral_response = 0 (correct), 1 (incorrect) or None (both)  
# Instantiating a temporal network object specifing trim_borders and wt.
##################################################################################
net =  temporal_network(data_raw_path = 'GrayLab/', tensor_raw_path = 'super_tensors/', monkey='lucy', session=1, date='150128', 
                        trim_borders=True, wt=(20,30), threshold=True, relative=False, q=q_thr)

##################################################################################
# COMPUTING THRESHOLD FOR EACH BAND
##################################################################################
#net.compute_coherence_thresholds(q = q_thr, relative=True)

# Printing the threshold values 
#print(r'Threshold in $\delta$ band = ' + str(net.coh_thr[0]))
#print(r'Threshold in $\alpha$ band = ' + str(net.coh_thr[1]))
#print(r'Threshold in $\beta$  band = ' + str(net.coh_thr[2]))
#print(r'Threshold in $\gamma$ band = ' + str(net.coh_thr[3]))
#print(r'Threshold in $\gamma$ band = ' + str(net.coh_thr[4]))

##################################################################################
# CONVERT SUPER TENSOR TO ADJACENCY MATRIX
# Dimensions [Number of channels, Number of channels Number of frequency bands, Number of trials * Time]
##################################################################################
net.convert_to_adjacency()

##################################################################################
# CREATE MASK TO TRACK EACH STAGE OF THE ODRT
##################################################################################
#net.create_stages_time_grid()
t_mask = create_stages_time_grid(net.session_info['t_cue_on'], net.session_info['t_cue_off'],
         net.session_info['t_match_on'], net.session_info['fsample'],
         net.tarray, net.session_info['nT'])

##################################################################################
# CREATE MASK TO TRACK EACH STIM OF THE ODRT
##################################################################################
#net.create_stim_grid()

##################################################################################
# DICTIONARIES TO STORE THE NETWORK MEASURES
##################################################################################
measures = ['degree', 'clustering', 'coreness', 'modularity']

# Q is used to represent an arbritary network measurement
Q = {}
# To store the measures on the randomized networks
Qr = {}

for m in measures:
    Q[m]  = {}
    Qr[m] = {}
    for i in range(len(net.bands)):
        Q[m][str(i)]  = {}
        Qr[m][str(i)] = {}

##################################################################################
# NODES' DEGREE
##################################################################################
for i in range(len(net.bands)):
    Q['degree'][str(i)]  = compute_nodes_degree(net.A[:,:,i,:], mirror=True)

##################################################################################
# NODES' CORENESS
##################################################################################
for i in range(len(net.bands)):
    Q['coreness'][str(i)] = compute_nodes_coreness(net.A[:,:,i,:], is_weighted=False)

##################################################################################
# NODES' CORENESS NULL MODEL
##################################################################################
for i in tqdm( range(len(net.bands)) ):
    Qr['coreness'][str(i)] = null_model_statistics(net.A[:,:,i,:], compute_nodes_coreness, 10, 
                                                   n_rewires=1000, n_jobs=10, seed=i)
for i  in tqdm( range(len(net.bands)) ):
    Qr['coreness'][str(i)] =  Qr['coreness'][str(i)].mean(axis = 0)

##################################################################################
# NODES' CLUSTERING
##################################################################################
for i in range(len(net.bands)):
    Q['clustering'][str(i)] = compute_nodes_clustering(net.A[:,:,i,:], is_weighted=False)

##################################################################################
# NODES' CLUSTERING NULL MODEL
##################################################################################
for i in tqdm( range(len(net.bands)) ):
    Qr['clustering'][str(i)] = null_model_statistics(net.A[:,:,i,:], compute_nodes_clustering, 10, 
                                                     n_rewires=1000, n_jobs=10, seed=i)

for i in tqdm( range(len(net.bands)) ):
    Qr['clustering'][str(i)] =  Qr['clustering'][str(i)].mean(axis = 0)

##################################################################################
# NODES' MODULARITY
##################################################################################
for i in range(len(net.bands)):
    Q['modularity'][str(i)] = compute_network_modularity(net.A[:,:,i,:], is_weighted=True)

##################################################################################
# NODES' MODULARITY NULL MODEL
##################################################################################
for i in tqdm( range(len(net.bands)) ):
    Qr['modularity'][str(i)] = null_model_statistics(net.A[:,:,i,:], compute_network_modularity, 10, 
                                                     n_rewires=1000, n_jobs=10, seed=i)

for i in tqdm( range(len(net.bands)) ):
    Qr['modularity'][str(i)] =  Qr['modularity'][str(i)].mean(axis = 0)

##################################################################################
# SAVING DICTIONARIES
##################################################################################
np.save('network_statistics/tt_'+str(trial_type)+'_br_'+str(behavioral_response)+'.npy', {'original': Q, 'null': Qr})
