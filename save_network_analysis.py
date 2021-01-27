##################################################################################
# Perform network measurements on the super tensor (coherence data)
##################################################################################
import sys
import numpy                 as     np
from   GDa.temporal_network              import temporal_network
from   GDa.graphics.plot_raster          import plot_nodes_raster_all_bands 
from   GDa.graphics.plot_adjacency       import plot_adjacency 
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
q_thr = 0.8 # Percentile to define coherence threshold
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
net =  temporal_network(raw_path = 'super_tensors/', monkey='lucy', session=1, date=150128, align_to = 'cue', 
                        trial_type = trial_type, behavioral_response = behavioral_response, 
                        trim_borders=True, wt_0=20, wt_1=30)


##################################################################################
# COMPUTING THRESHOLD FOR EACH BAND
##################################################################################
net.compute_coherence_thresholds(q = q_thr)

# Printing the threshold values 
print(r'Threshold in $\delta$ band = ' + str(net.coh_thr[0]))
print(r'Threshold in $\alpha$ band = ' + str(net.coh_thr[1]))
print(r'Threshold in $\beta$  band = ' + str(net.coh_thr[2]))
print(r'Threshold in $\gamma$ band = ' + str(net.coh_thr[3]))
print(r'Threshold in $\gamma$ band = ' + str(net.coh_thr[4]))

##################################################################################
# CONVERT SUPER TENSOR TO ADJACENCY MATRIX
# Dimensions [Number of channels, Number of channels Number of frequency bands, Number of trials * Time]
##################################################################################
net.convert_to_adjacency()

##################################################################################
# CREATE MASK TO TRACK EACH STAGE OF THE ODRT
##################################################################################
net.create_stages_time_grid()

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
    Q['degree'][str(i)]  = compute_nodes_degree(net.A[:,:,i,:], thr = net.coh_thr[i], mirror=True)

##################################################################################
# NODES' CORENESS
##################################################################################
for i in range(len(net.bands)):
    Q['coreness'][str(i)] = compute_nodes_coreness(net.A[:,:,i,:], thr=net.coh_thr[i])

##################################################################################
# NODES' CORENESS NULL MODEL
##################################################################################
for i in tqdm( range(len(net.bands)) ):
    Qr['coreness'][str(i)] = null_model_statistics(net.A[:,:,i,:], compute_nodes_coreness, 10, 
                                                   thr=net.coh_thr[i], n_rewires=1000, n_jobs=10, seed=i)
for i  in tqdm( range(len(net.bands)) ):
    Qr['coreness'][str(i)] =  Qr['coreness'][str(i)].mean(axis = 0)

##################################################################################
# NODES' CLUSTERING
##################################################################################
for i in range(len(net.bands)):
    Q['clustering'][str(i)] = compute_nodes_clustering(net.A[:,:,i,:], thr=net.coh_thr[i])

##################################################################################
# NODES' CLUSTERING NULL MODEL
##################################################################################
for i in tqdm( range(len(net.bands)) ):
    Qr['clustering'][str(i)] = null_model_statistics(net.A[:,:,i,:], compute_nodes_clustering, 10, 
                                                     thr=net.coh_thr[i], n_rewires=1000, n_jobs=10, seed=i)

for i in tqdm( range(len(net.bands)) ):
    Qr['clustering'][str(i)] =  Qr['clustering'][str(i)].mean(axis = 0)

##################################################################################
# NODES' MODULARITY
##################################################################################
for i in range(len(net.bands)):
    Q['modularity'][str(i)] = compute_network_modularity(net.A[:,:,i,:], thr=net.coh_thr[i])

##################################################################################
# NODES' MODULARITY NULL MODEL
##################################################################################
for i in tqdm( range(len(net.bands)) ):
    Qr['modularity'][str(i)] = null_model_statistics(net.A[:,:,i,:], compute_network_modularity, 10, 
                                                     thr=net.coh_thr[i], n_rewires=1000, n_jobs=10, seed=i)

for i in tqdm( range(len(net.bands)) ):
    Qr['modularity'][str(i)] =  Qr['modularity'][str(i)].mean(axis = 0)

##################################################################################
# SAVING DICTIONARIES
##################################################################################
np.save('network_statistics/tt_'+str(trial_type)+'_br_'+str(behavioral_response)+'.npy', {'original': Q, 'null': Qr})
