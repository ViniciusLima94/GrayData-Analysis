import GDa.stats.bursting                as     bst
from   GDa.session                       import session
from   GDa.temporal_network              import temporal_network
from   GDa.util                          import smooth

import seaborn                           as       sns
import numpy                             as       np
import xarray                            as       xr
import matplotlib.pyplot                 as       plt
import scipy.signal
import time
import os
import h5py

from   tqdm                              import tqdm
from   sklearn.manifold                  import TSNE
from   config                            import *
from   scipy     

# Bands names
band_names  = [r'band 1', r'band 2', r'band 3', r'band 4', r'band 5']
stages      = ['baseline', 'cue', 'delay', 'match']

_COH_FILE = f'super_tensor_k{500}.nc'

##################################################################################
# Load coherence file
##################################################################################
# Default threshold
kw = dict(q=None, keep_weights=False, relative=False)

# Instantiating a temporal network object without thresholding the data
net =  temporal_network(coh_file=_COH_FILE, monkey=dirs['monkey'][nmonkey],
                        session=1, date='150128', trial_type=[1],
                        behavioral_response=[1], wt=(20,20), drop_trials_after=True,
                        verbose=True, **kw)


