#####################################################################################################
# Read and save the LFP data and information for each trial in numpy format
#####################################################################################################
from   GDa.LFP         import LFP
from   joblib          import Parallel, delayed
import multiprocessing

nmonkey = 0
nses    = 6
ntype   = 0
#####################################################################################################
# Directories
#####################################################################################################
dirs = {'rawdata':'GrayLab/',
        'results':'Results/',
        'monkey' :['lucy', 'ethyl'],
        'session':'session01',
        'date'   :[['141014', '141015', '141205', '150128', '150211', '150304'], []]
        }

def save_data_npy(nmonkey, ntype, nses):
    # Instatiate LFP object
    lfp_data = LFP(raw_path = dirs['rawdata'], monkey = dirs['monkey'][nmonkey], stype = 'samplecor', date = dirs['date'][nmonkey][nses], 
                   session  = 1, evt_dt = [-0.65,3.00])
    # Read session info
    lfp_data.read_session_info()
    # Read LFP data
    lfp_data.read_lfp_data()
    # Saving npy file
    lfp_data.save_npy()

# Missing trials in 141014 thats why Im starting ses from 1
Parallel(n_jobs=nses)(delayed(save_data_npy)(nmonkey, ntype, ses) for ses in range(1,nses))