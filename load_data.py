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

date = [
141014,  141024,  141125,  141203,  141212,  150114,  150210,  150219,  150302,  150407,  150416,  150504,  150529,
141015,  141029,  141126,  141205,  141215,  150126,  150211,  150223,  150303,  150408,  150427,  150511,  150608,
141016,  141103,  141127,  141208,  141216,  150128,  150212,  150224,  150304,  150413,  150428,  150512,  
141017,  141112,  141128,  141209,  141217,  150129,  150213,  150226,  150305,  150414,  150429,  150527,
141023,  141113,  141202,  141211,  141218,  150205,  150217,  150227,  150403,  150415,  150430,  150528,
]

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
