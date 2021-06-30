#####################################################################################################
# Class to read and instantiate a session object
#####################################################################################################
import numpy         as np
import xarray        as xr
import pandas        as pd
import scipy.special
import glob
import h5py
import os
from  .io            import set_paths, read_mat
from  mne            import EpochsArray, create_info
from  frites.dataset import DatasetEphy
from  xarray         import DataArray
from  tqdm           import tqdm 

class session_info():
    
    def __init__(self, raw_path = 'GrayLab/', monkey = 'lucy', date = '150128', session = 1):
        r'''
        Session info class, it will store recording and trial info of the session specified.
        > INPUTS:
        - raw_path: Raw path to the LFP data and metadata
        - monkey: Monkey name
        - date: date of the recording session
        - session: session number
        '''
        #Check for incorrect parameter values
        assert monkey in ['lucy', 'ethyl'], 'monkey should be either "lucy" or "ethyl"'

        # Class atributes
        self.monkey  = monkey
        self.date    = date
        self.session = f'session0{session}'
        # Creating paths to load and save data
        self.__paths    = set_paths(raw_path = raw_path, monkey = monkey, date = date, session = session)
        # To load .mat files
        self.__load_mat = read_mat()
        # Actually read the info
        self.__read_session_info()
        
    def __read_session_info(self, ):
        # Recording and trial info
        info = ['recording_info.mat', 'trial_info.mat'] 
        ri   = self.__load_mat.read_mat(os.path.join(self.__paths.dir,info[0]) )['recording_info']
        ti   = self.__load_mat.read_HDF5(os.path.join(self.__paths.dir,info[1]) )['trial_info']
        # Storing the recording and trial infor into dictionaries
        self.trial_info     = {}
        self.recording_info = {}
        for key in ri._fieldnames:
            self.recording_info[key] = np.squeeze(ri.__dict__[key])
        for key in ti.keys():
            self.trial_info[key] = np.squeeze(ti[key])
        # Converting trial info to data frame
        self.trial_info     = pd.DataFrame.from_dict(self.trial_info, orient='columns')
    
    def print_paths(self, ):
        print('dir: ' + self.__paths.dir)
        print('dir_out: ' + self.__paths.dir_out)

class session(session_info):
    
    def __init__(self, raw_path = 'GrayLab/', monkey = 'lucy', date = '150128', session = 1,
                 slvr_msmod = False, align_to = 'cue', evt_dt = [-0.65, 3.00]):

        r'''
        Session class, it will store the data with the recording and trial info of the session specified.
        > INPUTS:
        - raw_path: Raw path to the LFP data and metadata
        - monkey: Monkey name
        - date: date of the recording session
        - session: session number
        - slvr_msmod: Whether to load or not channels with slvr_msmod
        - align_to: Wheter data is aligned to cue or match
        - trial_type: the type of trial (DRT/fixation) 
        - behavioral_response: Wheter to get sucessful (1) or unsucessful (0) trials
        - evt_dt: Get signal from evt_dt[0] to evt_dt[1]
        '''
        #Check for incorrect parameter values
        assert monkey in ['lucy', 'ethyl'], 'monkey should be either "lucy" or "ethyl"'
            
        assert align_to in ['cue', 'match'], 'align_to should be either "cue" or "match"'
        
        # Instantiating father class session_info
        super().__init__(raw_path = raw_path, monkey = monkey, date = date, session = session)
        
        # Creating paths to load and save data
        self.__paths    = set_paths(raw_path = raw_path, monkey = monkey, date = date, session = session)
        self.__load_mat = read_mat()
        
        # Storing class atributes
        self.slvr_msmod = slvr_msmod
        self.evt_dt     = evt_dt
        self.align_to   = align_to

        # Selecting trials
        self.trial_info = self.trial_info[ (self.trial_info['trial_type'].isin([1.0,2.0,3.0])) ]
        # Reset index and create new column with the index of select trials
        self.trial_info = self.trial_info.rename_axis('trial_index').reset_index()

    def read_from_mat(self, verbose=False):
        # Get file names
        files = sorted(glob.glob( os.path.join(self.__paths.dir, self.date+'*') ))

        # Cue onset/offset and match onset times
        t_con   = self.trial_info['sample_on'].values
        t_coff  = self.trial_info['sample_off'].values
        t_mon   = self.trial_info['match_on'].values
        
        # Choose if is aligned to cue or to match
        if self.align_to == 'cue':
            t0 = t_con
        elif self.align_to == 'match':
            t0 = t_mon
        
        # Channels index array
        indch   = np.arange(self.recording_info['channel_count'], dtype = int)

        # Exclude channels with short latency visual respose (slvr) and microsacade modulation (ms_mod)
        if self.slvr_msmod == False:
            idx_slvr_msmod = (self.recording_info['slvr'] == 0) & (self.recording_info['ms_mod'] == 0)
            indch          = indch[idx_slvr_msmod]
        
        # Number of trials selected
        n_trials   = len(self.trial_info)
        # Number of time points
        n_times    = int(self.recording_info['lfp_sampling_rate'] * (self.evt_dt[1]-self.evt_dt[0]))
        # Number of channels selected
        n_channels = len(indch)
        
        # Tensor to store the LFP data NtrialsxNchannelsxTime
        self.data = np.empty((n_trials, n_channels, n_times)) # LFP data
        # Time array
        self.time = np.arange(self.evt_dt[0], self.evt_dt[1], 1/self.recording_info['lfp_sampling_rate'])
       
        # For each selected trial
        itr = range(len(self.trial_info))
        for i in (tqdm(itr) if verbose else itr):
            f        = self.__load_mat.read_HDF5(files[self.trial_info.trial_index.values[i]])
            lfp_data = np.transpose( f['lfp_data'] )
            # Beggining and ending time index for this t0
            indb     = int(t0[i] + self.recording_info['lfp_sampling_rate']*self.evt_dt[0])
            inde     = int(t0[i] + self.recording_info['lfp_sampling_rate']*self.evt_dt[1])
            # Time index array
            ind      = np.arange(indb, inde+1, dtype = int)
            # LFP data, dimension NtrialsxNchannelsxTime
            self.data[i] = lfp_data[indch, indb:inde]
            
        # Stimulus presented for the selected trials 
        stimulus = self.trial_info['sample_image'].values
        # Labels of the selected channels
        labels   = self.recording_info['channel_numbers'][indch]
        # Number of possible pairs (undirected network)
        nP       = int( scipy.special.comb(n_channels, 2) )
        # Every pair combination
        i, j     = np.tril_indices(n_channels, k = -1)
        pairs    = np.array([j,i]).T
        # Area names for selected channels 
        area     = self.recording_info['area'][indch]
        area     = np.array(area, dtype='<U13')

        # Convert the data to an xarray
        self.data = xr.DataArray(self.data, dims = ("trials","roi","time"), 
                                 coords={"trials": self.trial_info.trial_index.values, 
                                         "roi":    area,
                                         "time":   self.time} )
        # Saving metadata
        self.data.attrs = {'nC': n_channels, 'nP':nP, 
                           'fsample': float(self.recording_info['lfp_sampling_rate']),
                           'channels_labels': labels.astype(np.int64), 'stim':stimulus,
                           'indch': indch, 't_cue_on': t_con,
                           't_cue_off': t_coff, 't_match_on': t_mon}

    def convert_to_xarray_ephy(self, ):
        # Create dataset
        return DatasetEphy([self.data], roi='roi', times='time')
