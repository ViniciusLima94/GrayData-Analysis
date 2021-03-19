#####################################################################################################
# Class to read and instantiate a session object
#####################################################################################################
import numpy         as np
import scipy.special
import glob
import h5py
import os
from  .io            import set_paths, read_mat
from  mne            import EpochsArray, create_info
from  frites.dataset import DatasetEphy
from  xarray         import DataArray

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
        if monkey not in ['lucy', 'ethyl']:
            raise ValueError('monkey should be either "lucy" or "ethyl"')
        # Class atributes
        self.monkey  = monkey
        self.date    = date
        self.session = 'session0' + str(session)
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
    
    def print_paths(self, ):
        print('dir: ' + self.__paths.dir)
        print('dir_out: ' + self.__paths.dir_out)

class session(session_info):
    
    def __init__(self, raw_path = 'GrayLab/', monkey = 'lucy', date = '150128', session = 1,
                 slvr_msmod = False, align_to = 'cue', trial_type = 1, 
                 behavioral_response = None, evt_dt = [-0.65, 3.00]):
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
        if monkey not in ['lucy', 'ethyl']:
            raise ValueError('monkey should be either "lucy" or "ethyl"')
            
        if align_to not in ['cue', 'match']:
            raise ValueError('align_to should be either "cue" or "match"')
            
        if behavioral_response not in [0, 1, None]:
            raise ValueError('behavioral_response should be either 0 (correct), 1 (incorrect) or None (both)')
            
        if trial_type not in [1, 2, 3, 4]:
            raise ValueError('trial_type should be either 1 (DRT), 2 (intervealed fixation), 3 (blocked fixation) or 4 (blank trials)')       
        
        if trial_type in [2,3] and behavioral_response is not None:
            raise ValueError('For trial type 2 or 3 behavioral_response should be None')
        
        # Instantiating father class session_info
        super().__init__(raw_path = raw_path, monkey = monkey, date = date, session = session)
        
        # Creating paths to load and save data
        self.__paths    = set_paths(raw_path = raw_path, monkey = monkey, date = date, session = session)
        self.__load_mat = read_mat()
        
        # Storing class atributes
        self.slvr_msmod = slvr_msmod
        self.trial_type = trial_type
        self.evt_dt     = evt_dt
        self.align_to   = align_to
        self.behavioral_response = behavioral_response
        
    def read_from_mat(self, ):
        
        # Get file names
        files = sorted(glob.glob( os.path.join(self.__paths.dir, self.date+'*') ))
        # Cue onset/offset and match onset times
        t_con   = self.trial_info['sample_on']
        t_coff  = self.trial_info['sample_off']
        t_mon   = self.trial_info['match_on']
        
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
        # Selecting trials
        if self.behavioral_response is not None:
            indt_idx = (self.trial_info['trial_type'] == self.trial_type) & (self.trial_info['behavioral_response'] == self.behavioral_response)
            indt = np.arange(self.trial_info['num_trials'], dtype=int)[indt_idx]
        else:
            indt_idx = self.trial_info['trial_type'] == self.trial_type
            indt = np.arange(self.trial_info['num_trials'], dtype=int)[indt_idx]
        
        # Number of trials selected
        n_trials = len(indt)
        # Number of time points
        n_times  = int( self.recording_info['lfp_sampling_rate'] * (self.evt_dt[1]-self.evt_dt[0]) )
        # Number of channels selected
        n_channels = len(indch)
        
        # Tensor to store the LFP data NtrialsxNchannelsxTime
        self.data = np.empty([n_trials, n_channels, n_times]) # LFP data
        # Time array
        self.time = np.arange(self.evt_dt[0], self.evt_dt[1], 1/self.recording_info['lfp_sampling_rate'])
       
        # For each selected trial
        for i,nt in zip(range(len(indt)), indt):
            f        = self.__load_mat.read_HDF5(files[nt])
            lfp_data = np.transpose( f['lfp_data'] )
            # Beggining and ending time index for this t0
            indb     = int(t0[nt] + self.recording_info['lfp_sampling_rate']*self.evt_dt[0])
            inde     = int(t0[nt] + self.recording_info['lfp_sampling_rate']*self.evt_dt[1])
            # Time index array
            ind      = np.arange(indb, inde+1, dtype = int)
            # LFP data, dimension NtrialsxNchannelsxTime
            self.data[i] = lfp_data[indch, indb:inde]
            
        # Stimulus presented for the selected trials 
        stimulus = self.trial_info['sample_image'][indt]
        # Labels of the selected channels
        labels   = self.recording_info['channel_numbers'][indch]
        # Number of possible pairs (undirected network)
        nP       = int( scipy.special.comb(n_channels, 2) )
        # Every pair combination
        i, j     = np.tril_indices(n_channels, k = -1)
        pairs    = np.array([j,i]).T
        # Area names for selected channels 
        area     = self.recording_info['area'][indch]
        area     = np.array(area, dtype='S')
        # Store to dictionary
        self.readinfo = {'nC': n_channels, 'nP':nP, 'nT':n_trials, 'pairs': pairs,
                         'indt': indt, 'fsample': float(self.recording_info['lfp_sampling_rate']),
                         'tarray': self.time, 'channels_labels': labels, 'stim':stimulus,
                         'indch': indch, 'areas': area, 't_cue_on': t_con[indt] ,
                         't_cue_off': t_coff[indt], 't_match_on': t_mon[indt]
                         }

    def convert_to_mne_ephy(self, baseline = None):
        # Create info
        info = create_info(self.readinfo['areas'].astype('str').tolist(), self.readinfo['fsample'])
        # Create epoch
        epoch = EpochsArray(self.data, info, tmin=self.readinfo['tarray'][0], baseline = baseline, verbose=False)
        # Creating dataset
        return DatasetEphy([epoch])

    def convert_to_xarray_ephy(self, ):
        # DataArray conversion
        arr_xr = DataArray(self.data, dims=('epochs', 'channels', 'times'),
        coords=(np.arange(self.readinfo['nT']), self.readinfo['areas'], self.readinfo['tarray']))
        # Create dataset
        return DatasetEphy([arr_xr], roi='channels', times='times')
        
    def read_from_h5(self,):
        file_name = os.path.join(self.__paths.dir, self.monkey + '_' + self.session + '_' + self.date + '.h5')
        try:
            hf = h5py.File(file_name, 'r')
        except (OSError):
            print('File for monkey ' + str(self.monkey) + ', date ' + str(self.date) + ' ' + self.session + ' not created yet')

        group = os.path.join('trial_type_'+str(self.trial_type), 
                             'aligned_to_' + str(self.align_to),
                             'behavioral_response_'+str(self.behavioral_response)) 
        g1 = hf.get(group)
        # Read data
        self.data = g1['data'][:]
        # Read info dict.
        self.readinfo = {}
        for k in g1['info'].keys():
            self.readinfo[k] = np.squeeze( np.array(g1['info/'+k]) )
    
    def save_h5(self,):
        file_name = os.path.join(self.__paths.dir, self.monkey + '_' + self.session + '_' + self.date + '.h5')
        try:
            hf = h5py.File(file_name, 'r+')
        except:
            hf = h5py.File(file_name, 'w')

        # Create group relative to the trial type, the aligment and to the behavioral response
        group = os.path.join('trial_type_'+str(self.trial_type), 
                             'aligned_to_' + str(self.align_to),
                             'behavioral_response_'+str(self.behavioral_response)) 
        #  Try to create group if it exists overwrite the values
        try:
            g1 = hf.create_group(group)
            # Save LFP data
            g1.create_dataset('data', data=self.data)
            # Save information on 'readinfo' dict.
            [g1.create_dataset('info/'+k, data=self.readinfo[k]) for k in self.readinfo.keys()]
            # Save dir and dir_out paths
            g1.create_dataset('path/dir', data=self.__paths.dir)
            g1.create_dataset('path/dir_out', data=self.__paths.dir_out)
            hf.close()
        except (ValueError):
            print('Data group already created for trial_type = ' + str(self.trial_type) + ', align_to = ' + \
                    str(self.align_to) + ', and behavioral_response = ' + str(self.behavioral_response) )
