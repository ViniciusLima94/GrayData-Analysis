import numpy                as     np
import xarray               as     xr
import GDa.session          
from   GDa.misc.reshape      import reshape_trials, reshape_observations
from   GDa.misc.create_grids import create_stages_time_grid
from   GDa.net.util          import compute_coherence_thresholds, convert_to_adjacency
from   scipy                 import stats
import os
import h5py

class temporal_network():

    def __init__(self, data_raw_path='GrayLab/', tensor_raw_path='super_tensors', monkey='lucy', session=1, 
                 date='150128', trial_type=None, behavioral_response=None, wt=None, 
                 relative=False, q=None, verbose=False):
        r'''
        Temporal network class, this object will have information about the session analysed and store the coherence
        networks (a.k.a. supertensor).
        > INPUTS:
        - raw_path: Raw path to the coherence super tensor
        - monkey: Monkey name
        - session: session number
        - date: date of the recording session
        - align_to: Wheter data is aligned to cue or match
        - trial_type: the type of trial (DRT/fixation) 
        - behavioral_response: Wheter to get sucessful (1) or unsucessful (0) trials
        - wt: Tuple. Trimming window will remove the wt[0] and wt[1] points in the start and end of each trial
        - relative: if threshold is true wheter to do it in ralative (one thr per link per band) or an common thr for links per band
        - q: Quartile value to use for thresholding
        '''

        #Check for incorrect parameter values
        if monkey not in ['lucy', 'ethyl']:
            raise ValueError('monkey should be either "lucy" or "ethyl"')

        # Asserting variable types
        assert isinstance(wt, (type(None), tuple))
        assert isinstance(q, (type(None), int, float))

        # Load session info
        self.info = GDa.session.session(raw_path=data_raw_path, monkey=monkey, date=date, session=session)
        # Storing recording info
        self.recording_info = self.info.recording_info
        # Storing trial info
        self.trial_info     = self.info.trial_info
        #
        del self.info

        # Setting up mokey and recording info to load and save files
        self.raw_path = tensor_raw_path
        self.monkey   = monkey
        self.date     = date                            
        self.session  = 'session0' + str(session)       
        self.trial_type          = trial_type
        self.behavioral_response = behavioral_response 
        
        # Load super-tensor
        self.__load_h5(wt)

        # Threshold the super tensor
        if isinstance(q, (int,float)):
            if verbose: print('Computing coherence thresholds') 
            self.coh_thr = compute_coherence_thresholds(self.super_tensor.stack(observations=('trials','time')).values, 
                                                        q=q,
                                                        relative=relative, verbose=verbose)
            self.super_tensor.values = (self.super_tensor.stack(observations=('trials','time')) > self.coh_thr).unstack().values

        # The trial selection is done at the end after thresholding because the thresholds are computed commonly for all trial types
        filtered_trials, filtered_trials_idx = self.__filter_trial_indexes(trial_type=trial_type, behavioral_response=behavioral_response)
        # Filtering super-tensor
        self.super_tensor = self.super_tensor.sel(trials = filtered_trials)

    def __load_h5(self, wt):
        # Path to the super tensor in h5 format 
        h5_super_tensor_path = os.path.join(self.raw_path, self.monkey+'_'+str(self.session)+'_'+self.date+'.h5')

        try:
            hf = h5py.File(h5_super_tensor_path, 'r')
        except (OSError):
            raise OSError('File for monkey ' + str(self.monkey) + ', date ' + self.date + ' ' + self.session + '  do not exist.')

        # Reade h5 file containing coherence data
        self.super_tensor = hf['coherence'][:]
        self.tarray       = hf['tarray'][:]
        self.freqs        = hf['freqs'][:]
        self.bands        = hf['bands'][:]

        # Reading metadata
        self.session_info = {}
        self.session_info['nT'] = self.super_tensor.shape[1]
        for k in hf['info'].keys():
            if k == 'nC':
                self.session_info[k] = int( np.squeeze( np.array(hf['info/'+k]) ) )
            else:
                self.session_info[k] = np.squeeze( np.array(hf['info/'+k]) )

        if isinstance(wt, tuple):
            self.tarray       = self.tarray[wt[0]:-wt[1]]
            self.super_tensor = self.super_tensor[:,:,:,wt[0]:-wt[1]]

        # Concatenate trials in the super tensor
        self.super_tensor = self.super_tensor.swapaxes(1,2)
        #self.super_tensor = self.reshape_observations()

        # Convert to xarray
        self.super_tensor = xr.DataArray(self.super_tensor, dims=("links","bands","trials","time"),
                                         coords={"trials": self.trial_info.trial_index.values, 
                                                 "time":   self.tarray} )
        # Metadata (the same as session_info)
        self.super_tensor.attrs = self.session_info

    def convert_to_adjacency(self, ):
        self.A = xr.DataArray( convert_to_adjacency(self.super_tensor.values), 
                dims=("roi_1","roi_2","bands","trials","time"),
                coords={"trials": self.super_tensor.trials.values, 
                        "time":   self.super_tensor.time.values,
                        "roi_1":  self.super_tensor.attrs['channels_labels'],
                        "roi_2":  self.super_tensor.attrs['channels_labels']})

    def create_stage_masks(self, flatten=False):
        filtered_trials, filtered_trials_idx = self.__filter_trial_indexes(trial_type=self.trial_type, behavioral_response=self.behavioral_response)
        self.s_mask = create_stages_time_grid(
                      self.super_tensor.attrs['t_cue_on'][filtered_trials_idx],
                      self.super_tensor.attrs['t_cue_off'][filtered_trials_idx],
                      self.super_tensor.attrs['t_match_on'][filtered_trials_idx], 
                      self.super_tensor.attrs['fsample'],
                      self.tarray, len(filtered_trials_idx), flatten=flatten
                      )
        # Convert each mask to xarray
        if flatten: 
            dims=("observations")
        else:
            dims=("trials","time")

        for k in self.s_mask.keys():
            self.s_mask[k] = xr.DataArray(self.s_mask[k], dims=dims)

    def get_data_from(self,stage=None, pad=False):
        r'''
        Return a copy of the super-tensor only for the data points correspondent for a 
        given experiment stage (baseline, cue, delay, match).
        > INPUT: 
        - stage: Name of the stage from which to get data from.
        - pad: If true will only zero out elements out of the specified task stage.
        > OUTPUTS:
        Copy of the super-tensor for the stage specified.
        '''
        assert stage in ['baseline','cue','delay','match'], "stage should be 'baseline', 'cue', 'delay' or 'match'."
        assert pad   in [False, True], "pad should be either False or True."

        # Check if the binary mask was already created
        if not hasattr(self, 's_mask'):
            self.create_stage_masks(flatten=True)
        # If the variable exists but the dimensios are not flattened create again
        if hasattr(self, 's_mask') and len(self.s_mask[stage].shape)==2:
            self.create_stage_masks(flatten=True)

        if pad:
            return self.super_tensor.stack(observations=("trials","time")) * self.s_mask[stage]
        else:
            return self.super_tensor.stack(observations=("trials","time")).isel(observations=self.s_mask[stage])

    def get_number_of_samples(self, stage=None):
        r'''
        Return the number of samples for a given stage for all the trials.
        > INPUT: 
        - stage: Name of the stage from which to get number of samples from.
        > OUTPUTS:
        Return the number of samples for the stage provided for all trials concatenated.
        '''
        assert stage in ['baseline','cue','delay','match'], "stage should be 'baseline', 'cue', 'delay' or 'match'."

        # Check if the binary mask was already created
        if not hasattr(self, 's_mask'):
            self.create_stage_masks(flatten=True)
        # If the variable exists but the dimensios are not flattened create again
        if hasattr(self, 's_mask') and len(self.s_mask[stage].shape)==2:
            self.create_stage_masks(flatten=True)

        return np.int( self.s_mask[stage].sum() )
    
    def __filter_trial_indexes(self,trial_type=None, behavioral_response=None):
        r'''
        Filter super-tensor by desired trials based on trial_type and behav. response.
        > INPUTS:
        - trial_type: List with the number of the desired trial_types to load
        - behavioral_response: List with the number of the desired behavioral_responses to load
        > OUTPUTS:
        - filtered_trials: The number of the trials correspondent to the selected trial_type and behavioral_response
        - filtered_trials_idx: The index of the trials corresponding to the selected trial_type and behavioral_response
        '''
        # Check for invalid values
        assert _check_values(trial_type,[None, 1.0, 2.0, 3.0]) is True, "Trial type should be either 1, 2, 3 or None."
        assert _check_values(behavioral_response,[None,np.nan, 0.0, 1.0]) is True, "Behavioral response should be either 0, 1, NaN or None."

        if trial_type is None and behavioral_response is None:
            #print('1')
            # Getting the number for ODRT trials
            filtered_trials     = self.trial_info.trial_index.values
            # Getting the index for those trials
            filtered_trials_idx = self.trial_info.index.values
            return filtered_trials, filtered_trials_idx
        else:
            if behavioral_response is None:
                #print('2')
                idx = self.trial_info['trial_type'].isin(trial_type)
            elif trial_type is None:
                #print('3')
                idx = self.trial_info['behavioral_response'].isin(behavioral_response)
            else:
                #print('4')
                idx = self.trial_info['trial_type'].isin(trial_type) & self.trial_info['behavioral_response'].isin(behavioral_response)
            # Getting the number for ODRT trials
            filtered_trials     = self.trial_info[idx].trial_index.values
            # Getting the index for those trials
            filtered_trials_idx = self.trial_info[idx].index.values
            return filtered_trials, filtered_trials_idx

    def reshape_trials(self, ):
        aux = reshape_trials( self.super_tensor, self.session_info['nT'], len(self.tarray) )
        return aux

    def reshape_trials(self, ):
        aux = reshape_trials( self.super_tensor, self.session_info['nT'], len(self.tarray) )
        return aux

    def reshape_observations(self, ):
        aux = reshape_observations( self.super_tensor, self.session_info['nT'], len(self.tarray) )
        return aux

################################################ AUX FUNCTIONS ################################################

def _check_values(values, in_list):
    if type(values) is not list: values=[values]
    is_valid=True
    for val in values:
        if val not in in_list:
            is_valid=False
            break
    return is_valid
