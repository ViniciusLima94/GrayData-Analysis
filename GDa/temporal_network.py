import numpy                as     np
import xarray               as     xr
import scipy

import GDa.session
from   GDa.misc.reshape      import reshape_trials, reshape_observations
from   GDa.misc.create_grids import create_stages_time_grid
from   GDa.net.util          import compute_coherence_thresholds, convert_to_adjacency
from   GDa.net.null_models   import randomize_edges

from   scipy                 import stats
from   tqdm                  import tqdm
from   frites.utils          import parallel_func
import os
import h5py

_DEFAULT_TYPE = np.float32
_COORDS_PATH  = 'storage1/projects/GrayData-Analysis/Brain Areas/lucy_brainsketch_xy.mat'
#  _COORDS_PATH = 'GrayData-Analysis/Brain Areas/lucy_brainsketch_xy.mat'

class temporal_network():

    def __init__(self, data_raw_path='GrayLab/', tensor_raw_path='super_tensors', monkey='lucy', session=1, 
                 date='150128', trial_type=None, behavioral_response=None, wt=None, 
                 drop_trials_after=True, relative=False, keep_weights=False, q=None, verbose=False):
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
        - drop_trials_after: If True the trials that are not in trial_type are droped only after 
            computing the thresholds which mean that the thresholds will be commom for all types of trials
            specified in trial_types.
        - relative: if threshold is true wheter to do it in ralative (one thr per link per band) or 
            an common thr for links per band.
        - keep_weights: If True, will keep the top weights after thresholding.
        - q: Quartile value to use for thresholding.
        - verbose: Wheter to print information or not.
        '''

        # Check for incorrect parameter values
        assert monkey in ['lucy', 'ethyl'], 'monkey should be either "lucy" or "ethyl"'
        # Input conversion
        if trial_type is not None: trial_type = np.asarray(trial_type)
        if behavioral_response is not None: behavioral_response = np.asarray(behavioral_response)

        # Load session info
        self.info           = GDa.session.session(raw_path=data_raw_path, monkey=monkey, date=date, session=session)
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
        self.session  = f'session0{session}'
        self.trial_type          = trial_type
        self.behavioral_response = behavioral_response 

        # Load super-tensor
        self.__load_h5(wt)

        # Threshold the super tensor
        if isinstance(q, (int,float)):
            # Drop trials before thresholding
            if (trial_type is not None or behavioral_response is not None) and (drop_trials_after is False):
                #  print(f'drop_trials_after={drop_trials_after}')
                self.__filter_trials(trial_type, behavioral_response)
            # Threshold
            self.__compute_coherence_thresholds(q, relative, keep_weights, verbose)
            # Drop trials after threshold
            if (trial_type is not None or behavioral_response is not None) and (drop_trials_after is True):
                #  print(f'drop_trials_after={drop_trials_after}')
                self.__filter_trials(trial_type, behavioral_response)
        else:
            # Othrwise simply drop the trials if needed
            if trial_type is not None or behavioral_response is not None:
                self.__filter_trials(trial_type, behavioral_response)

    def __load_h5(self, wt):
        # Path to the file
        h5_super_tensor_path = os.path.join(self.raw_path, 
                                            self.monkey, 
                                            self.date, 
                                            self.session,
                                            'super_tensor.nc')
        #  try:
        #      hf = h5py.File(h5_super_tensor_path, 'r')
        #  except (OSError):
        #      raise OSError('File "super_tensor.h5" not found for monkey')

        #  # Reade h5 file containing coherence data
        #  self.super_tensor = hf['coherence'][:]
        #  self.tarray       = hf['tarray'][:]
        #  self.freqs        = hf['freqs'][:]
        #  self.bands        = hf['bands'][:]
        #  self.roi          = hf['roi'][:]

        #  # Reading metadata
        #  self.session_info = {}
        #  self.session_info['nT'] = self.super_tensor.shape[1]
        #  for k in hf['info'].keys():
        #      if k == 'nC':
        #          self.session_info[k] = int( np.squeeze( np.array(hf['info/'+k]) ) )
        #      else:
        #          self.session_info[k] = np.squeeze( np.array(hf['info/'+k]) )

        #  if isinstance(wt, tuple):
        #      self.tarray       = self.tarray[wt[0]:-wt[1]]
        #      self.super_tensor = self.super_tensor[...,wt[0]:-wt[1]]

        #  # Concatenate trials in the super tensor
        #  self.super_tensor = self.super_tensor.swapaxes(1,2)

        #  # Convert to xarray
        #  self.super_tensor = xr.DataArray(self.super_tensor.astype(_DEFAULT_TYPE), dims=("links","bands","trials","time"),
        #                                   coords={"trials": self.trial_info.trial_index.values, 
        #                                           "time":   self.tarray} )

        #  # Metadata (the same as session_info)
        #  self.super_tensor.attrs         = self.session_info
        # Getting euclidian distance between each pair of nodes

        # Try to read the file in the path specified
        try:
            self.super_tensor = xr.load_dataarray(h5_super_tensor_path)
        except:
            raise OSError('File "super_tensor.h5" not found for monkey')

        # Copy axes values to class attributes
        self.time  = self.super_tensor.times.values 
        self.freqs = self.super_tensor.freqs.values 

        # Copying metadata as class attributes
        self.session_info = {}
        self.session_info['nT'] = self.super_tensor.sizes["trials"]
        for key in self.super_tensor.attrs.keys():
            self.session_info[key] = self.super_tensor.attrs[key]

        # Crop beginning/ending of super-tensor due to edge effects
        if isinstance(wt, tuple):
            self.time         = self.time[wt[0]:-wt[1]]
            self.super_tensor = self.super_tensor[...,wt[0]:-wt[1]]

        # Get euclidean distances
        #  self.super_tensor.attrs['d_eu'] = self.__get_euclidean_distances()

    def convert_to_adjacency(self, n_jobs=1, verbose=False):
        from frites.conn import conn_reshape_undirected
        
        # Get number of bands
        n_bands = len(self.freqs)
        
        def _for_band(band):
            A = []
            for i in range( self.A.sizes['trials'] ):
                A += [conn_reshape_undirected(self.super_tensor.isel(bands=band,trials=i), fill_diagonal=1)]
            A = xr.concat(A, dim="trials")
            return A

        # define the function to compute in parallel
        parallel, p_fun = parallel_func(
            _single_estimative, n_jobs=n_jobs, verbose=verbose,
            total=n_bands)

        # Converting to adjacency matrix for each band
        self.A = parallel(p_fun(band) for i in range(n_bands))
        self.A = xr.concat(self.A, dim="bands")
        # Reordering dimensions
        self.A = self.A.transpose(("sources","targets","bands","trials","times"))
        
        #  self.A = xr.DataArray( convert_to_adjacency(self.super_tensor.values), 
        #          dims=("roi_1","roi_2","bands","trials","time"),
        #          coords={"trials": self.super_tensor.trials.values, 
        #                  "time":   self.super_tensor.time.values,
        #                  "roi_1":  self.super_tensor.attrs['channels_labels'],
        #                  "roi_2":  self.super_tensor.attrs['channels_labels']})

    def create_null_ensemble(self, n_stat, n_rewires,seed, n_jobs=1, verbose=False):
        # Check if the adjacency matrix was already created
        if not hasattr(self, 'A'):
            self.convert_to_adjacency()

        # Compute null-model for one single estimative 
        def _single_estimative(band, seed):
            return randomize_edges(self.A.isel(bands=band), n_rewires, seed, verbose)

        # define the function to compute in parallel
        parallel, p_fun = parallel_func(
            _single_estimative, n_jobs=n_jobs, verbose=verbose,
            total=n_stat)
        
        self.A_null = []
        itr = range(len(self.freqs))
        for band in (tqdm(itr) if verbose else itr):
            # compute the single trial coherence
            A_tmp   = parallel(p_fun(band,i*(seed+100)) for i in range(n_stat))
            self.A_null += [xr.concat(A_tmp,dim="surrogates")]
            del A_tmp
        
        # Concatenate bands
        self.A_null = xr.concat(self.A_null,dim="bands")
        self.A_null = self.A_null.transpose("surrogates","sources","targets","freqs","trials","times")

    def create_stage_masks(self, flatten=False):
        filtered_trials, filtered_trials_idx = self.__filter_trial_indexes(trial_type=self.trial_type, behavioral_response=self.behavioral_response)
        self.s_mask = create_stages_time_grid(
                      self.super_tensor.attrs['t_cue_on'],
                      self.super_tensor.attrs['t_cue_off'],
                      self.super_tensor.attrs['t_match_on'], 
                      self.super_tensor.attrs['fsample'],
                      self.time, self.super_tensor.sizes['trials'], flatten=flatten
                      )
        if flatten: 
            dims=("observations")
        else:
            dims=("trials","times")

        for key in self.s_mask.keys():
            self.s_mask[key] = xr.DataArray(self.s_mask[key], dims=dims)

    def __filter_trials(self, trial_type, behavioral_response):
        filtered_trials, filtered_trials_idx = self.__filter_trial_indexes(trial_type=trial_type, behavioral_response=behavioral_response)
        self.super_tensor = self.super_tensor.sel(trials=filtered_trials)
        # Filtering attributes
        for key in ['stim', 't_cue_off', 't_cue_on', 't_match_on']:
            self.super_tensor.attrs[key] = self.super_tensor.attrs[key][filtered_trials_idx]

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

        if isinstance(trial_type, np.ndarray) and behavioral_response is None:
            idx = self.trial_info['trial_type'].isin(trial_type)
        if trial_type is None and isinstance(behavioral_response, np.ndarray):
            idx = self.trial_info['behavioral_response'].isin(behavioral_response)
        if isinstance(trial_type, np.ndarray) and isinstance(behavioral_response, np.ndarray):
            idx = self.trial_info['trial_type'].isin(trial_type) & self.trial_info['behavioral_response'].isin(behavioral_response)
        filtered_trials     = self.trial_info[idx].trial_index.values
        filtered_trials_idx = self.trial_info[idx].index.values
        return filtered_trials, filtered_trials_idx

    def get_data_from(self, stage=None, pad=False):
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
            return self.super_tensor.stack(observations=("trials","times")) * self.s_mask[stage]
        else:
            return self.super_tensor.stack(observations=("trials","times")).isel(observations=self.s_mask[stage])

    def get_number_of_samples(self, stage=None, total=False):
        r'''
        Return the number of samples for a given stage for all the trials.
        > INPUT: 
        - stage: Name of the stage from which to get number of samples from.
        - total: If True will get the total number of observations for all trials
                 otherwise will get the number of observations per trial.
        > OUTPUTS:
        Return the number of samples for the stage provided for all trials concatenated.
        '''
        assert stage in ['baseline','cue','delay','match'], "stage should be 'baseline', 'cue', 'delay' or 'match'."

        # Check if the binary mask was already created
        if not hasattr(self, 's_mask'):
            self.create_stage_masks(flatten=False)
        # If the variable exists but the dimensios are not flattened create again
        if hasattr(self, 's_mask') and len(self.s_mask[stage].shape)==1:
            self.create_stage_masks(flatten=False)
        if total:
            return np.int( self.s_mask[stage].sum() )
        else:
            return self.s_mask[stage].sum(dim='times')

    def get_averaged_st(self, win_delay=None):
        r'''
        Get the trial averaged super-tensor, it averages togheter the trials for delays in
        the ranges specified by win_delay.
        > INPUTS:
        - win_delay: The delay durations that should be averaged together, e.g., 
               if win_delay = [[800, 1000],[1000,1200]] all the trials
               in which the delays are between 800-1000ms will be averaged together, 
               likewise for 1000-1200ms, therefore two averaged super-tensors will be 
               returnd. If None the average is done for all trials.
        > OUTPUTS:
        - The trial averaged super-tensor.
        '''
        assert isinstance(win_delay, (type(None), list))

        # Delay duration for each trial
        delay = (self.super_tensor.attrs['t_match_on']-self.super_tensor.attrs['t_cue_off'])/self.super_tensor.attrs['fsample']
        avg_super_tensor = [] # Averaged super-tensor
        n_obs            = [] # Number of observations for each window
        # If no window is provided average over all trials otherwise average
        # for each delay duration window.
        if win_delay is None:
            return self.super_tensor.mean(dim='trials')
        else:
            for i, wd in enumerate( win_delay ):
                # Get index for delays within the window
                idx = (delay>=wd[0])*(delay<wd[-1])
                avg_super_tensor += [self.super_tensor.isel(trials=idx).mean(dim='trials')]
            return avg_super_tensor

    def __get_coords(self,):
        r'''
        Get the channels coordinates.
        '''
        from pathlib import Path
        _path = os.path.join(Path.home(), _COORDS_PATH)
        xy    = scipy.io.loadmat(_path)['xy']
        return xy

    def __get_euclidean_distances(self, ):
        r'''
        Get the channels euclidean distances based on their coordinates.
        '''
        xy   = self.__get_coords()
        d_eu = np.zeros(self.session_info['pairs'].shape[0])
        for i in range( self.session_info['pairs'].shape[0] ):
            c1 = self.session_info['channels_labels'].astype(int)[self.session_info['pairs'][i,0]]
            c2 = self.session_info['channels_labels'].astype(int)[self.session_info['pairs'][i,1]]
            dx = xy[c1-1,0] - xy[c2-1,0]
            dy = xy[c1-1,1] - xy[c2-1,1]
            d_eu[i] = np.sqrt(dx**2 + dy**2)
        return d_eu

    def __compute_coherence_thresholds(self, q, relative, keep_weights, verbose):
        if verbose: print('Computing coherence thresholds') 
        self.coh_thr = compute_coherence_thresholds(self.super_tensor.stack(observations=('trials','times')).values, 
                                                    q=q, relative=relative, verbose=verbose)
        # Temporarily store the stacked super-tensor
        tmp  = self.super_tensor.stack(observations=('trials','times'))
        # Create the mask by applying threshold
        mask        = xr.DataArray(np.empty(tmp.shape), dims=tmp.dims, coords=tmp.coords)
        mask.values = tmp > self.coh_thr.values[...,None]
        #  mask = self.super_tensor.stack(observations=('trials','times')) > self.coh_thr
        # Thrshold setting every element above the threshold as 1 otherwise 0
        if not keep_weights:
            self.super_tensor.values = mask.unstack().values
        # Threshold by leaving the weights for above the threshold and otherwise 0
        else:
            #  tmp = self.super_tensor.stack(observations=('trials','times'))*mask
            tmp *= mask
            self.super_tensor.values = tmp.unstack().values

    def reshape_trials(self, ):
        assert len(self.super_tensor.dims)==3
        self.super_tensor.unstack()

    def reshape_observations(self, ):
        assert len(self.super_tensor.dims)==4
        self.super_tensor.stack(observations=('trials','times'))

################################################ AUX FUNCTIONS ################################################
def _check_values(values, in_list):
    is_valid=True
    if values is None:
        return is_valid
    else:
        for val in values:
            if val not in in_list:
                is_valid=False
                break
        return is_valid
