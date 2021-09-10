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

# Define default return type
_DEFAULT_TYPE = np.float32
# Defining default paths
_COORDS_PATH  = 'storage1/projects/GrayData-Analysis/Brain Areas/lucy_brainsketch_xy.mat'
_DATA_PATH    = '../GrayLab'
_COH_PATH     = '../Results'

class temporal_network():

    def __init__(self, coh_file=None, monkey='lucy', session=1, coh_thr=None,
                 date='150128', trial_type=None, behavioral_response=None, wt=None,
                 drop_trials_after=True, relative=False, keep_weights=False, q=None, verbose=False):
        """
        Temporal network class, this object will have information about the session analysed and store the coherence
        networks (a.k.a. supertensor).
        Parameters
        ----------
        coh_file: string | None
            Path to the coherence file
        monkey: string | 'lucy'
            Monkey name
        session: int | 1
            session number
        coh_thr: array_like | None
            Thresholds to use (if not passed they will be computed according to the pars relative and q)
        date: string | '150128'
            date of the recording session
        trial_type: int | None
            the type of trial (DRT/fixation)
        behavioral_response: int | None
            Wheter to get sucessful (1) or unsucessful (0) trials
        wt Tuple | None 
            Trimming window will remove the wt[0] and wt[1] points in the start and end of each trial
        drop_trials_after: bool | True
            If True the trials that are not in trial_type are droped only after
            computing the thresholds which mean that the thresholds will be commom for all types of trials
            specified in trial_types.
        relative: bool | False 
            If threshold is true wheter to do it in ralative (one thr per link per band) or
            an common thr for links per band.
        keep_weights: bool | False 
            If True, will keep the top weights after thresholding.
        q: float | None
            Quartile value to use for thresholding.
        """

        # Check for incorrect parameter values
        assert monkey in ['lucy', 'ethyl'], 'monkey should be either "lucy" or "ethyl"'
        # Input conversion
        if trial_type is not None: trial_type = np.asarray(trial_type)
        if behavioral_response is not None: behavioral_response = np.asarray(behavioral_response)

        # Load session info
        self.info           = GDa.session.session(raw_path=_DATA_PATH, monkey=monkey, date=date, session=session)
        # Storing recording info
        self.recording_info = self.info.recording_info
        # Storing trial info
        self.trial_info     = self.info.trial_info
        #
        del self.info

        # Setting up mokey and recording info to load and save files
        #  self.raw_path = tensor_raw_path
        self.coh_file = coh_file
        self.monkey   = monkey
        self.date     = date
        self.coh_thr  = coh_thr
        self.session  = f'session0{session}'
        self.trial_type          = trial_type
        self.behavioral_response = behavioral_response

        # Load super-tensor
        self.__load_h5(wt)

        # Threshold the super tensor
        if isinstance(q, (int,float)) or isinstance(self.coh_thr, xr.DataArray):
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
        h5_super_tensor_path = os.path.join(_COH_PATH,
                                            self.monkey,
                                            self.date,
                                            self.session,
                                            self.coh_file)

        # Try to read the file in the path specified
        try:
            self.super_tensor = xr.load_dataarray(h5_super_tensor_path)
        except:
            raise OSError(f'File {self.coh_file} not found for monkey')

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
        self.super_tensor.attrs['d_eu'] = self.__get_euclidean_distances()

    def convert_to_adjacency(self,):
        self.A = xr.DataArray( convert_to_adjacency(self.super_tensor.values, self.super_tensor.attrs['sources'],self.super_tensor.attrs['targets']), 
                dims=("sources","targets","freqs","trials","times"),
                coords={"trials":   self.super_tensor.trials.values,
                        "times":    self.super_tensor.times.values,
                        "sources":  self.super_tensor.attrs['channels_labels'],
                        "targets":  self.super_tensor.attrs['channels_labels']})

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
        """
        Filter super-tensor by desired trials based on trial_type and behav. response.

        Parameters
        ----------
        trial_type: int | None
            the type of trial (DRT/fixation)
        behavioral_response: int | None
            Wheter to get sucessful (1) or unsucessful (0) trials
        Returns
        -------
        filtered_trials | array_like
            The number of the trials correspondent to the selected trial_type and behavioral_response
        filtered_trials_idx | array_like
            The index of the trials corresponding to the selected trial_type and behavioral_response
        """
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
        """
        Return a copy of the super-tensor only for the data points correspondent for a
        given experiment stage (baseline, cue, delay, match).

        Parameters
        ----------
        stage: string | None
            Name of the stage from which to get data from.
        - pad: bool | False
            If true will only zero out elements out of the specified task stage.
        Returns
        -------
            Copy of the super-tensor for the stage specified.
        """
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
        """
        Return the number of samples for a given stage for all the trials.

        Parameters
        ----------
        stage: string | None
            Name of the stage from which to get data from.
        total: bool | False
            If True will get the total number of observations for all trials
            otherwise will get the number of observations per trial.
        Returns
        -------
            Return the number of samples for the stage provided for all trials concatenated.
        """
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
        """
        Get the trial averaged super-tensor, it averages togheter the trials for delays in
        the ranges specified by win_delay.

        Parameters
        ----------
        win_delay: array_like | None
               The delay durations that should be averaged together, e.g.,
               if win_delay = [[800, 1000],[1000,1200]] all the trials
               in which the delays are between 800-1000ms will be averaged together,
               likewise for 1000-1200ms, therefore two averaged super-tensors will be
               returnd. If None the average is done for all trials.
        Returns
        -------
            The trial averaged super-tensor.
        """
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
        """
        Get the channels coordinates.
        """
        from pathlib import Path
        _path = os.path.join(Path.home(), _COORDS_PATH)
        xy    = scipy.io.loadmat(_path)['xy']
        return xy

    def __get_euclidean_distances(self, ):
        """
        Get the channels euclidean distances based on their coordinates.
        """
        xy   = self.__get_coords()
        d_eu = np.zeros(len(self.super_tensor.attrs['sources']))
        c1 = self.session_info['channels_labels'].astype(int)[self.session_info['sources']]
        c2 = self.session_info['channels_labels'].astype(int)[self.session_info['targets']]
        dx = xy[c1-1,0] - xy[c2-1,0]
        dy = xy[c1-1,1] - xy[c2-1,1]
        d_eu = np.sqrt(dx**2 + dy**2)
        return d_eu

    def __compute_coherence_thresholds(self, q, relative, keep_weights, verbose):
        if not isinstance(self.coh_thr, xr.DataArray):
            if verbose: print('Computing coherence thresholds')
            self.coh_thr = compute_coherence_thresholds(self.super_tensor.stack(observations=('trials','times')).values,
                                                        q=q, relative=relative, verbose=verbose)
        # Temporarily store the stacked super-tensor
        tmp  = self.super_tensor.stack(observations=('trials','times'))
        # Create the mask by applying threshold
        mask        = xr.DataArray(np.empty(tmp.shape), dims=tmp.dims, coords=tmp.coords)
        mask.values = tmp.values > self.coh_thr.values[...,None]
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
