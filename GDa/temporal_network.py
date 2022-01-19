import os
import numpy as np
import xarray as xr
import scipy
import GDa.session

from GDa.util import create_stages_time_grid, filter_trial_indexes
from GDa.net.util import compute_quantile_thresholds, convert_to_adjacency
from .config import (_DEFAULT_TYPE, _COORDS_PATH,
                     _DATA_PATH, _COH_PATH)


class temporal_network():

    ###########################################################################
    # CONSTRUCTOR
    ###########################################################################

    def __init__(self, coh_file=None, coh_sig_file=None, monkey='lucy',
                 session=1, coh_thr=None, date='150128', trial_type=None,
                 behavioral_response=None, wt=None, relative=False,
                 q=None, early_delay=0.3, verbose=False, n_jobs=1):
        """
        Temporal network class, this object will have information about the
        session analysed and store the coherence networks (a.k.a. supertensor).

        Parameters
        ----------
        coh_file: string | None
            Name of the coherence file
        coh_sig_file: array_like | None
            Path to the file containing the signicance values of the coherence
            values with shape (n_roi, n_freqs, n_times). If provided only
            coherence values above the significance values for each roi,
            frequency and time point will be kept.
        monkey: string | 'lucy'
            Monkey name
        session: int | 1
            session number
        coh_thr: array_like | None
            Thresholds to use (if not passed they will be computed
            according to the pars relative and q).
        date: string | '150128'
            date of the recording session
        trial_type: int | None
            the type of trial (DRT/fixation)
        behavioral_response: int | None
            Wheter to get sucessful (1) or unsucessful (0) trials
        wt: Tuple | None
            Trimming window will remove the wt[0] and wt[1] points
            in the start and end of each trial
        relative: bool | False
            If threshold is true wheter to do it in ralative
            (one thr per link per band) or an common thr for links per band.
        q: float | None
            Quartile value to use for thresholding.
        early_delay: float | 0.3
            The period at the beggining of the delay that should
            be used as early delay.
        n_jobs: int | 1
            Number of jobs to use when computing the threshold
            for the coherence tensor. Parallelized over bands.
        """

        # Check for incorrect parameter values
        assert monkey in [
            'lucy', 'ethyl'], 'monkey should be either "lucy" or "ethyl"'
        # Input conversion
        if trial_type is not None:
            trial_type = np.asarray(trial_type)
        if behavioral_response is not None:
            behavioral_response = np.asarray(behavioral_response)

        # Load session info
        info = GDa.session.session(
            raw_path=_DATA_PATH, monkey=monkey, date=date, session=session)
        # Storing trial info
        self.trial_info = info.trial_info
        #
        del info

        # Setting up mokey and recording info to load and save files
        self.coh_file = coh_file
        self.coh_sig_file = coh_sig_file
        self.monkey = monkey
        self.date = date
        self.coh_thr = coh_thr
        self.session = f'session0{session}'
        self.trial_type = trial_type
        self.behavioral_response = behavioral_response
        self.early_delay = early_delay
        # Get how the signal was aligned
        self.align_to = coh_file.split("_")[-1][:3]

        if isinstance(early_delay, float):
            self.stages = ["baseline", "cue", "delay_e", "delay_l", "match"]
        else:
            self.stages = ["baseline", "cue", "delay", "match"]

        # Load super-tensor
        self.__load_h5(wt)

        # Threshold the super tensor if needed
        if isinstance(q,  float) or isinstance(self.coh_thr, xr.DataArray):
            self.__compute_coherence_thresholds(q, relative, verbose, n_jobs)

        # At the end drop select the desired trials
        if trial_type is not None or behavioral_response is not None:
            self.__filter_trials(trial_type, behavioral_response)

    ###########################################################################
    # PRIVATE METHODS
    ###########################################################################

    def __set_path(self,):
        """
        Return path to the coherence tensor path
        """
        # Path to the file
        return os.path.join(_COH_PATH,
                            self.monkey,
                            self.date,
                            self.session,)

    def __load_h5(self, wt):
        """
        Load h5 file containing the coherence tensor
        for the sessions and monkey specified.
        """
        # Path to the file
        h5_super_tensor_path = os.path.join(self.__set_path(), self.coh_file)

        # Try to read the file in the path specified
        try:
            self.super_tensor = xr.load_dataarray(h5_super_tensor_path)
        except FileNotFoundError:
            raise OSError(f'File {self.coh_file} not found for monkey')

        # Copy axes values as class attributes
        self.time = self.super_tensor.times.values
        self.freqs = self.super_tensor.freqs.values

        # Copying metadata as class attributes
        self.session_info = {}
        self.session_info['nT'] = self.super_tensor.sizes["trials"]
        for key in self.super_tensor.attrs.keys():
            self.session_info[key] = self.super_tensor.attrs[key]

        # Crop beginning/ending of super-tensor due to edge effects
        if isinstance(wt, tuple):
            self.time = self.time[wt[0]:-wt[1]]
            self.super_tensor = self.super_tensor[..., wt[0]:-wt[1]]

        # Get euclidean distances
        self.super_tensor.attrs['d_eu'] = self.__get_euclidean_distances()

        # Correct values bellow significance level
        if self.coh_sig_file is not None:
            # Get full path to the file
            sig_file_path = os.path.join(self.__set_path(), self.coh_sig_file)
            # Try to read the file in the path specified
            try:
                sig_values = xr.load_dataarray(
                    sig_file_path).astype(_DEFAULT_TYPE)
            except FileNotFoundError:
                raise OSError(f'File {self.coh_sig_file} not found for monkey')

            # Should be an xarray with dimensions (n_roi, n_freqs, n_times)
            assert isinstance(sig_values, xr.DataArray)
            # Keep the attributes
            cfg = self.super_tensor.attrs
            # Removing values bellow siginificance level
            # self.super_tensor.values = self.super_tensor\
            # * (self.super_tensor >= sig_values).astype(_DEFAULT_TYPE)
            self.super_tensor.values = (
                self.super_tensor - sig_values).astype(_DEFAULT_TYPE)
            # Remove negative values
            self.super_tensor.values = np.clip(self.super_tensor.values,
                                               0,
                                               np.inf)
            # Restoring attributes
            self.super_tensor.attrs = cfg

    def __filter_trials(self, trial_type, behavioral_response):
        """
        Get only selected trials of the super_tensor and its attributes
        """
        filtered_trials, filtered_trials_idx = filter_trial_indexes(
            self.trial_info, trial_type=trial_type,
            behavioral_response=behavioral_response)
        self.super_tensor = self.super_tensor.sel(trials=filtered_trials)
        # Filtering attributes
        for key in ['stim', 't_cue_off', 't_cue_on', 't_match_on']:
            self.super_tensor.attrs[key] = \
                self.super_tensor.attrs[key][filtered_trials_idx]

    def __get_coords(self,):
        """
        Get the channels coordinates.
        """
        from pathlib import Path
        _path = os.path.join(Path.home(), _COORDS_PATH)
        xy = scipy.io.loadmat(_path)['xy']
        return xy

    def __get_euclidean_distances(self, ):
        """
        Get the channels euclidean distances based on their coordinates.
        """
        xy = self.__get_coords()
        d_eu = np.zeros(len(self.super_tensor.attrs['sources']))
        c1 = self.session_info['channels_labels'].astype(
            int)[self.session_info['sources']]
        c2 = self.session_info['channels_labels'].astype(
            int)[self.session_info['targets']]
        dx = xy[c1-1, 0] - xy[c2-1, 0]
        dy = xy[c1-1, 1] - xy[c2-1, 1]
        d_eu = np.sqrt(dx**2 + dy**2)
        return d_eu

    def __compute_coherence_thresholds(self, q, relative, verbose, n_jobs):
        """
        Compute coherence thresholds according to
        the parameters specified (see constructor).
        """
        if not isinstance(self.coh_thr, xr.DataArray):
            if verbose:
                print('Computing coherence thresholds')
            self.coh_thr = compute_quantile_thresholds(
              self.super_tensor.stack(observations=('trials', 'times')).values,
              q=q, relative=relative, verbose=verbose, n_jobs=n_jobs)
        # Temporarily store the stacked super-tensor
        tmp = self.super_tensor.stack(observations=('trials', 'times'))
        # Create the mask by applying threshold
        mask = xr.DataArray(np.empty(tmp.shape),
                            dims=tmp.dims, coords=tmp.coords)
        mask.values = tmp.values > self.coh_thr.values[..., None]
        # Thrshold setting every element above the threshold as 1 otherwise 0
        self.super_tensor.values = mask.unstack().values

    ###########################################################################
    # PUBLIC METHODS
    ###########################################################################

    def convert_to_adjacency(self,):
        """
        Convert coherence tensor to adjacency.
        """
        self.A = xr.DataArray(
            convert_to_adjacency(
                self.super_tensor.values, self.super_tensor.attrs['sources'],
                self.super_tensor.attrs['targets']),
            dims=("sources", "targets", "freqs", "trials", "times"),
            coords={
                "trials":   self.super_tensor.trials.values,
                "times":    self.super_tensor.times.values,
                "freqs":    self.freqs,
                "sources":  self.super_tensor.attrs['areas'],
                "targets":  self.super_tensor.attrs['areas']}).astype(
                    _DEFAULT_TYPE, keep_attrs=True)
        self.A.attrs = self.super_tensor.attrs

    def create_stage_masks(self, flatten=False):
        filtered_trials, filtered_trials_idx = filter_trial_indexes(
            self.trial_info, trial_type=self.trial_type,
            behavioral_response=self.behavioral_response)
        self.s_mask = create_stages_time_grid(
            self.super_tensor.attrs['t_cue_on']-0.2,
            self.super_tensor.attrs['t_cue_off'],
            self.super_tensor.attrs['t_match_on'],
            self.super_tensor.attrs['fsample'],
            self.time, self.super_tensor.sizes['trials'],
            early_delay=self.early_delay,
            align_to=self.align_to,
            flatten=flatten
        )
        if flatten:
            dims = ("observations")
        else:
            dims = ("trials", "times")

        for key in self.s_mask.keys():
            self.s_mask[key] = xr.DataArray(self.s_mask[key], dims=dims)

    def get_thresholded_coherence(self, q, relative=False,
                                  verbose=False, n_jobs=1):
        """
        The same as __compute_coherence_thresholds but instead of
        thresholding the coherence values it returns a copy of the
        coherence tensor thresholded (the super_tensor should not have been
        thresholded before, i.e., q=None and coh_thr=None).

        For information about the parameters see the constructor.
        Be aware because this will increase memory usage.
        """

        coh_thr = compute_coherence_thresholds(
            self.super_tensor.stack(observations=('trials', 'times')).values,
            q=q, relative=relative, verbose=verbose, n_jobs=n_jobs)
        # Temporarily store the stacked super-tensor
        tmp = self.super_tensor.stack(observations=('trials', 'times'))
        # Create the mask by applying threshold
        mask = xr.DataArray(np.empty(tmp.shape),
                            dims=tmp.dims, coords=tmp.coords)
        mask.values = tmp.values > coh_thr.values[..., None]
        # Thrshold setting every element above the threshold as 1 otherwise 0
        return xr.DataArray(mask.unstack().values.astype(bool),
                            dims=self.super_tensor.dims,
                            coords=self.super_tensor.coords,
                            attrs=self.super_tensor.attrs)

    def get_data_from(self, stage=None, pad=False):
        """
        Return a copy of the super-tensor only for the data points
        correspondent for a given experiment
        stage (baseline, cue, delay, match).

        Parameters
        ----------
        stage: string | None
            Name of the stage from which to get data from.
        - pad: bool | False
            If true will only zero out elements out of
            the specified task stage.
        Returns
        -------
            Copy of the super-tensor for the stage specified.
        """
        assert stage in self.stages 
        assert pad in [False, True]

        # Check if the binary mask was already created
        if not hasattr(self, 's_mask'):
            self.create_stage_masks(flatten=True)
        # If the variable exists but the dimensios are not flattened recreate
        if hasattr(self, 's_mask') and len(self.s_mask[stage].shape) == 2:
            self.create_stage_masks(flatten=True)

        if pad:
            return self.super_tensor.stack(
                observations=("trials",
                              "times")) * self.s_mask[stage]
        else:
            return self.super_tensor.stack(
                observations=("trials",
                              "times")).isel(observations=self.s_mask[stage])

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
            Return the number of samples for the stage provided
            for all trials concatenated.
        """
        assert stage in self.stages 

        # Check if the binary mask was already created
        if not hasattr(self, 's_mask'):
            self.create_stage_masks(flatten=False)
        # If the variable exists but the dimensios are not flattened recreate
        if hasattr(self, 's_mask') and len(self.s_mask[stage].shape) == 1:
            self.create_stage_masks(flatten=False)
        if total:
            return np.int(self.s_mask[stage].sum())
        else:
            return self.s_mask[stage].sum(dim='times')

    def get_averaged_st(self, win_delay=None):
        """
        Get the trial averaged super-tensor, it averages togheter the
        trials for delays in the ranges specified by win_delay.

        Parameters
        ----------
        win_delay: array_like | None
               The delay durations that should be averaged together, e.g.,
               if win_delay = [[800, 1000],[1000,1200]] all the trials
               in which the delays are between 800-1000ms will be averaged
               together, likewise for 1000-1200ms, therefore two averaged
               super-tensors will be returnd. If None the average
               is done for all trials.
        Returns
        -------
            The trial averaged super-tensor.
        """
        assert isinstance(win_delay, (type(None), list))

        # Delay duration for each trial
        delay = (self.super_tensor.attrs['t_match_on'] -
                 self.super_tensor.attrs['t_cue_off'])
        delay /= self.super_tensor.attrs['fsample']
        avg_super_tensor = []  # Averaged super-tensor
        # If no window is provided average over all trials otherwise average
        # for each delay duration window.
        if win_delay is None:
            return self.super_tensor.mean(dim='trials')
        else:
            for i, wd in enumerate(win_delay):
                # Get index for delays within the window
                idx = (delay >= wd[0])*(delay < wd[-1])
                avg_super_tensor += [self.super_tensor.isel(
                    trials=idx).mean(dim='trials')]
            return avg_super_tensor

    def reshape_trials(self, ):
        assert len(self.super_tensor.dims) == 3
        self.super_tensor.unstack()

    def reshape_observations(self, ):
        assert len(self.super_tensor.dims) == 4
        self.super_tensor.stack(observations=('trials', 'times'))
