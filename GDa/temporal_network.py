import numpy                as     np
import xarray               as     xr
import GDa.session          
from   GDa.misc.reshape     import reshape_trials, reshape_observations
from   GDa.net.util         import compute_coherence_thresholds, convert_to_adjacency
from   scipy                import stats
import os
import h5py

class temporal_network():

    def __init__(self, data_raw_path='GrayLab/', tensor_raw_path='super_tensors', monkey='lucy', session=1, 
                 date='150128', wt=(None,None), trim_borders=False, threshold=False, relative=False, q=0.8):
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
        - trim_borders: Wheter to trim or not the start/end of the super_tensor for each trial
        - threshold: Wheter to threshold or not the data
        - relative: if threshold is true wheter to do it in ralative (one thr per link per band) or an common thr for links per band
        - q: Quartile value to use for thresholding
        '''

        #Check for incorrect parameter values
        if monkey not in ['lucy', 'ethyl']:
            raise ValueError('monkey should be either "lucy" or "ethyl"')

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
        
        # Load super-tensor
        self.__load_h5(trim_borders, wt)

        # Threshold the super tensor
        if threshold:
            print('Computing coherence thresholds')
            self.coh_thr      = compute_coherence_thresholds(self.super_tensor.values, q=q, relative=relative)
            self.super_tensor.values = (self.super_tensor>self.coh_thr).values

    def __load_h5(self,trim_borders, wt):
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

        if trim_borders == True:
            self.tarray       = self.tarray[wt[0]:-wt[1]]
            self.super_tensor = self.super_tensor[:,:,:,wt[0]:-wt[1]]

        # Concatenate trials in the super tensor
        self.super_tensor = self.super_tensor.swapaxes(1,2)
        #self.super_tensor = self.reshape_observations()

        # Convert to xarray
        self.super_tensor = xr.DataArray(self.super_tensor, dims=("links","bands","trials","time"),
                                         coords={"trials": self.trial_info.index.values, 
                                                 "time":   self.tarray} )
        # Metadata (the same as session_info)
        self.super_tensor.attrs = self.session_info

    def convert_to_adjacency(self, ):
        self.A = xr.DataArray( convert_to_adjacency(self.super_tensor.values), 
                dims=("roi_1","roi_2","bands","trials","time"),
                coords={"trials": self.trial_info.index.values,
                        "time":   self.tarray,
                        "roi_1":  self.super_tensor.attrs['channels_labels'],
                        "roi_2":  self.super_tensor.attrs['channels_labels']})

    def reshape_trials(self, ):
        aux = reshape_trials( self.super_tensor, self.session_info['nT'], len(self.tarray) )
        return aux

    def reshape_trials(self, ):
        aux = reshape_trials( self.super_tensor, self.session_info['nT'], len(self.tarray) )
        return aux

    def reshape_observations(self, ):
        aux = reshape_observations( self.super_tensor, self.session_info['nT'], len(self.tarray) )
        return aux
