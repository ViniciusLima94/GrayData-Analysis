import numpy                as     np
from   GDa.misc.reshape     import reshape_trials, reshape_observations
from   GDa.net.util         import compute_thresholds
from   scipy                import stats
import os
import h5py

class temporal_network():

    def __init__(self, raw_path = 'super_tensors', monkey='lucy', session=1, date=150128, align_to = 'cue', 
                 trial_type = 1, behavioral_response = 1, wt = (None,None), trim_borders = False,
                 threshold=False, relative=False, q=0.8):
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
        - wt: Tuple. Trimming window will remove the wt[0] and wt[1] points in the start and end of each trial
        - trim_borders: Wheter to trim or not the start/end of the super_tensor for each trial
        - threshold: Wheter to threshold or not the data
        - relative: if threshold is true wheter to do it in ralative (one thr per link per band) or an common thr for links per band
        - q: Quartile value to use for thresholding
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

        # Setting up mokey and recording info to load and save files
        self.raw_path = raw_path
        self.monkey   = monkey
        self.date     = date                            
        self.session  = 'session0' + str(session)       
        self.trial_type          = trial_type
        self.align_to            = align_to
        self.behavioral_response = behavioral_response
        
        # Load super-tensor
        self.__load_h5()

        if trim_borders == True:
            self.tarray       = self.tarray[wt[0]:-wt[1]]
            self.super_tensor = self.super_tensor[:,:,:,wt[0]:-wt[1]]

        # Concatenate trials in the super tensor
        self.super_tensor = self.super_tensor.swapaxes(1,2)
        self.super_tensor = self.reshape_observations()

        # Will store the null model
        self.A_null = {}
        self.A_null['time']  = {} # Time randomization 
        self.A_null['edges'] = {} # Edges randomization 

        # Threshold the super tensor
        if thr:
            coh_thr = compute_thresholds(self.super_tensor, q=q, relative=relative)
            for i in range( len(self.bands) ):
                self.super_tensor[:,i,:] = self.super_tensor[:,i,:]>coh_thr[i]

    def __load_h5(self,):
        # Path to the super tensor in h5 format 
        h5_super_tensor_path = os.path.join(self.raw_path, self.monkey+'_'+str(self.session)+'_'+str(self.date)+'.h5')

        try:
            hf = h5py.File(h5_super_tensor_path, 'r')
        except (OSError):
            raise OSError('File for monkey ' + str(self.monkey) + ', date ' + str(self.date) + ' ' + self.session + ' not created yet')

        group = os.path.join('trial_type_'+str(self.trial_type), 
                             'aligned_to_' + str(self.align_to),
                             'behavioral_response_'+str(self.behavioral_response)) 

        g1 = hf.get(group)
        # Read coherence data
        self.super_tensor = g1['coherence'][:]
        self.tarray       = g1['tarray'][:]
        self.freqs        = g1['freqs'][:]
        self.bands        = g1['bands'][:]
        # Read info dict.
        self.session_info = {}
        for k in g1['info'].keys():
            if k == 'nC':
                self.session_info[k] = int( np.squeeze( np.array(g1['info/'+k]) ) )
            else:
                self.session_info[k] = np.squeeze( np.array(g1['info/'+k]) )

#    def compute_coherence_thresholds(self, q = .80, relative = False):
#        #  Coherence thresholds
#        if relative:
#            self.coh_thr = np.zeros([len(self.bands), self.session_info['pairs'].shape[0]]) 
#            for i in range(len(self.bands)):
#                self.coh_thr[i] = np.squeeze( stats.mstats.mquantiles(self.super_tensor[:,i,:], prob=q, axis=1) )
#        else:
#            self.coh_thr = np.zeros(len(self.bands)) 
#            for i in range(len(self.bands)):
#                self.coh_thr[i] = stats.mstats.mquantiles(self.super_tensor[:,i,:].flatten(), prob=q)
#
#    def convert_to_adjacency(self, thr = None):
#        self.A = np.zeros([self.session_info['nC'], self.session_info['nC'], len(self.bands), self.session_info['nT']*len(self.tarray)]) 
#        for p in range(self.session_info['pairs'].shape[0]):
#            i, j              = self.session_info['pairs'][p,0], self.session_info['pairs'][p,1]
#            self.A[i,j,:,:]   = self.super_tensor[p,:,:]

    def reshape_trials(self, ):
        #  Reshape the tensor to have trials and time as two separeted dimension
        #  print(len(tensor.shape))
        # if len(tensor.shape) == 1:
        #     aux = tensor.reshape([self.session_info['nT'], len(self.tarray)])
        # if len(tensor.shape) == 2:
        #     aux = tensor.reshape([tensor.shape[0], self.session_info['nT'], len(self.tarray)])
        # if len(tensor.shape) == 3:
        #     aux = tensor.reshape([tensor.shape[0], tensor.shape[1], self.session_info['nT'], len(self.tarray)])
        # if len(tensor.shape) == 4:
        #     aux = tensor.reshape([tensor.shape[0], tensor.shape[1], tensor.shape[2], self.session_info['nT'], len(self.tarray)])
        aux = reshape_trials( self.super_tensor, self.session_info['nT'], len(self.tarray) )
        return aux

    def reshape_observations(self, ):
        #  Reshape the tensor to have all the trials concatenated in the same dimension
        # if len(tensor.shape) == 2:
        #     aux = tensor.reshape([self.session_info['nT'] * len(self.tarray)])
        # if len(tensor.shape) == 3:
        #     aux = tensor.reshape([tensor.shape[0], self.session_info['nT'] * len(self.tarray)])
        # if len(tensor.shape) == 4:
        #     aux = tensor.reshape([tensor.shape[0], tensor.shape[1], self.session_info['nT'] * len(self.tarray)])
        # if len(tensor.shape) == 5:
        #     aux = tensor.reshape([tensor.shape[0], tensor.shape[1], tensor.shape[2], self.session_info['nt'] * len(self.tarray)])
        aux = reshape_observations( self.super_tensor, self.session_info['nT'], len(self.tarray) )
        return aux
