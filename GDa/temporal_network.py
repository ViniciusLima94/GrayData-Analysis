import numpy            as     np
from   scipy                 import stats
import os
import h5py

class temporal_network():

    def __init__(self, raw_path = 'super_tensors', monkey='lucy', session=1, date=150128, align_to = 'cue', 
                 trial_type = 1, behavioral_response = 1, wt = None, trim_borders = False):

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
            self.tarray       = self.tarray[wt:-wt] 
            self.super_tensor = self.super_tensor[:,:,:,wt:-wt]

        # Concatenate trials in the super tensor
        self.super_tensor = self.super_tensor.swapaxes(1,2)
        self.super_tensor = self.reshape_observations(self.super_tensor)

        # Will store the null model
        self.A_null = {}
        self.A_null['time']  = {} # Time randomization 
        self.A_null['edges'] = {} # Edges randomization 

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

    def convert_to_adjacency(self,):
        self.A = np.zeros([self.session_info['nC'], self.session_info['nC'], len(self.bands), self.session_info['nT']*len(self.tarray)]) 
        for p in range(self.session_info['pairs'].shape[0]):
            i, j              = self.session_info['pairs'][p,0], self.session_info['pairs'][p,1]
            self.A[i,j,:,:]   = self.super_tensor[p,:,:]

    def compute_coherence_thresholds(self, q = .80):
        #  Coherence thresholds
        self.coh_thr = np.zeros(len(self.bands)) 
        for i in range(len(self.bands)):
            self.coh_thr[i] = stats.mstats.mquantiles( self.super_tensor[:,i,:].flatten(), prob = q )

    def reshape_trials(self, tensor):
        #  Reshape the tensor to have trials and time as two separeted dimension
        #  print(len(tensor.shape))
        if len(tensor.shape) == 1:
            aux = tensor.reshape([self.session_info['nT'], len(self.tarray)])
        if len(tensor.shape) == 2:
            aux = tensor.reshape([tensor.shape[0], self.session_info['nT'], len(self.tarray)])
        if len(tensor.shape) == 3:
            aux = tensor.reshape([tensor.shape[0], tensor.shape[1], self.session_info['nT'], len(self.tarray)])
        if len(tensor.shape) == 4:
            aux = tensor.reshape([tensor.shape[0], tensor.shape[1], tensor.shape[2], self.session_info['nT'], len(self.tarray)])
        return aux

    def reshape_observations(self, tensor):
        #  Reshape the tensor to have all the trials concatenated in the same dimension
        if len(tensor.shape) == 2:
            aux = tensor.reshape([self.session_info['nT'] * len(self.tarray)])
        if len(tensor.shape) == 3:
            aux = tensor.reshape([tensor.shape[0], self.session_info['nT'] * len(self.tarray)])
        if len(tensor.shape) == 4:
            aux = tensor.reshape([tensor.shape[0], tensor.shape[1], self.session_info['nT'] * len(self.tarray)])
        if len(tensor.shape) == 5:
            aux = tensor.reshape([tensor.shape[0], tensor.shape[1], tensor.shape[2], self.session_info['nt'] * len(self.tarray)])
        return aux
