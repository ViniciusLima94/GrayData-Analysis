#####################################################################################################
# Class to read and instantiate a LFP object
#####################################################################################################
import numpy         as np 
import scipy.io      as scio
import scipy.special as spe
import h5py
import glob
from  .misc          import set_paths

class LFP(set_paths):

	def __init__(self, raw_path = 'GrayLab/', monkey = 'lucy', date = '150128', stype = 'samplecor', 
				 session = 'session01', evt_dt = [-0.65,3.00]):
		'''
		Constructor
		Inputs: 
		> raw_path : Path containing the raw data.
		> monkey   : Monkey name, should be either lucy or ethyl.
		> type     : Session type, should be either samplecor, sampleinc or samplecorinc. 
		'''
		super().__init__(raw_path = 'GrayLab/', monkey = 'lucy', date = '150128')
		#self.raw_path = raw_path
		#self.monkey   = monkey
		#self.date     = date
		self.stype    = stype
		self.session  = session
		self.evt_dt   = evt_dt 

		self.define_paths()

		# Save session info, such as paths
	'''	
	def define_paths(self,):
		self.dir     = self.raw_path + self.monkey + '/' + self.date + '/' + self.session + '/' 
		self.dir_out = 'Results/'    + self.monkey + '/' + self.date + '/' + self.session + '/' 

		# Create out folder in case it not exist yet
		try:
		    os.makedirs(self.dir_out)
		except:
		    None
	'''
	def read_session_info(self,):
		'''
		Recording and trials info dicitionaries
		'''
		info = ['recording_info.mat', 'trial_info.mat']
		ri = scio.loadmat(self.dir+info[0])['recording_info']
		with h5py.File(self.dir+info[1], 'r') as ti:
			ti = ti['trial_info']
			self.trial_info     = {'num_trials': int(ti['num_trials'][0,0]),
								   'trial_type': ti['trial_type'][:].T[0],
			                       'behavioral_response': ti['behavioral_response'][:].T[0],
			                       'sample_image': ti['sample_image'][:].T[0],
			                       'nonmatch_image': ti['nonmatch_image'][:].T[0],
			                       'match_image': ti['match_image'][:].T[0],
			                       'reaction_time': ti['reaction_time'][:].T[0],
			                       'sample_on': ti['sample_on'][:].T[0], #1th image is shown
			                       'match_on': ti['match_on'][:].T[0],   #2nd image is shown
			                       'sample_off': ti['sample_off'][:].T[0],}

		self.recording_info = {'channel_count': ri['channel_count'].astype(int)[0][0],
		                       'channel_numbers':ri['channel_numbers'][0,0][0],
		                       'area':  ri['area'][0,0][0],
		                       'fsample': ri['lfp_sampling_rate'].astype(int)[0][0],
		                       'ms_mod': ri['ms_mod'][0,0][0],                        
		                       'slvr': ri['slvr'][0,0][0],}   

	def read_lfp_data(self,):

		# Files to read
		self.files   = sorted(glob.glob(self.dir + '/' + self.date + '*' )) 

		# Find zero time wrt to selected event
		self.t0   = self.trial_info['sample_on']
		self.t0_f = self.trial_info['sample_off']
		self.t1   = self.trial_info['match_on']        

		# Get only LFP channels does not contain slvr and ms_mod
		self.indch  = (self.recording_info['slvr'] == 0) & (self.recording_info['ms_mod'] == 0)
		self.indch  = np.arange(1, self.recording_info['channel_count']+1, 1)[self.indch]
		# Trial type
		if  self.stype == 'samplecor':
			# Use only completed trials and correct
			self.indt = (self.trial_info['trial_type'] == 1) & (self.trial_info['behavioral_response'] == 1)
			self.indt = np.arange(1, self.trial_info['num_trials']+1, 1)[self.indt]
		elif self.stype == 'sampleinc':
			# Use only completed trials and incorrect
			self.indt = (self.trial_info['trial_type'] == 1) & (self.trial_info['behavioral_response'] == 0)
			self.indt = np.arange(1, self.trial_info['num_trials']+1, 1)[self.indt]
		elif self.stype == 'samplecorinc':
			# Use all completed correct and incorrect trials 
			self.indt = (self.trial_info['trial_type'] == 1) 
			self.indt = np.arange(1, self.trial_info['num_trials']+1, 1)[self.indt]   

		# Stimulus presented
		self.stimulus = self.trial_info['sample_image'][self.indt-1]-1

		# Duration of the cue (trial dependent)
		self.dcue = (self.t0_f - self.t0)[self.indt-1]
		# Distance sample on match on (trial dependent)
		self.dsm  = (self.t1 - self.t0)[self.indt-1]

		# Record choices, i.e. find which oculomotor choice was performed
		self.choice = np.nan*np.ones(self.trial_info['sample_image'].shape[0])
		# Incorrect response means the monkey chose the nonmatch image
		self.ind = self.trial_info['behavioral_response'] == 0
		self.choice[self.ind] = self.trial_info['nonmatch_image'][self.ind]
		# Correct response means the monkey chose the match image
		self.ind = self.trial_info['behavioral_response'] == 1
		self.choice[self.ind] = self.trial_info['match_image'][self.ind]
		self.trial_info['choice'] = self.choice

		# Data matrix dimensions
		self.L = int( (1000*self.evt_dt[1] - 1000*self.evt_dt[0]) + 1 ) # Number of time points
		self.T = len(self.indt)  # Number of trials
		self.C = len(self.indch) # Number of channels

		self.data = np.empty([self.T, self.C, self.L]) # LFP data
		self.time = np.empty([self.T, self.L])         # Time vector for each trial
		self.trialinfo = np.empty([self.T, 5])         # Info about each trial

		# Loop over trials
		print('Reading data...')
		i = 0
		delta_t   = 1.0 / self.recording_info['fsample']
		for nt in self.indt:
		    # Reading file with LFP data
		    #print(nt)
		    f     = h5py.File(self.files[nt-1], "r")
		    lfp_data = np.transpose( f['lfp_data'] )
		    f.close()
		    # Beginning and ending index
		    indb = int(self.t0[nt-1] + 1000*self.evt_dt[0])
		    inde = int(self.t0[nt-1] + 1000*self.evt_dt[1])
		    # Find time index
		    ind = np.arange(indb, inde+1).astype(int)
		    # Super tensor containing LFP data, dimension NtrialsxNchannelsxTime
		    self.data[i] = lfp_data[self.indch-1, indb:inde+1]
		    # Time vector, one for each trial, dimension NtrialsxTime
		    self.time[i] = np.arange(self.evt_dt[0], self.evt_dt[1]+delta_t, delta_t)
		    # Keep track of [ real trial number, sample image, choice, outcome (correct or incorrect), reaction time ]
		    #self.trialinfo[i] = np.array([nt, self.trial_info['sample_image'][nt-1]-1, self.trial_info['choice'][nt-1], self.trial_info['behavioral_response'][nt-1], self.trial_info['reaction_time'][nt-1]/1000.0])
		    i = i + 1

		# Original channel's labels
		self.labels = self.recording_info['channel_numbers'][self.indch-1].astype(str)
		# Parameters
		self.nP = int( spe.comb(self.C, 2) )
		self.pairs = np.zeros([self.nP, 2], dtype=int)
		count = 0
		for i in range(self.C):
			for j in range(i+1, self.C):
				self.pairs[count, 0] = i 
				self.pairs[count, 1] = j
				count += 1
		# Number of trials
		self.nT = self.T      

		#self.indt = np.arange(self.dt, self.data.shape[2]-self.dt+self.step-self.step, self.step)
		#self.taxs = self.time[0][self.indt]
		self.readinfo = {'nP':self.nP, 'nT':self.nT, 'pairs':self.pairs, 'indt':self.indt, 'fsample': self.recording_info['fsample'],
		            'tarray': self.time[0], 'channels_labels': self.labels, 
		            'dcue': self.dcue, 'dsm': self.dsm, 'stim':self.stimulus}      

	def print_info(self,):
		print('\n-------------------+-----------')
		print('Number of channels | '   + str(self.recording_info['channel_count']))
		print('-------------------+-----------')
		print('Number of trials   | '   + str(self.trial_info['num_trials']))
		print('-------------------+-----------')
		print('Sample frequency   | '   + str(self.recording_info['fsample']) + ' Hz')
		print('-------------------+-----------')

	def print_saved_info(self,):
		print('------------------------+------------------------------------------------------------------------')
		print('Number of channels      | '  + str(self.C) + '/' + str(self.recording_info['channel_count']))
		print('------------------------+------------------------------------------------------------------------')
		print('Number of trials        | '  + str(self.nT) + '/'+ str(self.trial_info['num_trials']))
		print('------------------------+------------------------------------------------------------------------')
		print('Number of channel pairs | '   + str(self.nP) )
		print('------------------------+------------------------------------------------------------------------')
		print('Pairs matrix            | '   + str(list(self.pairs[:5,:])) + '...' )
		print('------------------------+------------------------------------------------------------------------')
		print('Real trial index        | '   + str(self.indt[:10]) + '...' )
		print('------------------------+------------------------------------------------------------------------')
		print('Sample frequency        | '   + str(self.recording_info['fsample']) + ' Hz')
		print('------------------------+------------------------------------------------------------------------')
		print('Time array              | '   + str(self.time[0][:10]) + '...' )
		print('------------------------+------------------------------------------------------------------------')
		print('Real channel labels     | '   + str(self.labels[:10])  + '...')
		print('------------------------+------------------------------------------------------------------------')
		print('Duration sample on/off  | '   + str(self.dcue[:10]) + '...' )
		print('------------------------+------------------------------------------------------------------------')
		print('Duration sample/match   | '   + str(self.dsm[:10]) + '...')
		print('------------------------+------------------------------------------------------------------------')
		print('Stimulus label          | '   + str(self.stimulus[:10]) +  '...')
		print('------------------------+------------------------------------------------------------------------')

	def save_npy(self):
		LFPdata = {}
		LFPdata['data'] = self.data
		LFPdata['info'] = self.readinfo
		np.save('raw_lfp/'+ self.monkey + '_' + self.session + '_' + self.date + '.npy', LFPdata)

	def save_mat(self):
		None