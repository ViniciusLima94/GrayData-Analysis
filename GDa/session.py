#####################################################################################################
# Class to read and instantiate a session object
#####################################################################################################
import numpy         as np
import scipy.special as spe
import glob
import os
from  .io            import set_paths, read_mat

class session(set_paths):

	def __init__(self, raw_path = 'GrayLab/', monkey = 'lucy', date = '150128', stype = 'samplecor',
				 session = 1, evt_dt = [-0.65,3.00]):
		'''
		Constructor method.
		Inputs
			> raw_path : Path containing the raw data.
			> monkey   : Monkey name, should be either lucy or ethyl.
			> date     : The date of the session to use.
			> stype    : Session type, should be either samplecor, sampleinc or samplecorinc.
			> session  : The number of the session, should be either session01 or session02
			> evt_dt   : The time window used to align the signal, shoud be a list with [ti, tf].
		'''
		super().__init__(raw_path = raw_path, monkey = monkey, date = date, session = session)
		self.load_mat = read_mat()
		self.stype    = stype
		self.evt_dt   = evt_dt

	def read_session_info(self,):
		'''
		Method to read recording, and trials info dicitionaries.
		'''
		info = ['recording_info.mat', 'trial_info.mat']
		#ri   = self.load_mat.read_mat(self.dir+info[0])['recording_info']
		#ti   = self.load_mat.read_HDF5(self.dir+info[1])['trial_info']
		ri   = self.load_mat.read_mat(os.path.join(self.dir,info[0]) )['recording_info']
		ti   = self.load_mat.read_HDF5(os.path.join(self.dir,info[1]) )['trial_info']

		self.trial_info     = {}
		self.recording_info = {}

		for key in ri._fieldnames:
			if key == 'lfp_sampling_rate':
				self.recording_info['fsample'] = np.squeeze(ri.__dict__[key])
			else:
				self.recording_info[key]       = np.squeeze(ri.__dict__[key])

		for key in ti.keys():
			self.trial_info[key] = np.squeeze(ti[key])

	def read_lfp_data(self,):
		'''
		Method to read the LFP data
		'''

		# Get LFP file names for this session
		#self.files   = sorted(glob.glob(self.dir + '/' + self.date + '*' ))
		self.files   = sorted(glob.glob( os.path.join(self.dir, self.date+'*') ))

		# Find zero time wrt to selected event
		self.t0   = self.trial_info['sample_on']
		self.t0_f = self.trial_info['sample_off']
		self.t1   = self.trial_info['match_on']

		# Get only LFP channels does not contain slvr and ms_mod
		self.indch  = (self.recording_info['slvr'] == 0) & (self.recording_info['ms_mod'] == 0)
		self.indch  = np.arange(1, self.recording_info['channel_count']+1, 1, dtype=int)[self.indch]
		# Trial type
		if  self.stype == 'samplecor':
			# Use only completed trials and correct
			self.indt = (self.trial_info['trial_type'] == 1) & (self.trial_info['behavioral_response'] == 1)
			self.indt = np.arange(1, self.trial_info['num_trials']+1, 1, dtype=int)[self.indt]
		elif self.stype == 'sampleinc':
			# Use only completed trials and incorrect
			self.indt = (self.trial_info['trial_type'] == 1) & (self.trial_info['behavioral_response'] == 0)
			self.indt = np.arange(1, self.trial_info['num_trials']+1, 1, dtype=int)[self.indt]
		elif self.stype == 'samplecorinc':
			# Use all completed correct and incorrect trials
			self.indt = (self.trial_info['trial_type'] == 1)
			self.indt = np.arange(1, self.trial_info['num_trials']+1, 1, dtype=int)[self.indt]

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
		    f = self.load_mat.read_HDF5(self.files[nt-1])
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
		self.nT   = self.T
		# Saving areas only for the channels used
		self.area = self.recording_info['area'][self.indch-1]

		self.readinfo = {'nC': self.C, 'nP':self.nP, 'nT':self.nT, 'pairs':self.pairs,
						 'indt':self.indt, 'fsample': self.recording_info['fsample'],
		                 'tarray': self.time[0], 'channels_labels': self.labels, 'dcue': self.dcue,
		                 'dsm': self.dsm, 'stim':self.stimulus,
		                 'indch': self.indch, 'areas': self.area, 't_cue_on': self.t0[self.indt-1] ,
		                 't_cue_off': self.t0_f[self.indt-1], 't_match_on': self.t1[self.indt-1]
		            	}

	def print_info(self,):
		print('\n-------------------+-----------')
		print('Number of channels | '   + str(self.recording_info['channel_count']))
		print('-------------------+-----------')
		print('Number of trials   | '   + str(self.trial_info['num_trials']))
		print('-------------------+-----------')
		print('Sample frequency   | '   + str(self.recording_info['fsample']) + ' Hz')
		print('-------------------+-----------')

	def save_npy(self):
		session_data = {}
		session_data['data'] = self.data
		session_data['info'] = self.readinfo
		session_data['path'] = {}
		session_data['path']['dir'] = self.dir
		session_data['path']['dir_out'] = self.dir_out
		#np.save('raw_lfp/'+ self.monkey + '_' + self.session + '_' + self.date + '.npy', LFPdata)
		np.save(os.path.join('raw_lfp', self.monkey + '_' + self.session + '_' + self.date + '.npy'), session_data)

	def save_mat(self):
		session_data = {}
		session_data['data'] = self.data
		session_data['info'] = self.readinfo
		session_data['path'] = {}
		session_data['path']['dir'] = self.dir
		session_data['path']['dir_out'] = self.dir_out
		#self.load_mat.save_mat('raw_lfp/'+ self.monkey + '_' + self.session + '_' + self.date + '.mat', LFPdata)
		np.save(os.path.join('raw_lfp', self.monkey + '_' + self.session + '_' + self.date + '.mat'), session_data)
