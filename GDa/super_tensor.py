import numpy  as      np 
import os
from  .io     import set_paths 
from   joblib import Parallel, delayed
class super_tensor(set_paths):

	def __init__(self, raw_path = 'GrayLab/', monkey = 'lucy', date = '150128', 
		         session = 1, delta = 1, freqs = np.arange(6,60,1), trial_subset = None):
		'''
		Constructor method.
		Inputs
			> raw_path : Path containing the raw data.
			> monkey   : Monkey name, should be either lucy or ethyl.
			> date     : The date of the session to use.
			> session  : The number of the session, should be either session01 or session02
		'''
		super().__init__(raw_path = raw_path, monkey = monkey, date = date, session = session)
		###
		self.monkey   = monkey
		self.raw_path = raw_path
		self.date     = date 
		self.session  = 'session0' + str(session)

		# Path to the raw LPF in npy format
		npy_raw_lfp_path = os.path.join('raw_lfp', monkey+'_'+'session0'+str(session)+'_'+str(date)+'.npy')
		# Loading session data
		session_data = np.load(npy_raw_lfp_path, allow_pickle=True).item()
		# Data is deleted to not use memory
		del session_data['data']
		# Copy info in session to the object
		self.session_data = session_data
		# Loading session info
		self.nP      = session_data['info']['nP']
		if trial_subset == None:
			#print(session_data['info']['nT'])
			self.nT      = session_data['info']['nT']
		else:
			self.nT = trial_subset
		self.dir_out = session_data['path']['dir_out']
		self.freqs   = freqs
		self.tarray  = session_data['info']['tarray'][::delta]
		self.pairs   = session_data['info']['pairs']
		
	def load_super_tensor(self, bands = None, average_bands=True):

		self._super_tensor = np.zeros([self.nP, self.nT, self.freqs.shape[0], self.tarray.shape[0]])
		#print('Trial = ' + str(i) + '/540')
		for j in range(self.nP):
			#print('pair = ' + str(j))
			path                        = os.path.join(self.dir_out, 
				                                       'ch1_'+str(self.pairs[j,0])+'_ch2_'+str(self.pairs[j,1])+'.npy' )
			self._super_tensor[j,:,:,:] = np.load(path, allow_pickle=True).item()['coherence']

		if average_bands == True:
			temp = np.zeros([self.nP, self.nT, len(bands), self.tarray.shape[0]])

			for i in range( len(bands) ):
				idx = (self.freqs>=bands[i][0])*(self.freqs<bands[i][1])
				temp[:,:,i,:] = self._super_tensor[:,:,idx,:].mean(axis=2)

			self._super_tensor = temp
			del temp

	def save_npy(self):
		path = os.path.join('super_tensors', self.monkey + '_' + self.session + '_' + self.date + '.npy')
		self.session_data['super_tensor'] = self._super_tensor
		np.save(path, self.session_data)

	def separate_task_stages(self, ):
		None
