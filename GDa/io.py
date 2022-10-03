import scipy.io as scio
import h5py
import os


class read_mat:

	def __init__(self,):
		None

	def read_HDF5(self, path=None):
		return h5py.File(path, 'r')

	def read_mat(self, path=None):
		return scio.loadmat(path, squeeze_me=True, struct_as_record=False)

	def save_mat(self, path, dic):
		scio.savemat(path, dic)


class set_paths:

	def __init__(self, raw_path='GrayLab/', monkey='lucy', date='150128', session=1):
		self.date = date
		self.monkey = monkey
		self.raw_path = raw_path
		self.session = 'session0' + str(session)  # session

		self.__define_paths()

	def __define_paths(self,):
		self.dir = os.path.join(self.raw_path, self.monkey, self.date, self.session)
		self.dir_out = os.path.join('Results', self.monkey, self.date, self.session)

		# Create out folder in case it not exist yet
		try:
		    os.makedirs(self.dir_out)
		except:
		    None
