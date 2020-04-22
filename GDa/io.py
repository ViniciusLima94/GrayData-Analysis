import numpy         as np 


class set_paths:

	def __init__(self, raw_path = 'GrayLab/', monkey = 'lucy', date = '150128', session = 'session01'):
		self.date     = date
		self.monkey   = monkey
		self.raw_path = raw_path
		self.session  = session 
		
		self.define_paths()

	def define_paths(self,):
		self.dir     = self.raw_path + self.monkey + '/' + self.date + '/' + self.session + '/' 
		self.dir_out = 'Results/'    + self.monkey + '/' + self.date + '/' + self.session + '/' 

		# Create out folder in case it not exist yet
		try:
		    os.makedirs(self.dir_out)
		except:
		    None