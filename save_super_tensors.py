import sys
import time
import numpy                      as     np
from   GDa.super_tensor           import super_tensor

idx = 3#int(sys.argv[-1])

nmonkey = 0
nses    = 6
ntype   = 0
#####################################################################################################
# Directories
#####################################################################################################
dirs = {'rawdata':'GrayLab/',
        'results':'Results/',
        'monkey' :['lucy', 'ethyl'],
        'session':'session01',
        'date'   :[['141014', '141015', '141205', '150128', '150211', '150304'], []]
        }

# Frequency axis used to compute coherence
freqs = np.arange(4,60,2)
# Delta used to downsample time in coherence computation
delta = 15
# Intantiating supe_tensor class
st = super_tensor(raw_path = 'GrayLab/', monkey = 'lucy', 
                  date = '150128', session = 1, delta = delta, freqs = freqs, trial_subset = None)

# Frequency bands to be averaged
bands = np.array([[4,8],[8,15],[15,30],[30,60]])

if  __name__ == '__main__':

	import time

	start = time.time()
	st.load_super_tensor(bands = bands, average_bands=True)
	end = time.time()

	print('Elapsed time to load super-tensor: ' +str((end - start)/60) + ' min.' )

	st.save_npy()