#####################################################################################################
# Read and save the LFP data and information for each trial in numpy format
#####################################################################################################
from   GDa.LFP              import LFP
from   joblib               import Parallel, delayed
import multiprocessing
import mne
import numpy                as     np 
import matplotlib.animation as     animation

nmonkey = 0
nses    = 3
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

# Instatiate LFP object
lfp_data = LFP(raw_path = dirs['rawdata'], monkey = dirs['monkey'][nmonkey], stype = 'samplecor', date = dirs['date'][nmonkey][nses], 
               session  = 1, evt_dt = [-0.65,3.00])
# Read session info
lfp_data.read_session_info()
# Read LFP data
lfp_data.read_lfp_data()
# Saving npy file
lfp_data.save_npy()


info = mne.create_info(ch_names = lfp_data.readinfo['areas'].tolist(),   sfreq= lfp_data.readinfo['fsample'])   
raw = mne.EpochsArray(lfp_data.data, info)
#raw = mne.Epochs(lfp_data.data[0,:,:], info)
#con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(raw, sfre = info['sfreq'], fmin=6, fmax=60, n_jobs=-1)  
st_power, itc, freqs = mne.time_frequency.tfr_array_stockwell(lfp_data.data[0,:,:][np.newaxis, :, :], info['sfreq'], fmin=6, fmax=60, n_jobs=-1)

fig = plt.figure()
ims = []
for i in range(49):
	plt.title('Area: ' + info['ch_names'][i])
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [ms]')
	im = plt.imshow(st_power[i], aspect='auto', cmap='jet', origin='lower', extent=[0,3.51, 6, 60]) 
	ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                repeat_delay=1000)

writergif = animation.PillowWriter(fps=1)
ani.save('power.gif',writer=writergif)