import numpy  as np
import xarray as xr

from   frites.utils import parallel_func


# Define constants
pi = np.pi

def _is_odd(number):
    return bool(number%2)

def phase_rand_surrogates(x, val=1, seed=0, verbose=False, n_jobs=1):
    r'''
    PhaseRand_surrogates(x, val) takes time-series array (x) and type of randomization (val)
    as inputs and provides phase-randomized TS as output.
    > INPUTS:
    - x: data array with dimensions ("trials","roi","time").
    - val: '1' (default) phases are coherently randomized, i.e. to preserve the same sample covariance matrix as the
                original TS(thus randomizing dFC, but not FC) '0' phases are incoherently randomized (thus randomizing
                both dFC and FC)
    - n_jobs: Number of jobs to parallelize over trials
    > OUTPUTS:
    x_surr: Surrogated signal.
    '''
    
    np.random.seed(seed)

    assert isinstance(x, (np.ndarray, xr.DataArray))
    assert isinstance(val, int)

    # Get number of nodes and time points
    n_trials, n_nodes, n_times = x.shape[0], x.shape[1], x.shape[2]

    def _for_trial(trial):
        # Get fft of the signal
        x_fft = np.fft.fft(x[trial], axis=-1)

        if val==0:
            phase_rnd = np.zeros((n_nodes,n_times))
            # In case the number of time points is odd
            if _is_odd(n_times):
                ph = 2*pi*np.random.rand(n_nodes,(n_times-1)//2)-pi
                phase_rnd[:,1:] = np.concatenate((ph,-ph[:,::-1]),axis=-1) 
            # In case the number of points in even
            if not _is_odd(n_times):
                ph = 2*pi*np.random.rand(n_nodes,(n_times-2)//2)-pi
                phase_rnd[:,1:] = np.concatenate((ph,np.zeros((n_nodes,1)),-ph[:,::-1]),axis=-1)
            # Randomize the phases of each channel
            x_fft_rnd = np.zeros_like(x_fft)
            for m in range(n_nodes):
                x_fft_rnd[m,:] = np.abs(x_fft[m,:])*np.exp(1j*(np.angle(x_fft[m,:])+phase_rnd[m,:]))
                x_fft_rnd[m,0] = x_fft[m,0]
        if val==1:
            # Construct (conjugate symmetric) array of random phases
            phase_rnd    = np.zeros(n_times)
            # Define first phase
            phase_rnd[0] = 0
            # In case the number of time points is odd
            if _is_odd(n_times):
                ph = 2*pi*np.random.rand((n_times-1)//2)-pi
                phase_rnd[1:] = np.concatenate((ph,-ph))
            # In case the number of points in even
            if not _is_odd(n_times):
                ph = 2*pi*np.random.rand((n_times-2)//2)-pi
                phase_rnd[1:] = np.concatenate((ph,np.zeros(1),-ph[::-1]))
            # Randomize the phases of each channel
            x_fft_rnd = np.zeros_like(x_fft)
            for m in range(n_nodes):
                x_fft_rnd[m,:] = np.abs(x_fft[m,:])*np.exp(1j*(np.angle(x_fft[m,:])+phase_rnd))
                x_fft_rnd[m,0] = x_fft[m,0]
        # Transform back to time domain
        x_rnd = np.fft.ifft(x_fft_rnd)
        return x_rnd.real

    # Parallelize on trials
    parallel, p_fun = parallel_func( 
                      _for_trial, n_jobs=n_jobs, verbose=verbose,
                      total=n_trials)

    x_rnd =  parallel(p_fun(t) for t in range(n_trials))
    # Transform to array
    x_rnd = np.asarray(x_rnd)   
    # In case the input is xarray converts the output to DataArray
    if isinstance(x, xr.DataArray):
        x_rnd = xr.DataArray(x_rnd, dims=x.dims, coords=x.coords)
    return x_rnd 

if __name__=='__main__':

    import sys
    # Get paths and insert path
    sys.path.insert(1, '/home/vinicius/storage1/projects/GrayData-Analysis')
    import scipy.io  as scio
    import glob
    from   GDa.session import *

    def read_HDF5(path = None):
        return h5py.File(path, 'r')

    # Get file names for session 150128
    files = sorted(glob.glob( '../../GrayLab/lucy/150128/session01/150128*' ))

    # Get info for this session
    info  = session_info(raw_path='../../GrayLab/', monkey='lucy', date='150128', session=1)

    # Number of trials selected
    #  n_trials   = len(info.trial_info)
    #  # Number of time points
    #  n_times    = int(info.recording_info['lfp_sampling_rate'] * (self.evt_dt[1]-self.evt_dt[0]))
    #  # Number of channels selected
    #  n_channels = len(indch)

    # Load a trial of the data
    data = []
    for i in range(len(files)):
        f        = read_HDF5(files[i])
        data    += [np.transpose( f['lfp_data'] )]
                
    # Get a trial of the data
    x = np.array( data[0] )
        
