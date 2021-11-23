import numpy as np
import xarray as xr

from tqdm import tqdm
from frites.utils import parallel_func


# Define constants
pi = np.pi


def _is_odd(number):
    return bool(number % 2)


def trial_swap_surrogates(x, seed=0, verbose=False):
    """
    Given the data, randomly swap the trials of the channels.

    Parameters
    ----------
    x: array_like
        data array with dimensions ("trials","roi","time").
    seed: int
        seed used for the trial swapping
    Returns
    -------
    x_surr: array_like
        Data with randomized trials ("trials","roi","time").
    """

    np.random.seed(seed)

    assert isinstance(x, (np.ndarray, xr.DataArray))

    # Get number of nodes and time points
    n_trials, n_nodes = x.shape[0], x.shape[1]

    # Surrogate data
    x_surr = np.zeros_like(x)
    # Array with trial indexes
    trials = np.arange(n_trials, dtype=int)

    itr = range(n_nodes)
    for c in tqdm(itr) if verbose else itr:
        # Swapped indexes
        np.random.shuffle(trials)
        # Attribute the signal in random order to surrogate data
        x_surr[:, c, :] = x[trials, c, :]

    if isinstance(x, xr.DataArray):
        x_surr = xr.DataArray(x_surr, dims=x.dims, coords=x.coords)

    return x_surr


def phase_rand_surrogates(x, seed=0, verbose=False, n_jobs=1):
    """
    PhaseRand_surrogates takes time-series array.
    Phases are coherently randomized, i.e. to preserve the same
    sample covariance matrix as the original
    TS (thus randomizing dFC, but not FC).

    Parameters
    ----------
    x: array_like
        data array with dimensions ("trials","roi","time").
    seed: int | 0
        seed used for the trial swapping
    n_jobs: int | 1
        Number of jobs to parallelize over trials

    Returns
    -------
    x_surr: array_like
        Phase-randomized surrogated signal ("trials","roi","time").
    """

    np.random.seed(seed)

    assert isinstance(x, (np.ndarray, xr.DataArray))

    # Get number of nodes and time points
    n_trials, n_nodes, n_times = x.shape[0], x.shape[1], x.shape[2]

    def _for_trial(trial):
        # Get fft of the signal
        x_fft = np.fft.fft(x[trial, ...], axis=-1)
        # Construct (conjugate symmetric) array of random phases
        phase_rnd = np.zeros(n_times)
        # Define first phase
        phase_rnd[0] = 0
        # In case the number of time points is odd
        if _is_odd(n_times):
            ph = 2*pi*np.random.rand((n_times-1)//2)-pi
            phase_rnd[1:] = np.concatenate((ph, -np.flip(ph, -1)))
        # In case the number of points in even
        if not _is_odd(n_times):
            ph = 2*pi*np.random.rand((n_times-2)//2)-pi
            phase_rnd[1:] = np.concatenate((ph, np.zeros(1), -np.flip(ph, -1)))
        # Randomize the phases of each channel
        x_fft_rnd = np.zeros_like(x_fft)
        for m in range(n_nodes):
            x_fft_rnd[m, :] = np.abs(
                x_fft[m, :])*np.exp(1j*(np.angle(x_fft[m, :])+phase_rnd))
            x_fft_rnd[m, 0] = x_fft[m, 0]
        # Transform back to time domain
        x_rnd = np.fft.ifft(x_fft_rnd, axis=-1)
        return x_rnd.real

    # Parallelize on trials
    parallel, p_fun = parallel_func(
        _for_trial, n_jobs=n_jobs, verbose=verbose,
        total=n_trials)

    x_rnd = parallel(p_fun(t) for t in range(n_trials))
    # Transform to array
    x_rnd = np.asarray(x_rnd)
    # In case the input is xarray converts the output to DataArray
    if isinstance(x, xr.DataArray):
        x_rnd = xr.DataArray(x_rnd, dims=x.dims, coords=x.coords)
    return x_rnd
