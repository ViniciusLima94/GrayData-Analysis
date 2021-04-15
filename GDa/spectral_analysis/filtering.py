import mne.filter

def bp_filter(data = None, fs = 20, f_low = 30, f_high = 60, n_jobs = 1):
    r'''
    Bandpass data.
    > INPUTS:
    - data: Array containing data. Should have dim ("trials","roi","time")
    - fs: Sampling frequency of the signals
    - f_low: Low freq. If None the data are only low-passed
    - f_high: High freq. If None the data are only high-passed.
    - n_jobs: Number of jobs to use
    > OUTPUTS:
    - signal_filtered: Filtered signal
    '''
    
    signal_filtered = mne.filter.filter_data(data, fs, f_low, f_high,
                                             method = 'iir', verbose=False, n_jobs=n_jobs)
    return signal_filtered
