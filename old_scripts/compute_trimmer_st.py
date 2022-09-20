
###############################################################################
# Compute trimmer strengths
###############################################################################


def compute_trimer_st(MC, x_s, x_t):
    # Get number of rois based on source/targets arrays
    n_rois = int(np.hstack((x_s, x_t)).max() + 1)
    ts = np.zeros(n_rois)
    for i in range(n_rois):
        idx = np.logical_or(x_s == i, x_t == i)
        ts[i] = MC[np.ix_(idx, idx)].sum()
    return ts


def trimmer_strength(MC, n_jobs=1, verbose=False):
    """
    Compute trimmer strengths for meta connectivity
    tensor (roi, roi, times).
    """

    # Get rois names
    roi_s, roi_t = _extract_roi(MC.sources.data, "-")
    # Create a mapping to track rois and indexes
    mapping = creat_roi_mapping(roi_s, roi_t)
    x_s, x_t = area2idx(roi_s, mapping), area2idx(roi_t, mapping)
    # Get number of rois based on source/targets arrays
    n_rois = int(np.hstack((x_s, x_t)).max() + 1)
    # Get number of times and trials
    n_times = MC.sizes["times"]
    nt = n_times 

    # Stack trials and times
    A = MC.data

    # Compute for a single observation
    def _for_frame(t):
        # Call core function
        Tst = compute_trimer_st(A[..., t], x_s, x_t)
        return Tst

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_frame, n_jobs=n_jobs, verbose=verbose,
        total=nt)
    # Compute the single trial coherence
    Tst = parallel(p_fun(t) for t in range(nt))
    # Convert to numpy array
    Tst = np.asarray(Tst)

    # Unstack trials and time
    Tst = Tst.reshape((n_rois, n_times))

    Tst = xr.DataArray(Tst,
                       dims=("roi", "times"),
                       coords={
                           "roi": list(mapping.keys()),
                           "times": MC.times.data,
                       })
    return Tst


def creat_roi_mapping(x_s, x_t):
    """
    Create hash table to access rois names
    x_s: array_like
        Source areas names
    x_t: array_like
        Target areas names
    """
    # Get unique names
    areas = np.sort(np.unique(np.hstack((x_s, x_t))))
    # Number of areas
    n_areas = len(areas)
    # Numerical index of each area
    idx = np.arange(0, n_areas, 1, dtype=np.int32)
    # Create mapping
    mapping = dict(zip(areas, idx))
    return mapping


def area2idx(areas, mapping):
    """
    Given a list with names of areas and a mapping
    return the indexes correponding to the areas.
    """
    return np.asarray([mapping[a] for a in areas])


Tst = []
for f in range(MC.sizes["freqs"]):
    Tst += [trimmer_strength(MC.isel(freqs=f), n_jobs=20)]
Tst = xr.concat(Tst, "freqs").transpose("roi", "freqs", "times")
Tst.attrs = MC.attrs
