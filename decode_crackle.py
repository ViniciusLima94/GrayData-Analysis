import os
import numpy as np
import xarray as xr
import argparse

from config import get_dates, return_delay_split
from frites.dataset import DatasetEphy
from frites.estimator import GCMIEstimator
from frites.workflow import WfMi
from GDa.util import average_stages
from tqdm import tqdm

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("S_ID", help="which session to do the analysis", type=int)
parser.add_argument("BAND", help="which band to use", type=str)
parser.add_argument("Q", help="threshold used to binarize the data", type=float)
parser.add_argument("MONKEY", help="which monkey to use", type=str)

args = parser.parse_args()

# Index of the session to be load
s_id = args.S_ID
band = args.BAND
tt = 1
br = 1
q = args.Q
at = "cue"
monkey = args.MONKEY


stages = {}
stages["lucy"] = [[-0.4, 0], [0, 0.4], [0.5, 0.9], [0.9, 1.3], [1.1, 1.5]]
stages["ethyl"] = [[-0.4, 0], [0, 0.4], [0.5, 0.9], [0.9, 1.3], [1.1, 1.5]]
stage_labels = ["P", "S", "D1", "D2", "Dm"]

sessions = get_dates(monkey)
s_id = sessions[s_id]

n_rois = 10

##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser("~/funcog/gda")

###############################################################################
# Iterate over all sessions and concatenate power
###############################################################################


# Load power
_FILE_NAME = f"power_tt_{tt}_br_{br}_at_{at}.nc"
path_pow = os.path.join(_ROOT, f"Results/{monkey}/{s_id}/session01", _FILE_NAME)
power = xr.load_dataarray(path_pow)
attrs = power.attrs

# Compute and apply thresholds
thr = power.quantile(q, ("trials", "times"))
power.values = (power >= thr).values

out = []
for t0, t1 in stages[monkey]:
    out += [power.sel(times=slice(t0, t1)).mean("times")]
out = xr.concat(out, "times")
out = out.transpose("trials", "roi", "freqs", "times")
out = out.assign_coords({"trials": attrs["stim"]})
out.attrs = attrs


###############################################################################
# Decoding
###############################################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

bands = {
    "theta": [0, 3],
    "alpha": [6, 14],
    "beta": [14, 26],
    "high_beta": [26, 43],
    "gamma": [43, 80],
}


def compute_cv_scores(times, band, shuffle=False):

    f_0, f_1 = bands[band][0], bands[band][1]

    X = (
        out.groupby("roi")
        .mean("roi")
        .sel(freqs=slice(f_0, f_1), times=times)
        .stack(z=("freqs", "trials"))
    )

    y = X.trials.data.copy()
    X = X.data.copy()

    if shuffle:
        np.random.shuffle(y)

    x_train, x_test, y_train, y_test = train_test_split(
        X.T, y, test_size=0.33, shuffle=True
    )

    max_depths = np.arange(1, 21, 1, dtype=int)
    n_estimators = [100, 200, 500, 1000]
    max_features = ["sqrt", X.shape[0]]

    parameters = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "ccp_alpha": [0, 0.001, 0.01, 0.1],
    }

    est = RandomForestClassifier(n_jobs=20, bootstrap=True)

    clf = GridSearchCV(
        estimator=est,
        param_grid=parameters,
        cv=5,
        scoring="accuracy",
        n_jobs=10,
        return_train_score=True,
        verbose=0,
    )

    clf.fit(x_train, y_train)

    est = RandomForestClassifier(
        **clf.best_params_,
        n_jobs=-1,
    )

    return cross_val_score(est, x_train, y_train, n_jobs=20, cv=5, verbose=0)


times = [[-0.6, -0.2], [0, 0.4], [0.5, 0.9], [0.9, 1.3], [1.1, 1.5]]

cv_scores = []

for time in tqdm(range(5)):
    cv_scores += [compute_cv_scores(time, band)]

cv_scores_surr = []

for time in tqdm(range(5)):
    cv_scores_surr += [compute_cv_scores(time, band, shuffle=True)]

cv_scores = xr.DataArray(np.stack(cv_scores, 0), dims=("times", "k"))
cv_scores_surr = xr.DataArray(np.stack(cv_scores_surr, 0), dims=("times", "k"))

###############################################################################
# Saving results
###############################################################################

# Path to results folder
_RESULTS = os.path.join(_ROOT, f"Results/{monkey}/decoding/crackles")

path_cv = os.path.join(_RESULTS, f"cv_crackle_{band}_{s_id}.nc")
path_cv_surr = os.path.join(_RESULTS, f"cv_surr_crackle_{band}_{s_id}.nc")

cv_scores.to_netcdf(path_cv)
cv_scores_surr.to_netcdf(path_cv_surr)
