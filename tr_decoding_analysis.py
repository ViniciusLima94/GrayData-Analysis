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
parser.add_argument("BAND", help="which band to use", type=float)
parser.add_argument("ROI", help="which roi to decode from", type=str)
parser.add_argument("MONKEY", help="which monkey to use", type=str)

args = parser.parse_args()

# Index of the session to be load
s_id = args.S_ID
band = args.BAND
roi = args.ROI
at = "cue"
monkey = args.MONKEY


##############################################################################
# Get root path
###############################################################################

_ROOT = os.path.expanduser("~/funcog/gda")

###############################################################################
# Iterate over all sessions and concatenate power
###############################################################################

tt = 1
br = 1

# Load power
_FILE_NAME = f"power_tt_{tt}_br_{br}_at_{at}.nc"
path_pow = os.path.join(_ROOT, f"Results/{monkey}/{s_id}/session01", _FILE_NAME)
power = xr.load_dataarray(path_pow)
attrs = power.attrs


###############################################################################
# Decoding
###############################################################################

import matplotlib.pyplot as plt
from brainconn.centrality import eigenvector_centrality_und
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def compute_cv_scores(data, stim, time_stamp, shuffle=False):

    X = data.isel(times=time_stamp).copy()
    y = stim.copy()

    if shuffle:
        np.random.shuffle(y)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, shuffle=True
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
        n_jobs=20,
        return_train_score=True,
        verbose=0,
    )

    clf.fit(x_train, y_train)

    est = RandomForestClassifier(
        **clf.best_params_,
        n_jobs=-1,
    )

    return cross_val_score(est, x_train, y_train, n_jobs=20, cv=5, verbose=0)


###############################################################################
# Prepare data
###############################################################################

data = power.sel(freqs=band, roi=roi, times=slice(0.5, 0.7))
stim = attrs["stim"]
print(data.shape)
print(stim.shape)

# Cross-validation scores
cvs = []
for t in tqdm(range(data.sizes["times"])):
    cvs += [compute_cv_scores(data, stim, t, shuffle=False)]

cvs = np.stack(cvs, 1)

# Cross-validation scores shuffled
cvs_surr = []
for i in tqdm(range(100)):
    cvs_s = []
    for t in range(data.sizes["times"]):
        cvs_s += [compute_cv_scores(data, stim, t, shuffle=True)]
    cvs_s = np.stack(cvs_s, 1)
    cvs_surr += [cvs_s.copy()]

cvs_surr = np.stack(cvs_surr, 0)

print(cvs.shape)
print(cvs_surr.shape)


cvs = xr.DataArray(cvs, dims=("k", "times"), coords={"times": data.times})

cvs_surr = xr.DataArray(
    cvs_surr, dims=("boot", "k", "times"), coords={"times": data.times}
)


fname = f"cv_{s_id}_{int(band)}_{roi}.nc"
cvs.to_netcdf(os.path.join(_ROOT, "Results", monkey, "decoding", fname))
fname = f"cv_surr_{s_id}_{int(band)}_{roi}.nc"
cvs_surr.to_netcdf(os.path.join(_ROOT, "Results", monkey, "decoding", fname))
