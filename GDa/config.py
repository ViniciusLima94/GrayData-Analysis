import numpy as np
import os

# Define default return type
_DEFAULT_TYPE = np.float32
# Defining default paths
_ROOT = '~/storage1/projects/GrayData-Analysis/'
_COORDS_PATH = os.path.expanduser(_ROOT+'Brain Areas/lucy_brainsketch_xy.mat')
_DATA_PATH = os.path.expanduser(_ROOT+'GrayLab')
_COH_PATH = os.path.expanduser(_ROOT+'Results')