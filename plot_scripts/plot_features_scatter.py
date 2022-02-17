import os
import scipy

from pathlib import Path
from GDa.config import _COORDS_PATH

_path = os.path.join(Path.home(), _COORDS_PATH)
xy = scipy.io.loadmat(_path)['xy']
