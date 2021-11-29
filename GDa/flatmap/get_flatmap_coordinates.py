import numpy as np
import os
import scipy.io


class flatmap():

    _FILE_NAME = "all_flatmap_areas.mat"
    _AREAS = np.array(['Pi', 'V1', 'V2', 'V4t', 'DP', 'V3',
                       'V3A', 'a5', 'a7M', 'AIP', 'VIP', 'V6A',
                       'V6', 'PPT', 'MT', 'TEO', 'TEOm', 'PG',
                       'TPOC', 'TPt', 'TPO', 'TEpv', 'TEpd',
                       'FST', 'MST', 'a2', 'a7A', 'a1', 'PBc', 'a7B',
                       'LIP', 'MIP', 'PIP', 'a3', 'a7op', 'F1',
                       'SII', 'F3', 'a24D', 'F5', 'F4', 'F2', 'a44',
                       'OPRO', 'ProM', 'a23', 'a8M', 'a24c', 'a8L', 'F7',
                       'F6', 'a45B', 'a9/46V', 'a46d', 'a46V', 'a8B', 'a8r',
                       'a45A', 'a8/32', 'a9/46D', 'a9', 'a11', 'a12', 'a13',
                       'a14', 'a32', 'Ins', 'PGa', 'STPc', 'PBr', 'LB',
                       'Core', 'MB'])

    def __init__(self, values=None, areas=None, cmap="viridis"):
        """
        Constructor method. Receive the values that will be plotted
        in the flatmap and the respective areas in which the values
        will be displayed.

        Parameters:
        ----------

        values: array_like | None
            Values that will be used to plot on the flatmap
            (e.g., number of channels, power, mutual information).
        areas: array_like | None
            Areas in which the values will be plotted.
        cmap: string | viridis
        """
        # Check input types
        assert isinstance(values, (list, tuple, np.ndarray))
        assert isinstance(areas, (list, tuple, np.ndarray))
        # values and areas should have the same size
        assert len(values) == len(areas)

        # Assign inputs to attributes
        self.values = values
        self.areas = areas

    def get_flatmap_coordinates(self, area):
        """
        Gets the coordinates to fill in an area on the flatmap.
        Will load the "all_flatmpap_areas.mat" and find the
        area that is input that
        matches the all_flatmap_areas.mat area.
        Use for plotting on the "Flatmap_outlines.jpg".

        Parameters:
        ----------

        area: string
            Name of the area in which it will be plotted in the
            flatmap.
        """

        assert area in self._AREAS, "Area not found!"

        # Check if file existis in the folder
        if not os.path.isfile(self._FILE_NAME):
            raise FileNotFoundError(f'No file {self._FILE_NAME} found')

        file = scipy.io.loadmat(self._FILE_NAME)

        return file["area"]
