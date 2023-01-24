import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io
import os

# Root path containing the coordinates file
_ROOT = os.path.expanduser(
    "~/storage1/projects/GrayData-Analysis/GDa/flatmap")


def plot_flatmap(ax):
    """
    Auxiliary function to read flatmap image in jpeg.
    """
    png = plt.imread(os.path.join(_ROOT, 'Flatmap_outlines.jpg'))
    plt.sca(ax)
    plt.imshow(png, interpolation='none')
    plt.axis('off')


class flatmap():

    # Name of the file with the areas' coordinates
    try:
        _FILE_NAME = "all_flatmap_areas.mat"
        _FILE_NAME = os.path.join(_ROOT, _FILE_NAME)
        __FILE = scipy.io.loadmat(_FILE_NAME)
        # Convert all keys to lowercase to avoid problems
        __FILE = {k.lower(): v for k, v in __FILE.items()}
        # Replace / by _ for some areas name is needed to avoid
        # errors
        __FILE = {k.replace("_", "/"): v for k, v in __FILE.items()}
    except FileNotFoundError:
        raise FileNotFoundError("File with coordinates of areas not found.")

    _AREAS = np.array(list(__FILE.keys())[3:])

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
        """
        # Check input types
        assert isinstance(values, (list, tuple, np.ndarray))
        assert isinstance(areas, (list, tuple, np.ndarray))
        # values and areas should have the same size
        assert len(values) == len(areas)

        # Assign inputs to attributes
        self.values = values
        self.areas = areas

    def plot(self, ax, ax_colorbar=None, colormap="viridis", alpha=0.2,
             vmin=None, vmax=None, extend=None, cbar_title=None, cbar_fontsize=12,
             figsize=None, colors=None, dpi=None):
        """
        ax: pyplot.axis | None
            Axis in which to plot the flatmap.
        ax_colorbar: pyplot.axis | None
            Axis in which to plot the colorbar,
            if None no colorbar is ploted.
        colormap: string | viridis
            Colormap to use when plotting.
        alpha: float | 0.2
            Transparency of the colored area.
        vmin: float | None
            Minimum value for the colorbar.
        vmax: float | None
            Maximum value for the colorbar.
        extend: string | None
            To indicate in the coloabr wheter vmin or vmax
            are bigger then the values of the data.
        cbar_title: string | None
            Title of the colorbar.
        figsize: tuple | None
            Size of the figure.
        dpi : int, float | None
            Density of pixel in the plot.
        """
        if vmin is None:
            vmin = np.min(self.values)
        if vmax is None:
            vmax = np.max(self.values)
        norm = matplotlib.colors.Normalize(vmin=vmin,
                                           vmax=vmax)
        # Get colormap
        if isinstance(colormap, str):
            cmap = matplotlib.cm.get_cmap(colormap)
            colors = [cmap(norm(val)) for val in self.values]
        else:
            assert isinstance(colors, (list, np.ndarray, tuple)) 

        ####################################################################
        # Plot flatmap and colors
        ####################################################################
        plot_flatmap(ax)
        # For each pair (value, area) plot in the brain map
        for i, (val, loc) in enumerate(zip(self.values, self.areas)):
            # Get color for region
            c = colors[i]
            # Get coordinates for the region
            X, Y = self.get_flatmap_coordinates(loc)
            plt.fill(X, Y, color=c, alpha=alpha)

        ####################################################################
        # Plot colorbar if needed
        ####################################################################

        if ax_colorbar is not None:
            cbar = plt.colorbar(
                mappable=plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                cax=ax_colorbar, extend=extend)
            cbar.ax.set_ylabel(
                cbar_title, rotation='vertical', fontsize=cbar_fontsize)

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

        assert area in self._AREAS, f"Area {area} not found!"

        # Return x and y coordinates
        return self.__FILE[area][:, 0], self.__FILE[area][:, 1]
