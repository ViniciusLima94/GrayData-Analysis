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
    png = plt.imread('Flatmap_outlines.jpg')
    plt.sca(ax)
    plt.imshow(png, interpolation='none')
    plt.axis('off')
    pad = 10
    plt.xlim(-pad, png.shape[1]+pad)
    plt.ylim(png.shape[0]+pad, -pad)


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

    def plot(self, colormap="viridis", alpha=0.2, colorbar=False,
             vmin=None, vmax=None, extend=None, cbar_title=None,
             figsize=None, dpi=None):
        """
        colormap: string | viridis
            Colormap to use when plotting.
        alpha: float | 0.2
            Transparency of the colored area.
        colorbar: bool | False
            Wheter to display or not the colorbar.
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
        # Get colormap
        cmap = matplotlib.cm.get_cmap(colormap)
        colors = [cmap(val) for val in self.values]

        ####################################################################
        # Create gridspec to plot on
        ####################################################################
        fig = plt.figure(figsize=figsize, dpi=dpi)
        # Checks if needs colorbar
        width_ratios = None
        ncols = 1
        if colorbar:
            width_ratios = (1, 0.1)
            ncols = 2
        gs1 = fig.add_gridspec(nrows=1, ncols=ncols, width_ratios=width_ratios,
                               left=0.05, right=0.95, bottom=0.05, top=0.95)
        ax1 = plt.subplot(gs1[0])
        if colorbar:
            ax2 = plt.subplot(gs1[1])

        ####################################################################
        # Plot flatmap and colors
        ####################################################################
        plot_flatmap(ax1)
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
        if vmin is None:
            vmin = np.min(self.values)
        if vmax is None:
            vmax = np.max(self.values)

        if colorbar:
            norm = matplotlib.colors.Normalize(vmin=vmin,
                                               vmax=vmax)
            cbar = plt.colorbar(
                mappable=plt.cm.ScalarMappable(cmap=colormap, norm=norm),
                cax=ax2, extend=extend)
            cbar.ax.set_ylabel(cbar_title, rotation='vertical')

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

        # Return x and y coordinates
        return self.__FILE[area][:, 0], self.__FILE[area][:, 1]
