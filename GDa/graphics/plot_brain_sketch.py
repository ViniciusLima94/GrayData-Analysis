import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.io

# Path to brainsketch
_ROOT = os.path.expanduser('~/storage1/projects/GrayData-Analysis')
_COORDS = os.path.join(_ROOT, 'Brain Areas/lucy_brainsketch_xy.mat')
_FIG = os.path.join(_ROOT, 'Brain Areas/ethyl_brainsketch.jpg')
# Channels coordinates
xy = scipy.io.loadmat(_COORDS)['xy']
# Brain sketch
ethyl_brainsketch = plt.imread(_FIG)


def plot_node_brain_sketch(channel_label, features, alpha, beta, cmap, sketch=True):
    """
    Plot nodes in the brain sketch.

    Parameters:
    ----------
    channel_label: array_like
        List containing the channel labels.
    features: array_like
        List with the features to be represented in the brain map.
    alpha: float
        Scaling paramter for the node size
    beta: float
        Scaling paramter for the node size
    cmap: matplotlib.colors.Colormap
        Colormap to be used
    sketch: bool | True
        If false show only the nodes and hide the
        brainsketch.
    """
    if sketch:
        plt.imshow(ethyl_brainsketch)
    plt.scatter(xy[channel_label-1, 0],
                xy[channel_label-1, 1],
                s=alpha * np.abs(features) ** beta,
                c=features,
                cmap=cmap)
    plt.axis('off')


def plot_edge_brain_sketch(edge_list, node_list, edge_width):
    plt.imshow(ethyl_brainsketch)
    for i in range(edge_list.shape[0]):
        c1, c2 = node_list[edge_list[i, 0]], node_list[edge_list[i, 1]]
        p1 = [xy[c1-1, 0], xy[c2-1, 0]]
        p2 = [xy[c1-1, 1], xy[c2-1, 1]]
        if edge_width[i] > 0:
            edge_color = 'b'
        else:
            edge_color = 'm'
        plt.plot(p1, p2, '.-', color=edge_color, lw=np.abs(edge_width[i]))
    plt.xticks([])
    plt.yticks([])
