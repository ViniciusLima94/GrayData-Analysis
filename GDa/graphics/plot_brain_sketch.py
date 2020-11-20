import matplotlib.pyplot as plt 
import numpy             as np
import scipy.ndimage 
import scipy.io 

xy                = scipy.io.loadmat('Brain Areas/lucy_brainsketch_xy.mat')['xy'] # Channels coordinates
ethyl_brainsketch = scipy.ndimage.imread('Brain Areas/ethyl_brainsketch.jpg')     # Brainsketch 

def plot_node_brain_sketch(node_list, node_size):
    plt.imshow(ethyl_brainsketch)
    for i,c in zip(range(len(node_list)), node_list):
        plt.plot(xy[c-1,0], xy[c-1,1], 'mo', ms = node_size[i])
    plt.xticks([])
    plt.yticks([])

def plot_edge_brain_sketch(edge_list, node_list, edge_width, edge_color='b'):
    plt.imshow(ethyl_brainsketch)
    for i in range(edge_list.shape[0]):
        c1, c2 = node_list[edge_list[i,0]], node_list[edge_list[i,1]]
        p1  = [xy[c1-1,0], xy[c2-1,0]]
        p2  = [xy[c1-1,1], xy[c2-1,1]]
        plt.plot(p1, p2, '.-', color = edge_color, lw = edge_width[i])
    plt.xticks([])
    plt.yticks([])
