import numpy             as np 
import matplotlib.pyplot as plt

def plot_adjacency(adj, area_names, cmap, figsize):
    adj = adj + adj.T
    plt.figure(figsize=figsize)
    plt.imshow(adj, aspect = 'auto', cmap = cmap, origin = 'lower', vmin=0, vmax=1)
    plt.xticks(range(adj.shape[0]), area_names, rotation = 90)
    plt.yticks(range(adj.shape[0]), area_names)
