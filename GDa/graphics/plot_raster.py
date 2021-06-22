import numpy             as np
import matplotlib.pyplot as plt 

def plot_nodes_raster_all_bands(m, xmin, xmax, ymin, ymax, vmin, vmax, ylabel, xlabel, titles, figsize, cmap, thrs=None):
    plt.figure(figsize=figsize)
    for i in range(4):
        plt.subplot(2,2,i+1)
        if type(thrs)==type(None):
            plt.imshow(m[:,i,:], aspect = 'auto', cmap = cmap, origin = 'lower', vmin=vmin, vmax=vmax, extent=[xmin, xmax, ymin, ymax])
        else:
            plt.imshow(m[:,i,:]>thrs[i], aspect = 'auto', cmap = cmap, origin = 'lower', vmin=vmin, vmax=vmax, extent=[xmin, xmax, ymin, ymax])
        plt.colorbar()
        plt.title(titles[i])
        if i == 0 or i == 2:
            plt.ylabel(ylabel)
        if i == 2 or i == 3:
            plt.xlabel(xlabel)

        

