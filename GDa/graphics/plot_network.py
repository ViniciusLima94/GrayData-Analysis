import numpy             as np 
import matplotlib.pyplot as plt

def plot_temp_net_ring(adj, degrees, area_names, radius = 1, node_size = 1, link_width = 1):
    Nnodes  = adj.shape[0]
    layers  = adj.shape[2]
    v1,v2   = np.triu_indices_from(adj[:,:,0], k=1)
    d_theta = 2 * np.pi / Nnodes
    theta   = np.arange(0, 2 * np.pi + d_theta, d_theta)
    x, y    = radius*np.cos(theta), radius*np.sin(theta)
    for t in range(layers):
        plt.figure()
        for i, j in zip(v1,v2):
            p1, p2 = [x[i],x[j]], [y[i],y[j]]
            plt.plot(p1, p2, linewidth = link_width * adj[i,j,t], c='k')
        for i in range(Nnodes):
            fs = 10
            if len(area_names[i]) == 3:
                fs = 7.5
            if len(area_names[i]) == 4:
                fs = 6.5
            if len(area_names[i]) == 5:
                fs = 5.5
            plt.text(x[i], y[i], area_names[i],   horizontalalignment='center',
                     verticalalignment='center', fontsize = fs * degrees[i,t] * node_size, 
                     bbox=dict(boxstyle="circle", facecolor='blue', alpha=.8))
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.tight_layout()
        #  plt.savefig('frame_'+str(t)+'.png', dpi = 600)
        #  plt.close()
