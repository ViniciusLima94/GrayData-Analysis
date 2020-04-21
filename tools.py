import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as scio

import seaborn as sns

from MulticoreTSNE import MulticoreTSNE as TSNE

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler

'''
Auxiliar functions to read the coherence matrices, for each pair.
'''
def filename(session, date, trial, ch1, ch2):
    ''' 
        Returns the name of the file where the coherence matrix
        of a given trial between channel 1 and channel 2.
        trial: trial number
        ch1:   channel 1 number
        ch2:   channel 2 number
    '''
    return session+'_'+str(date)+'_trial_'+str(trial)+'_pair_'+str(ch1)+'_'+str(ch2)+'.mat'


def loadcohm(session, date, trial, ch1, ch2, folder):
    '''
        Reads the coherence matrix .mat file.
        trial: trial number
        ch1:   channel 1 number
        ch2:   channel 2 number
        folder:   directory where the files are located
    '''
    return scio.loadmat( folder+filename(session, date, trial, ch1, ch2) )['trial']

def computemeancoh(cue, nC, nT, ch1, ch2, folder):
    '''
        Compute the mean coherence matrix for a given pair of channels, using trials
        in which a specific cue was presented.
    '''
    trials = np.arange(0, nT).astype(int) # Trials number
    idx    = cue == nC                    # Get index where cue nC was presented
    n      = trials[idx]                  # Getting trials number where the cue nC was presented

    coh    = np.zeros([28, 77])           # Coherence matrix
    # Computing mean coherence matrix
    for j in n:
        coh += loadcohm(j, ch1, ch2, folder) / float( len(n) )
    return np.reshape( coh, (1, 28*77) )

#--------------------------------------------------------------------------
# Clustering analysis
#--------------------------------------------------------------------------
def silhoutte_(X, krange=np.arange(2,15)):
	'''
		Compute the mean silhouette score for each number of clusters
        in krange.
        Inputs:
        X: Input data
        krange: Number of clusters to test, must be list (or array).
        Outputs:
        krange: Array containing number os clusters
        s_score: Array containing silhoutte scores for each number of clusters
	'''
	s_score = [] # Stores silhoette scores
    # For each number of clusters
	for k in krange:
		kmeans = KMeans(n_clusters=k, n_init=100, max_iter = 600).fit(X)
		y_kmeans = kmeans.predict(X)
		s_score.append( silhouette_score(X, y_kmeans) )
	return krange, np.array(s_score)

def elbow_(X, krange=np.arange(2,15)):
    '''
        Compute the elbow curve, dependent of the numver of clusters
        in krange.
        Inputs:
        X: Input data
        krange: Number of clusters to test, must be list (or array)
        Outputs:
        krange: Array containing number os clusters
        Sum_of_squared_distances: Sum of squared distances of samples to their closest cluster center.
        for each number of clusters.
	'''
    Sum_of_squared_distances = []
    for k in krange:
        kmeans = KMeans(n_clusters=k).fit(X)
        Sum_of_squared_distances.append(kmeans.inertia_)
    return krange, Sum_of_squared_distances

def samples_idx(X, k, nsamples, thr):
    '''
        Computes the silhouette score for each sample with a given number of clusters K.
        And returns the idexes samples with silhouette score grater than thr and cluster labels.
        Inputs:
        X: Input data
        k: Number of clusters
        nsamples: Total number of samples
        thr: Threshold to include or not a given sample
        Outputs:
        sample_number[idx]: Sample indexes which has silhoutte score > thr.
        sample_number[nidx]: Sample indexes which has silhoutte score < thr.
        y_kmeans[idx]: kmeans labels for sample samples with silhoutte score > thr
    '''
    sample_number = np.arange(0, nsamples) # Index of the samples
    kmeans = KMeans(n_clusters=k).fit(X)
    y_kmeans = kmeans.predict(X)
    s      = silhouette_samples(X, y_kmeans) # Compule the silhouette score for each sample
    return (s >= thr).astype(bool)

def contingency_matrix_(links, Nclusters, Nbands, norm = True):
    '''
        Build the contingency matrices between two clusters.
        Inputs:
        links: Numbers of pairs composing each cluster in each frequency band,
        is a dictionary with Nbands for each frequency band, each Nband cells
        has Nclusters entries for each cluster containing the pair numbers which 
        compose the k-th cloud of the cluster.
        Nclusters: Number of clusters.
        Nbands: Number of frequency bands.
        norm: Whether to normalize the matrix or not (default True)
        Outputs:
        a: Contingency matrix with dimensions [Nbands*Nclusters, Nbands*Nclusters,].
        
    '''
    a = []
    for i in range(Nbands):
        for k in range(Nclusters):
            for j in range(Nbands):
                for w in range(Nclusters):
                    if norm == True:
                        a.append( 1.0*len(np.intersect1d(links[i][k], links[j][w])) / max(len(links[i][k]), len(links[j][w]))  )
                    else:
                        a.append( 1.0*len(np.intersect1d(links[i][k], links[j][w])) )
    return np.reshape(a, (Nclusters*Nbands, Nclusters*Nbands))

def plot_contingency_matrix(cm, Nclusters, Nbands, file_name=None, color = 'k', save=False):
    '''
        Plot the contingency matrix with specific delimitations.
        Inputs:
        cm: Contingency matrix
        Nclusters: Number of clusters.
        Nbands: Number of frequency bands.
        save: Wheter to save or not the figure (default false)
        Outputs:
        Plot the figure
    '''
    # Matrix dimensions
    r, c = cm.shape
    delta = 0.5
    h_min, h_max = 0-delta, c - delta
    plt.imshow(cm, aspect='auto', cmap='jet')
    hpos = Nclusters
    while hpos < r - delta:
        plt.hlines(hpos-delta, h_min, h_max, colors=color)
        plt.vlines(hpos-delta, h_min, h_max, colors=color)
        hpos += Nclusters
    if save==True:
        plt.savefig(file_name, dpi = 600)
    
def plot_tsne(cm, STm, session, Nclusters, title=None,file_name=None, save=False):
    '''
        Plot the dimensionality reduction in 2D. Use tSNE coordinates
        and kMeans label to color.
        Inputs:
        cm: Contingency matrix
        Outputs:
    '''
    # Pallete to color the points
    palette   = np.array(sns.color_palette("hls", Nclusters))
    idx       = cm[:,0].astype(bool)
    idx2      = np.invert(idx)
    plt.subplot(1,2,1)
    plt.title(title)
    plt.scatter(cm[idx,1], cm[idx,2], c=palette[cm[idx,3].astype(int)])
    plt.scatter(cm[idx2,1], cm[idx2,2], c='gray', alpha = 0.4, s=5)
    plt.subplot(1,2,2)
    p_idx = np.arange(cm.shape[0])[cm[:,0].astype(bool)]
    for i in range(Nclusters):
        idx = (cm[p_idx,3]==i)
        Cmean = STm[p_idx[idx], :].mean(axis = 0)
        plt.plot(session['fc'], Cmean, color = palette[i])
        plt.ylabel('Coherence')
        plt.xlabel('Frequency [Hz]')
    if save==True:
        plt.savefig(file_name, dpi = 600)

def cluster_analysis(X, Nclusters, band_width, nsamples, use_samples=False, thr=None):
    '''
        Performs the clustering analysis in the coherence matrix.
        Inputs:
        X: Average coherecen matrix across observations (Trials x Time) points.
        Nclusters: Number of clusters
        band_width: Distance between frequency points (index distance not in Hz)
        nsamples: Total number of samples (pairs)
        use_samples: Whether to use samples or not. If True, will only use samples,
        use_clustering: Whether to use tsne or ntf
        i.e., pairs with silhouette value grater than or equal thr.
        thr: Threshold used to select samples in the silhouette analysis.
        use_mean: If True will use the mean silhouette value as threshold.
        file_name: Figure file name
        Outputs:
        
    '''
    p_idx = np.arange(nsamples)
    # If use_samples true, select pairs with silhouette socore above thr
    if use_samples == True:
        # Return the pairs indexes above, the indexes below thr and the k-means labels
        sn = samples_idx(X, Nclusters, nsamples, thr)
    else:
        sn = True * np.ones(nsamples)
    kmeans = KMeans(n_clusters=Nclusters, n_init=100, max_iter = 600).fit(X)
    ykm = kmeans.predict(X)
    tsne = TSNE(n_jobs=40, perplexity = 30)
    Y = tsne.fit_transform(X)
    '''
    Matrix with information about clusterirng analysis
    First column: Pair's index
    Second column: First dimension of tSNE
    Third column: Second dimension of tSNE
    Fourth column: kMeans labels for each pair
    The further dimensions are binary arrays indicanting whether or not pair i is in cluster n, where
    n >= 4th column of the matrix.
    '''
    cluster_analysis_matrix = np.zeros([len(sn), 1+Y.shape[1]+1+Nclusters])
    cluster_analysis_matrix[:, 0] = sn
    cluster_analysis_matrix[:, 1] = Y[:,0]
    cluster_analysis_matrix[:, 2] = Y[:,1]
    cluster_analysis_matrix[:, 3] = ykm.astype(int)
    for i in range(Nclusters):
        cluster_analysis_matrix[:, 4+i] = ykm==i
    return cluster_analysis_matrix

def find_links(X, cma, Nclusters):
    '''
        Find the links that compose a cluster in each frequency band. Ordered by the cluster with highst peak in the 
        frequency range considered.
        Inputs:
        cma: The matrix with the cluster analysis data,
        First column: Pair's index
        Second column: First dimension of tSNE
        Third column: Second dimension of tSNE
        Fourth column: kMeans labels for each pair
        The further dimensions are binary arrays indicanting whether or not pair i is in cluster n, where
        n >= 4th column of the matrix.
        Outputs:
        links: Dicitionary in which each cell is an array with pairs of channels, the cells are ordered
        from the cluster with biggest to lowest coherence peak in the range considered.
    '''
    links = {}
    C_max = []
    index_track = {}
    p_idx = np.arange(cma.shape[0])[cma[:,0].astype(bool)]
    for i in range(Nclusters):
        idx = (cma[p_idx,3]==i)
        Cmean = X[p_idx[idx], :].mean(axis = 0)
        C_max.append(Cmean.max())
        index_track[i] = C_max[i]
    aux = 0
    for i in sorted(index_track, key=index_track.get)[::-1]:
        idx = (cma[p_idx,3]==i)
        links[aux] = p_idx[idx]
        aux += 1
    return links    

def rank(matrix, x):
    '''
        Computes rank for every element of the matrix in respect to a quantitie x.
        Inputs:
        matrix: Matrix with values.
        x: Quantitie in which the matrix will be ranked.
        Outputs: rank
    '''
    return np.dot(matrix, x)