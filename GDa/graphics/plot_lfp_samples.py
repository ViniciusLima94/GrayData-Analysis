#####################################################################################################
# Class with methods to plot data
#####################################################################################################
import numpy             as np
import matplotlib.pyplot as plt

def plot_lfp_samples(data, trials, channels, colors = None):

	count    = 1
	plt.figure(figsize=(7,4))
	for i in range( len(trials) ):
		for j in range( len(channels) ):
			plt.subplot(len(trials), len(channels), count)
			plt.subplots_adjust(wspace=.1, hspace=.1)
			plt.box(False)
			plt.plot(data[trials[i],channels[j],1000:2000], c = colors[j])
			plt.xticks([])
			plt.yticks([])
			if colors[j] == colors[0]:
				plt.ylabel('Trial ' + str(trials[i]))
			if trials[i] == trials[0]:
				plt.title('Ch. ' + str(channels[j]))
			count += 1
	plt.show()

