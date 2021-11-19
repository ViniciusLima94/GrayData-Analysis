import numpy as np
import os
import sys
from GDa.temporal_network import temporal_network
import matplotlib.pyplot as plt

idx = int(sys.argv[-1])

###########################################################################
# Get sessions names
###########################################################################
sessions = np.array(os.listdir("Results/lucy/"))

###########################################################################
# Plotting
###########################################################################
s = sessions[idx]

net = temporal_network(coh_file='coh_k_0.3_morlet.nc',
                       coh_sig_file='coh_k_0.3_morlet_surr.nc',
                       date=s, trial_type=[1], behavioral_response=[1])

# Average over the same rois
out = net.super_tensor.groupby("roi").mean("roi")
# .interp(# {"freqs": np.linspace(3, 75, 30)}, "quadratic")

# Average over time
out.mean("times").mean("trials").plot(x="freqs", hue="roi", color="b", lw=1)
plt.xlim(out.freqs[0], out.freqs[-1])
plt.legend([])
plt.title(f"{s}")
plt.savefig(f"figures/avg_coh/avg_coh_{s}.png", dpi=150)
plt.close()
del net, out
