import numpy                         as     np
import matplotlib.pyplot             as     plt
from   GDa.spectral_analysis         import spectral

#############################################################################################
# Testing with ZAP signal
#############################################################################################
class ZAP():

	def __init__(self, A, ttstart, ttstop, Fstart, Fstop):
		self.A       = A
		self.ttstart = ttstart
		self.ttstop  = ttstop
		self.Fstart  = Fstart
		self.Fstop   = Fstop


	def evaluate(self, t):
		Fzap = (self.Fstart + (self.Fstop-self.Fstart) * (t - self.ttstart) / (self.ttstop - self.ttstart))
		Szap = self.A * np.sin(2 * np.pi * (Fzap - self.Fstart) * (t - self.ttstart) / 2)

		return Fzap, Szap

# Time axis 
dt = 1e-3 
t  = np.arange(0, 620, dt)
F, Z  = ZAP(10.0, 2.0, 620.0, 0.001, 20.001).evaluate(t)


spec = spectral()
f    = spec.compute_freq(len(Z), 1/dt)
P    = spec.instantaneous_power(signal = Z, fs = fs, f_low = 3, f_high = 6, n_jobs = 10)

plt.subplot2grid((2,2),(0,0),colspan=2)
plt.plot(t, Z, lw=.3)
plt.xlabel('Time (s)')
plt.ylabel('ZAP')
plt.subplot2grid((2,2),(1,0))
plt.plot(t, F)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.subplot2grid((2,2),(1,1))
plt.plot(t, P)
plt.ylabel('Instantaneous Power')
plt.xlabel('Time (s)')
plt.title('3-6 Hz band')
plt.tight_layout()
plt.show()