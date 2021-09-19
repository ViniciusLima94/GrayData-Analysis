import numpy as np 
import cupy  as cp
from   bursting import *
import matplotlib.pyplot as plt
import time

def timer(function, **kwargs):
    start = time.time()
    function(**kwargs)
    end   = time.time()
    return (end-start)

#  masked_find_activation_sequences(spike_train, mask, dt=None, drop_edges=False, target="cpu")

if __name__ == '__main__':

    def _gen_data(n):
        return np.random.rand(n)>0.5

    # Different data sizes 
    n = np.linspace(10000, 540*1176, 50, dtype=int)

    t_cpu = []
    t_gpu = []

    #  for i in range(n.shape[0]):
    #      t_cpu += [timer(find_activation_sequences,spike_train=_gen_data(n[i]), dt=None, target="cpu")]
    #      t_gpu += [timer(find_activation_sequences,spike_train=cp.array(_gen_data(n[i])), dt=None, target="gpu")]
    
    #  def masked_find_activation_sequences(spike_train, mask, dt=None, drop_edges=False, target="cpu"):
    #  for i in range(n.shape[0]):
    #      x     = _gen_data(n[i])
    #      mask  = np.ones_like(x); mask[n[i]//2:]=0
    #      t_cpu += [timer(masked_find_activation_sequences,spike_train=x, mask=mask,dt=None, drop_edges=True, target="cpu")]
    #      t_gpu += [timer(masked_find_activation_sequences,spike_train=cp.array(x), mask=cp.array(mask),drop_edges=True,dt=None, target="gpu")]

    #  plt.plot(n,t_cpu, label='cpu')
    #  plt.plot(n,t_gpu, label='gpu')
    #  plt.legend()
    #  plt.show()

    spike_train = np.random.rand(1176,540,200)>0.5
    mask        = np.ones((540,200))
    mask[:,100:]=0
    mask = mask.astype(bool)
    tensor_find_activation_sequences(spike_train, mask, dt=None, drop_edges=False, target="cpu", n_jobs=-1)
