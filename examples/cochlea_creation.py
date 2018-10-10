# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join('.', 'cochlea', 'base')))
# sys.path.insert(0, os.path.abspath(os.path.join('.', 'cochlea', 'spikes')))
import numpy as np
import matplotlib
from simplecochlea import *
matplotlib.use('TkAgg')


fs, fmin, fmax, freq_scale, n_channels = 44100, 200, 8000, 'erbscale', 100
comp_factor, comp_gain = 0.3, 1.5
tau, v_thresh, v_spike = np.linspace(0.001, 0.0004, n_channels), np.linspace(0.3, 0.17, n_channels), 0.5


cochlea_simp = Cochlea(n_channels, fs, fmin, fmax, freq_scale, comp_factor=comp_factor, comp_gain=comp_gain,
                       lif_tau=tau, lif_v_thresh=v_thresh, lif_v_spike=v_spike)

print(cochlea_simp)

spikelist_sin = cochlea_simp.process_test_signal('sin', channel_pos=50)
spikelist_sin.plot()
