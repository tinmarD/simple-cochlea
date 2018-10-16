"""
================
Cochlea Creation 
================

Create a simple cochlea model and test it on a sinusoidal input signal

"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from simplecochlea import Cochlea

############################
# Create the cochlea
fs, fmin, fmax, freq_scale, n_channels = 44100, 200, 8000, 'erbscale', 1000
comp_factor, comp_gain = 0.3, 1.5
tau, v_thresh, v_spike = np.linspace(0.001, 0.0004, n_channels), np.linspace(0.3, 0.17, n_channels), 0.5

cochlea_simp = Cochlea(n_channels, fs, fmin, fmax, freq_scale, comp_factor=comp_factor, comp_gain=comp_gain,
                       lif_tau=tau, lif_v_thresh=v_thresh, lif_v_spike=v_spike)


###############################
# Print the description
print(cochlea_simp)

###############################
# Process a sin input signal and plot the output spikelist
spikelist_sin = cochlea_simp.process_test_signal('sin', channel_pos=50, f_sin=400)
spikelist_sin.plot()

###############################
# Save the cochlea with pickle
cochlea_simp.save()

