"""
===============================================
  Process a single channel of the cochlea
===============================================

This example show how to run a signal through a specific channel of the cochlea

"""

import os
import numpy as np
from scipy.io import wavfile
import seaborn as sns
from simplecochlea import Cochlea, generate_signals
sns.set_context('paper')


############################
# Create the cochlea
fs, fmin, fmax, freq_scale, n_channels = 44100, 200, 8000, 'erbscale', 100
comp_factor, comp_gain = 0.3, 1.5
tau, v_thresh, v_spike = np.linspace(0.001, 0.0004, n_channels), 0, 0.5
# Adaptive threshold parameters
tau_j, alpha_j = np.array([0.010, 0.200]), np.array([0.010, 0.000002])
omega = np.linspace(0.15, 0.2, n_channels)

cochlea_adaptive_thresh = Cochlea(n_channels, fs, fmin, fmax, freq_scale, comp_factor=comp_factor, comp_gain=comp_gain,
                                  lif_tau=tau, lif_v_thresh=v_thresh, lif_v_spike=v_spike, tau_j=tau_j, alpha_j=alpha_j,
                                  omega=omega)

##############################
# Generate a sinusoidal signal
x_sin = generate_signals.generate_sinus(fs, 1800, t_offset=0, t_max=0.25, amplitude=1)

##########################################################
# Pass the input signal through one channel of the cochlea
# The `plot_channel_evolution` method allows to visualize the differents steps
#  of the cochlea processing
cochlea_adaptive_thresh.plot_channel_evolution(x_sin, 30)


