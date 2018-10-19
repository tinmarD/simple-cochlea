"""
=======================================
   LIB Bank with lateral Inhibition
=======================================

Analysis of the implementation of the neural inhibition described in [1]. These different types of connection are
supposed to produce contrast enhancement, i.e. for the cochlea it can lead to a sharpening of its frequency sensitivity.

Four types of inhibition are described :
 * Forward-subtractive inhibition
 * Backward-subtractive inhibition
 * Forward-shunting inhibition
 * Backward-shunting inhibition


References
----------

.. [1] Gershon G. Furman and Lawrence S. Frishkopf. Model of Neural Inhibition in the Mammalian Cochlea.
       The Journal of the Acoustical Society of America 1964 36:11, 2194-2201

"""

import numpy as np
from simplecochlea import Cochlea
from simplecochlea import generate_signals



#######################
# For testing the inhibition, we will use a signal composed of 3 sinusoids close in frequency
fs = 44100
test_sig = generate_signals.generate_sinus(fs, f_sin=[1500, 2000, 2100], t_offset=[0.15, 0.1, 0.2], t_max=1)
generate_signals.plot_signal(test_sig, fs)


######################
# Construct the cochlea :
fmin, fmax, freq_scale, n_channels = 200, 8000, 'erbscale', 1000
comp_factor, comp_gain = 0.3, 1.5
tau, v_thresh, v_spike = np.linspace(0.001, 0.0004, n_channels), np.linspace(0.3, 0.17, n_channels), 0.5

cochlea = Cochlea(n_channels, fs, fmin, fmax, freq_scale, comp_factor=comp_factor, comp_gain=comp_gain,
                  lif_tau=tau, lif_v_thresh=v_thresh, lif_v_spike=v_spike)




