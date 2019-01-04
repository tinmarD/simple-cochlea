"""
=======================================
   LIB Bank with lateral Inhibition
=======================================

Analysis of the implementation of the neural inhibition described in [1]. These different types of connection are
supposed to produce contrast enhancement, i.e. for the cochlea it can lead to a sharpening of its frequency sensitivity.

We selected one model of lateral inhibition : the forward-shunting inhibition

References
----------

.. [1] Gershon G. Furman and Lawrence S. Frishkopf. Model of Neural Inhibition in the Mammalian Cochlea.
       The Journal of the Acoustical Society of America 1964 36:11, 2194-2201

"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from simplecochlea import Cochlea
from simplecochlea import generate_signals
sns.set()
sns.set_context('paper')

#######################
# For testing the inhibition, we will use a signal composed of 3 sinusoids close in frequency
fs = 44100
test_sig = generate_signals.generate_sinus(fs, f_sin=[1500, 2000, 2100], t_offset=[0.15, 0.1, 0.2], t_max=1)
generate_signals.plot_signal(test_sig, fs)


##########################################
# Construct a cochlea without inhibition :
fmin, fmax, freq_scale, n_channels = 200, 8000, 'erbscale', 1000
comp_factor, comp_gain = 0.3, 1.5
tau, v_thresh, v_spike = np.linspace(0.001, 0.0004, n_channels), np.linspace(0.3, 0.17, n_channels), 0.5

cochlea = Cochlea(n_channels, fs, fmin, fmax, freq_scale, comp_factor=comp_factor, comp_gain=comp_gain,
                  lif_tau=tau, lif_v_thresh=v_thresh, lif_v_spike=v_spike)

############################################
# Construct a second cochlea with inhibition
# We will use a forward-shunting type of inhibition
############################################
# We define an inhibition vector which gives the strenght of the inhibition of channel i related with its neighbours
N, inhib_sum = 50, 1
inhib_vect = signal.gaussian(2*N+1, std=15)
inhib_vect[N] = -2
inhib_vect_norm = inhib_sum * inhib_vect / inhib_vect.sum()
############################################
# Let's plot the normalized inhibition vector
f = plt.figure()
plt.plot(np.arange(-N, N+1), inhib_vect_norm)

cochlea_with_inhib = Cochlea(n_channels, fs, fmin, fmax, freq_scale, comp_factor=comp_factor, comp_gain=comp_gain,
                             lif_tau=tau, lif_v_thresh=v_thresh, lif_v_spike=v_spike, inhib_vect=inhib_vect_norm)


############################################
# Run the test signal through the 2 cochleas
spikelist_sin, _ = cochlea.process_input(test_sig)
spikelist_sin.plot()

############################################
# With inhibition :
spikelist_sin_inhib, _ = cochlea_with_inhib.process_input(test_sig)
spikelist_sin_inhib.plot()

