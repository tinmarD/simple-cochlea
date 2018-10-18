"""
================
Frequency Scale
================

Illustration of the different frequency scales

"""

from simplecochlea.utils import utils_cochlea
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context('paper')


#############################################
# Suppose we want a cochlea whose frequency domain range from 20 Hz to 20000 Hz with 100 hair cells
# Each hair cell can be modeled as a band-pass filter. Each one selecting a certain frequency range.
# An important parameter is the way these band-pass filters are organized and cover the whole frequency range
# of hearing.
fmin, fmax = 20, 20000
n_filters = 100
fs = 44100

#############################################
# A unrealistic but simple way to organize the band-pass filters is to use a *linear scale*.
# The :func:`utils_cochlea.linearscale` returns both the filters cutoff and center frequencies
wn_lin, cf_lin = utils_cochlea.linearscale(fs, fmin, fmax, n_filters)

#############################################
# A more realistic solution to model the tonotopy of the cochlea is to use the ERB scale (Equivalent Rectangular
# Bandwitdh) :
wn_erb, cf_erb = utils_cochlea.erbscale(fs, fmin, fmax, n_filters)

#############################################
# Let's plot the evolution of the center frequencies for both scales :
f = plt.figure()
ax = f.add_subplot(111)
ax.stem(cf_lin, markerfmt='C0o')
ax.stem(cf_erb, markerfmt='C1o')
ax.set(xlabel='Filter Number', ylabel='Frequency', title='Evolution of the Center Frequency of Bandpass filters')
ax.legend(['Linear Scale', 'ERB Scale'])


