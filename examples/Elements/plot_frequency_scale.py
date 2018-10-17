"""
================
Frequency Scale
================

Illustration of the different frequency scales

"""

from simplecochlea.utils import utils_cochlea


#############################################
# Suppose we want a cochlea whose frequency domain range from 20 Hz to 20000 Hz with 100 filters
fmin, fmax = 20, 20000
n_filters = 100
fs = 44100

#############################################
# First let's use a linear scale
# The utils_cochlea.linearscale returns both the filters coefficients and the center frequencies.

wn, cf = utils_cochlea.linearscale(fs, fmin, fmax, n_filters)




