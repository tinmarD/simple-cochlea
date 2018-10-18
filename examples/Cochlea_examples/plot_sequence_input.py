"""
==========================
Repeating pattern sequence
==========================

Run the cochlea on a sequence composed of 1 repeating pattern
This pattern of 50ms appears 10 times and each repetition is separated by a noise segment (i.e. a non-repeating pattern)

"""

import os
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from scipy.io import wavfile
import seaborn as sns
from simplecochlea import Cochlea
import simplecochlea
sns.set_context('paper')


#############################
# Load the file
root_dirpath = os.path.dirname(simplecochlea.__file__)
sample_data_dir = os.path.join(root_dirpath, 'sample_data')
fs, sequence = wavfile.read(os.path.join(sample_data_dir, 'sample_sequence_10_50ms_1.wav'))

#############################
# Create the cochlea
fmin, fmax, freq_scale, n_channels = 200, 8000, 'erbscale', 1000
comp_factor, comp_gain = 0.3, 1.5
tau, v_thresh, v_spike = np.linspace(0.001, 0.0004, n_channels), np.linspace(0.3, 0.17, n_channels), 0.5

cochlea = Cochlea(n_channels, fs, fmin, fmax, freq_scale, comp_factor=comp_factor, comp_gain=comp_gain,
                       lif_tau=tau, lif_v_thresh=v_thresh, lif_v_spike=v_spike)

#############################
# Run the sequence through the cochlea
spikelist_seq = cochlea.process_input(sequence)

#############################
# Plot the spikelist
spikelist_seq.plot()

#############################
# We know the repeating pattern is repeating every 50ms, the sequence starts with a noise segment and in total, there
# are 20 segments (10 time the pattern and 10 interleaved noise segments).
# Thus we can set the pattern_id of the spikes in the output spikelist, with the set_pattern_id_from_time_limits method.
chunk_duration, n_chunks = 0.050, 20
t_start = np.arange(0, chunk_duration*n_chunks, chunk_duration)
t_end = t_start + chunk_duration
pattern_id = [1, 2] * 10
pattern_names = {1: 'Noise', 2: 'Pattern'}

spikelist_seq.set_pattern_id_from_time_limits(t_start, t_end, pattern_id, pattern_names)

#############################
# Replot the spikelist to see the results :
spikelist_seq.plot()









