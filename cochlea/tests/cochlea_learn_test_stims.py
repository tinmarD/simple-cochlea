import numpy as np
import os
from freq_analysis_utils import *
from scipy.io import wavfile
import seaborn as sns
import pandas as pd
import cochlea_utils
from cochlea import *
from STDP import *
sns.set()
sns.set_context('paper')


cochlea_dir = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_models'
cochlea_name = 'cochlea_model_adapt_thresh_FS_inhib_1k_8kHz_lowomega.p'
stim_learn_path = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\speech\Tim_JAST.wav'
stim_test_path = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\speech\Tim_comment_JAST_short.wav'

# Import  cochlea
cochlea = load_cochlea(cochlea_dir, cochlea_name)
print(cochlea)

fs, stim_learn = wavfile.read(stim_learn_path)
stim_learn = stim_learn[:, 0] if stim_learn.ndim == 2 else stim_learn
stim_learn = cochlea_utils.normalize_vector(stim_learn)
fs, stim_test = wavfile.read(stim_test_path)
stim_test = stim_test[:, 0] if stim_test.ndim == 2 else stim_test
stim_test = cochlea_utils.normalize_vector(stim_test)


## JAST2 params
kMax = 1
M, P = 1024, 1024
N, W, T_i, T_f, T_firing = 32, 16, 5, 13, 11
refract_period_s = 0.002
dT, d_n_swap = 0.1, 1
learn_ratio_threshold = 2
active_chan_threshold = 1

spikelist_in_learn = cochlea.process_input(stim_learn)
spikelist_out_learn, weights, _ = STDP_v2(spikelist_in_learn, fs, N, W, M, P, dT=0.5, n_swap_i=[], d_n_swap=[], min_n_swap=1, T_i=T_i,
                                        T_f=T_f, T_firing=T_firing, same_chan_in_buffer_max=kMax, full_mode=0)

spikelist_in_test = cochlea.process_input(stim_test)
spikelist_out_test, _, _ = STDP_v2(spikelist_in_test, fs, N, W, M, P, dT=0.5, n_swap_i=[], d_n_swap=[], min_n_swap=1, T_i=T_i,
                                   T_f=T_f, T_firing=T_firing, same_chan_in_buffer_max=kMax, full_mode=0,
                                   weight_init=weights, freeze_weight=1)


dual_spikelist_plot(spikelist_in_learn, spikelist_out_learn)
dual_spikelist_plot(spikelist_in_test, spikelist_out_test)

