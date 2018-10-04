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
stim_path = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\speech\Tim_comment_JAST.wav'

# Import  cochlea
cochlea = load_cochlea(cochlea_dir, cochlea_name)
print(cochlea)

fs, stim = wavfile.read(stim_path)
if stim.ndim == 2:
    stim = stim[:, 0]
stim = cochlea_utils.normalize_vector(stim)

## JAST2 params
kMax = 1
M, P = 1024, 1024
N, W, T_i, T_f, T_firing = 32, 16, 5, 13, 11
refract_period_s = 0.002
dT, d_n_swap = 0.1, 1
learn_ratio_threshold = 2
active_chan_threshold = 1

spikelist_in_ext = cochlea.process_input(stim)
spikelist_out_ext, _, _ = STDP_v2(spikelist_in_ext, fs, N, W, M, P, dT=0.5, n_swap_i=[], d_n_swap=[], min_n_swap=1, T_i=T_i,
                                  T_f=T_f, T_firing=T_firing, same_chan_in_buffer_max=kMax, full_mode=0)
dual_spikelist_plot(spikelist_in_ext, spikelist_out_ext)

