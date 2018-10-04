## Try to analyse why a stim is not recognized by JAST, compare different cochlea models, ...
import numpy as np
import os
from freq_analysis_utils import *
from scipy.io import wavfile
import seaborn as sns
import pandas as pd
from cochlea import *
from STDP import *
sns.set()
sns.set_context('paper')


database_root_path = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases\Database_10_50ms_1'
i_stim = 17
M = 1024

cochlea_names = os.listdir(os.path.join(database_root_path, 'SpikeLists'))
n_cochleas = len(cochlea_names)

if n_cochleas == 0:
    raise ValueError('No spike lists')

cochlea_dir, cochlea_dir_in, cochlea_dir_out = [], [], []
for i, cochlea_name_i in enumerate(cochlea_names):
    cochlea_dir.append(os.path.join(database_root_path, 'SpikeLists', cochlea_name_i))
    cochlea_dir_in.append(os.path.join(cochlea_dir[i], 'IN'))
    cochlea_dir_out.append(os.path.join(cochlea_dir[i], os.listdir(cochlea_dir[i])[1]))

spikelist_in_name = os.listdir(cochlea_dir_in[0])[i_stim]
spikelist_out_name = os.listdir(cochlea_dir_out[0])[i_stim]

# Get stim
stim_dir = os.path.join(database_root_path, 'STIM')
fs, stim_i = wavfile.read(os.path.join(stim_dir, os.listdir(stim_dir)[i_stim]))


for i in range(n_cochleas):
    spikelist_in_i = import_spikelist_from_mat(os.path.join(cochlea_dir_in[i], spikelist_in_name), M)
    spikelist_out_i = import_spikelist_from_mat(os.path.join(cochlea_dir_out[i], spikelist_out_name), M)
    dual_spikelist_plot(spikelist_in_i, spikelist_out_i)


# # other cochlea:
# cochlea_dir = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_models'
# cochlea_ext_name = 'cochlea_model_adapt_thresh_FS_inhib_1k_8kHz_lowomega.p'
# cochlea_ext = load_cochlea(cochlea_dir, cochlea_ext_name)
# #
# # # JAST2 params
# kMax = 1
# M, P = 1024, 1024
# N, W, T_i, T_f, T_firing = 32, 16, 5, 13, 11
# refract_period_s = 0.002
# dT, d_n_swap = 0.1, 1
# learn_ratio_threshold = 2
# active_chan_threshold = 1
#
# spikelist_in_ext = cochlea_ext.process_input(stim_i)
# spikelist_out_ext, _, _ = STDP_v2(spikelist_in_ext, fs, N, W, M, P, dT=0.5, n_swap_i=[], d_n_swap=[], min_n_swap=1, T_i=T_i,
#                                   T_f=T_f, T_firing=T_firing, same_chan_in_buffer_max=kMax, full_mode=0)
# dual_spikelist_plot(spikelist_in_ext, spikelist_out_ext)
#
