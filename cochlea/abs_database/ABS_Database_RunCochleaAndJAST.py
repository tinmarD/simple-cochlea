from cochlea import *
import cochlea_utils
from STDP import *
import pandas as pd
import os
import re
import tqdm
import multiprocessing as mp
import datetime


cochlea_dirpath = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_models'
cochlea_name = 'cochlea_model_adapt_thresh_FS_inhib_1k_8kHz_lowomega.p'
n_channels = 1024

wav_stim_dir = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_ExperienceManu\EXP_stim_set2\stimuli'
res_dir = r'C:\Users\deudon\Desktop\M4\_Results\ABS_Exp_Results'
stim_filename = r'stim_set_#2_carac.csv'

# Create result directory with cochlea name
cochlea_res_dir = os.path.join(res_dir, cochlea_name[:-2])
if not os.path.exists(cochlea_res_dir):
    os.mkdir(cochlea_res_dir)

# JAST2 parameters
kMax = 1
M, P = 1024, 1024
N, W, T_i, T_f, T_firing = 32, 16, 5, 13, 11
refract_period_s = 0.002
dT, d_n_swap = 0.1, 1
learn_ratio_threshold = 2
active_chan_threshold = 1

# Load cochlea
cochlea = load_cochlea(cochlea_dirpath, cochlea_name)
print(cochlea)

# Load csv file with stimuli informations
df_stim = pd.read_csv(os.path.join(res_dir, stim_filename), index_col=0, sep=';')

learn_ratio_prop_all = np.zeros(df_stim.shape[0])
active_chan_prop_all = np.zeros(df_stim.shape[0])
if 'JAST_learning_ratio' not in df_stim.keys():
    df_stim['JAST_learning_ratio'] = 0.0
if 'JAST_active_chan_ratio' not in df_stim.keys():
    df_stim['JAST_active_chan_ratio'] = 0.0
if 'prop_target_noise' not in df_stim.keys():
    df_stim['prop_target_noise'] = 0.0


for i in tqdm.tqdm(range(df_stim.shape[0])):
    stim_name_i = df_stim.stim_name[i]
    # Read stim (wavfile)
    fs_i, wav_stim_i = wavfile.read(os.path.join(wav_stim_dir, stim_name_i))
    if wav_stim_i.ndim > 1:
        wav_stim_i = wav_stim_i[:, 0]
    if not fs_i == cochlea.fs:
        raise ValueError('Sampling rate of the stimulus is not equal to the sampling rate of the cochlea')
    wav_stim_i = cochlea_utils.normalize_vector(wav_stim_i)
    # Run cochlea
    spikelist_in_i = cochlea.process_input(wav_stim_i)
    target_pos_sec_i, snip_duration_i = df_stim.target_pos_sec[i], df_stim.snip_duration[i]
    if type(target_pos_sec_i) is str:
        target_pos_sec_i = re.sub('^\[ +', '[', target_pos_sec_i)
        target_pos_sec_i = re.sub(' +]$', ']', target_pos_sec_i)
        target_pos_sec_i = re.sub(' +', ',', target_pos_sec_i)
        target_pos_sec_i = eval(target_pos_sec_i)

    target_pos_sec_i = np.atleast_1d(np.array(target_pos_sec_i).squeeze())
    t_start = np.sort(np.hstack([0, target_pos_sec_i, target_pos_sec_i-snip_duration_i]))
    t_end = np.sort(np.hstack([target_pos_sec_i-snip_duration_i, target_pos_sec_i, wav_stim_i.size / fs_i]))
    pattern_id = 2*np.ones(t_start.size, dtype=int)
    pattern_id[1::2] = 1
    pattern_dict = {1: 'Target', 2: 'Noise'}
    spikelist_in_i.set_pattern_id_from_time_limits(t_start, t_end, pattern_id, pattern_dict)

    # Run JAST
    spikelist_out_i, _, _ = STDP_v2(spikelist_in_i, fs_i, N, W, M, P, dT=0.5, n_swap_i=[], d_n_swap=[], min_n_swap=1, T_i=T_i,
                                    T_f=T_f, T_firing=T_firing, same_chan_in_buffer_max=kMax, full_mode=0)
    # Compute scores
    prop_target_noise = df_stim.snip_duration[i] * df_stim.n_rep[i] / spikelist_in_i.tmax

    learn_ratio_prop_all[i] = np.sum(spikelist_out_i.pattern_id == 1) / np.sum(spikelist_out_i.pattern_id == 2) / prop_target_noise
    active_chan_prop_all[i] = np.unique(spikelist_out_i.channel[spikelist_out_i.pattern_id == 1]).size /\
                              np.unique(spikelist_out_i.channel[spikelist_out_i.pattern_id == 2]).size / prop_target_noise

    df_stim['JAST_learning_ratio'][i] = learn_ratio_prop_all[i]
    df_stim['JAST_active_chan_ratio'][i] = active_chan_prop_all[i]
    df_stim['prop_target_noise'][i] = prop_target_noise

datetime_str = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%M")
df_stim.to_csv(os.path.join(cochlea_res_dir, '{}_{}_out.csv'.format(stim_filename[:-4], datetime_str)), sep=';')

