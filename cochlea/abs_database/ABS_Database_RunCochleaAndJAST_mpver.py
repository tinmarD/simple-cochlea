from cochlea import *
import cochlea_utils
from STDP import *
import pandas as pd
import os
import re
import tqdm


cochlea_dirpath = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_models'
cochlea_name = 'cochlea_model_inhib_160418_1324.p'
n_channels = 1024

wav_stim_dir = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_ExperienceManu\EXP_stim_set1\stimuli'
res_dir = r'C:\Users\deudon\Desktop\M4\_Results\ABS_Exp_Results'
stim_filename = r'stim_set_#1_carac.csv'
abs_spikelist_dir = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_ExperienceManu_Spikelists'

stim_base_num = int(re.search('\d+', re.search('set\d+', wav_stim_dir)[0])[0])
if not int(re.search('\d+', re.search('_#\d+', stim_filename)[0])[0]) == stim_base_num:
    raise ValueError('Cannot dertermine stim_base_num')

# Create output directory
in_spikelist_dir = os.path.join(abs_spikelist_dir, 'set_#{}_spikelist_IN_{}'.format(stim_base_num, cochlea_name[:-2]))
out_spikelist_dir = os.path.join(abs_spikelist_dir, 'set_#{}_spikelist_OUT_{}'.format(stim_base_num, cochlea_name[:-2]))
if os.path.isdir(in_spikelist_dir):
    raise ValueError('Directory already exist : {}'.format(in_spikelist_dir))
else:
    os.mkdir(in_spikelist_dir)
os.mkdir(out_spikelist_dir)

# JAST2 parameters
kMax = 1
M, P = 1024, 1024
N, W, T_i, T_f, T_firing = 32, 16, 5, 13, 11
refract_period_s = 0.002
dT, d_n_swap = 0.1, 1

# Load cochlea
cochlea = load_cochlea(cochlea_dirpath, cochlea_name)
print(cochlea)

# Load csv file with stimuli informations
df_stim = pd.read_csv(os.path.join(res_dir, stim_filename), index_col=0, sep=';')
learn_ratio_prop_all = np.zeros(df_stim.shape[0])
active_chan_prop_all = np.zeros(df_stim.shape[0])

n_stim = df_stim.shape[0]
# n_stim = 5

learn_ratio_prop_all, active_chan_prop_all, prop_target_noise, n_spikes_target = np.zeros((4, n_stim))
if 'JAST_learning_ratio' not in df_stim.keys():
    df_stim['JAST_learning_ratio'] = 0.0
if 'JAST_active_chan_ratio' not in df_stim.keys():
    df_stim['JAST_active_chan_ratio'] = 0.0
if 'prop_target_noise' not in df_stim.keys():
    df_stim['prop_target_noise'] = 0.0
if 'n_spikes_target_in' not in df_stim.keys():
    df_stim['n_spikes_target_in'] = 0.0

target_name, wav_stim = [], []
for i in range(n_stim):
    stim_name_i = df_stim.stim_name[i]
    # Read stim (wavfile)
    fs_i, wav_stim_i = wavfile.read(os.path.join(wav_stim_dir, stim_name_i))
    if wav_stim_i.ndim == 2:
        wav_stim_i = wav_stim_i[:, 0]
    wav_stim_i = cochlea_utils.normalize_vector(wav_stim_i)
    wav_stim.append(wav_stim_i)
    target_name.append(stim_name_i.split('_')[1])
    if not fs_i == cochlea.fs:
        raise ValueError('Sampling rate of the stimulus is not equal to the sampling rate of the cochlea')

pool = mp.Pool(processes=6)
output = [pool.apply_async(cochlea.process_input, args=(wav_stim_i,)) for wav_stim_i in wav_stim]
spike_list_all = [p.get() for p in output]

for i, spikelist_i in enumerate(spike_list_all):
    spikelist_i.export(in_spikelist_dir, df_stim.stim_name[i])

for i in tqdm.tqdm(range(n_stim)):
    spikelist_in_i = spike_list_all[i]
    target_pos_sec_i, snip_duration_i = df_stim.target_pos_sec[i], df_stim.snip_duration[i]
    if type(target_pos_sec_i) is str:
        target_pos_sec_i = re.sub('^\[ +', '[', target_pos_sec_i)
        target_pos_sec_i = re.sub(' +]$', ']', target_pos_sec_i)
        target_pos_sec_i = re.sub(' +', ',', target_pos_sec_i)
        target_pos_sec_i = eval(target_pos_sec_i)

    target_pos_sec_i = np.atleast_1d(np.array(target_pos_sec_i).squeeze())
    t_start = np.sort(np.hstack([0, target_pos_sec_i, target_pos_sec_i-snip_duration_i]))
    t_end = np.sort(np.hstack([target_pos_sec_i-snip_duration_i, target_pos_sec_i, wav_stim[i].size / fs_i]))
    pattern_id = 2*np.ones(t_start.size, dtype=int)
    pattern_id[1::2] = 1
    pattern_dict = {1: 'Target', 2: 'Noise'}
    spikelist_in_i.set_pattern_id_from_time_limits(t_start, t_end, pattern_id, pattern_dict)
    n_spikes_target[i] = sum(spikelist_in_i.pattern_id == 1)

    # Run JAST
    spikelist_out_i, _, _ = STDP_v2(spikelist_in_i, fs_i, N, W, M, P, dT=0.5, n_swap_i=[], d_n_swap=[], min_n_swap=1, T_i=T_i,
                                    T_f=T_f, T_firing=T_firing, same_chan_in_buffer_max=kMax, full_mode=0)

    spikelist_out_i.export(out_spikelist_dir, df_stim.stim_name[i])

    # Compute scores
    prop_target_noise[i] = df_stim.snip_duration[i] * df_stim.n_rep[i] / spikelist_in_i.tmax

    learn_ratio_prop_all[i] = np.sum(spikelist_out_i.pattern_id == 1) / np.sum(spikelist_out_i.pattern_id == 2)
    active_chan_prop_all[i] = np.unique(spikelist_out_i.channel[spikelist_out_i.pattern_id == 1]).size /\
                              np.unique(spikelist_out_i.channel[spikelist_out_i.pattern_id == 2]).size

df_stim['JAST_learning_ratio'][:n_stim] = learn_ratio_prop_all
df_stim['JAST_active_chan_ratio'][:n_stim] = active_chan_prop_all
df_stim['prop_target_noise'][:n_stim] = prop_target_noise
df_stim['n_spikes_target_in'][:n_stim] = n_spikes_target

df_stim.to_csv(os.path.join(res_dir, '{}_out.csv'.format(stim_filename[:-4])), sep=';')
