from cochlea import *
import seaborn as sns
import matplotlib.patches as mpatches
import os
import pandas as pd
import re
from STDP import *
sns.set()
sns.set_context('paper')

fs = 44100
M = 1024
spikelist_dir = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_spikelist_base_2'
save_dir = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\JAST2_on_base'
# cochlea_dir = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_models'
# cochlea_name = 'cochlea_model_inhib_160418_1324.p'
# JAST 2 parameters
M, P, N, W = 1024, 1024, 32, 16
T_i, T_firing, T_f = 5, 9, 9
kmax = 1

# Create save directory
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
base_name = re.findall(r'\\[^\\]+', spikelist_dir)[-1][1:]
save_dir_date = os.path.join(save_dir, '{}_{}'.format(datetime.strftime(datetime.now(), '%d%m%y_%H%M'), base_name))
os.mkdir(save_dir_date)



# cochlea = load_cochlea(cochlea_dir, cochlea_name)
# print(cochlea)

n_spikelists = len(os.listdir(spikelist_dir))
n_target_spikes, learning_ratio, active_channel_ratio, sound_learned = np.zeros((4, n_spikelists))
learn_ratio_threshold, active_chan_threshold = 2, 1

# for i, spikelist_name in enumerate(os.listdir(spikelist_dir)):
    print('{}/{}'.format(i+1, n_spikelists))
    in_spikelist = import_spikelist_from_mat(os.path.join(spikelist_dir, spikelist_name), n_channels=M)
    # Learning is on
    spikelist_learn, weights, _ = STDP_v2(in_spikelist, fs, N, W, M, P, T_i=T_i, T_firing=T_firing, T_f=T_f,
                                          same_chan_in_buffer_max=kmax, freeze_weight=False)
    # Learning is off
    out_spikelist, _, _ = STDP_v2(in_spikelist, fs, N, W, M, P, T_i=T_i, T_firing=T_firing, T_f=T_f,
                                  same_chan_in_buffer_max=kmax, weight_init=weights, freeze_weight=True)
    # dual_spikelist_plot(in_spikelist, spikelist_learn)
    # dual_spikelist_plot(in_spikelist, out_spikelist)

    noise_spikes_ind, target_spikes_ind = spikelist_learn.pattern_id == 0, spikelist_learn.pattern_id == 1
    n_noise_spikes, n_target_spikes[i] = np.sum(noise_spikes_ind), np.sum(target_spikes_ind)
    learning_ratio[i] = n_target_spikes[i] / max(1, n_noise_spikes)
    active_channel_ratio[i] = np.unique(spikelist_learn.channel[target_spikes_ind]).size / \
                                        max(1, np.unique(spikelist_learn.channel[noise_spikes_ind]).size)
    if learning_ratio[i] > learn_ratio_threshold and active_channel_ratio[i] > active_chan_threshold:
        sound_learned[i] = 1
    else:
        sound_learned[i] = 0


df = pd.DataFrame({'Name': os.listdir(spikelist_dir), 'Learning ratio': learning_ratio,
                   'Active channel ratio': active_channel_ratio, 'n target spikes out': n_target_spikes,
                   'Sound learned': sound_learned.astype(int)},
                  columns=['Name', 'n target spikes out', 'Learning ratio', 'Active channel ratio', 'Sound learned'])


df.to_csv(os.path.join(save_dir_date, 'results_{}.csv'.format(base_name)))