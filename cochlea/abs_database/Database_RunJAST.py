import os
from scipy.io import wavfile
import tqdm
import re
import csv
from cochlea import *
from STDP import *

spikelist_dirpath = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases\Database_6_20ms_1\SpikeLists\cochlea_model_inhib_160418_1324\IN'
database_params = re.findall('\d+', re.search('Database_\d+_\d+ms_\d+', spikelist_dirpath)[0])
n_repeat_target, chunk_duration_ms,  n_noise_iter = [int(param) for param in database_params]

if n_repeat_target % 2:
    raise ValueError('compute_scores is defined only for even number of repetitions')

fs, n_channels = 44100, 1024
spikelist_files = os.listdir(spikelist_dirpath)
n_spikelist = len(spikelist_files)

# JAST2 parameters
kMax = 1
M, P = 1024, 1024
N, W, T_i, T_f, T_firing = 32, 16, 5, 13, 11
refract_period_s = 0.002
dT, d_n_swap = 0.1, 1
learn_ratio_threshold = 2
active_chan_threshold = 1

# Make dir for output spikelist
datetime_str = datetime.strftime(datetime.now(), '%d%m%Y_%Hh%M')
out_spike_dirpath = os.path.join(spikelist_dirpath[:-2], 'OUT_{}'.format(datetime_str))
if os.path.exists(out_spike_dirpath):
    raise ValueError('Directory already exists : {}'.format(out_spike_dirpath))
os.mkdir(out_spike_dirpath)

# Compute scores and store them in a csv file
target_pattern_id, noise_pattern_id = 1, 2
# Open csv file
csv_filename = 'JAST2_results.csv'
f = open(os.path.join(out_spike_dirpath, csv_filename), 'w')
csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)


def compute_scores(out_spikelist, target_pattern_id=1, noise_pattern_id=2, learn_ratio_threshold=2,
                   active_chan_threshold=1):
    out_spikelist_sel = out_spikelist[out_spikelist.time > out_spikelist.tmax/2]
    target_spikes_ind = out_spikelist_sel.pattern_id == target_pattern_id
    noise_spikes_ind = out_spikelist_sel.pattern_id == noise_pattern_id
    n_target_spikes, n_noise_spikes = np.sum(target_spikes_ind), np.sum(noise_spikes_ind)
    if n_target_spikes == 0 and n_noise_spikes == 0:
        learning_ratio, active_channel_ratio, sound_learned = np.nan, np.nan, 0
    else:
        learning_ratio = n_target_spikes / max(1, n_noise_spikes)
        active_channel_ratio = np.unique(out_spikelist_sel.channel[target_spikes_ind]).size / \
                               max(1, np.unique(out_spikelist_sel.channel[noise_spikes_ind]).size)
        if (learning_ratio > learn_ratio_threshold) and (active_channel_ratio > active_chan_threshold):
            sound_learned = 1
        else:
            sound_learned = 0
    return learning_ratio, active_channel_ratio, sound_learned


for i, spikelist_filename_i in tqdm.tqdm(enumerate(spikelist_files)):
    spikelist_in_i = import_spikelist_from_mat(os.path.join(spikelist_dirpath, spikelist_filename_i), n_channels)
    target_name_i = spikelist_filename_i.split('_')[4][:-4]
    # Run JAST2
    spikelist_out_i, _, _ = STDP_v2(spikelist_in_i, fs, N, W, M, P, dT=0.5, n_swap_i=[], d_n_swap=[], min_n_swap=1, T_i=T_i,
                                    T_f=T_f, T_firing=T_firing, same_chan_in_buffer_max=kMax, full_mode=0)
    # Save output spikelist
    spikelist_out_i.export(out_spike_dirpath, '{}_1_{}'.format(spikelist_filename_i[:12], spikelist_filename_i[15:]))
    print('tmax_in :{} - tmax_out : {}'.format(spikelist_in_i.tmax, spikelist_out_i.tmax))
    # Compute scores
    learning_ratio, active_channel_ratio, sound_learned = compute_scores(spikelist_out_i)
    # Write row in the csv file
    csv_writer.writerow([i, target_name_i, spikelist_in_i.n_spikes, spikelist_out_i.n_spikes, learning_ratio,
                         active_channel_ratio, sound_learned])


f.close()

