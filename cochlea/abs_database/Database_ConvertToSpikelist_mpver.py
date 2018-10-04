## Given the path to the database containing the wav files and the cochlea model, convert these stimuli into spikelists
# Pattern id is set for each spikelist. Spikelist are then exported

import os
from scipy.io import wavfile
import tqdm
import multiprocessing as mp
import re
from cochlea import *
from generate_signals import *


database_dirpath = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases'
database_name = r'Database_10_50ms_3'
n_repeat_target, chunk_duration_ms,  n_noise_iter = [int(param) for param in re.findall('\d+', database_name)]
cochlea_dirpath = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_models'
cochlea_name = 'cochlea_model_270218_1358.p'
n_channels = 1024

stim_path = os.path.join(database_dirpath, database_name, 'STIM')
spikelist_dir = os.path.join(database_dirpath, database_name, 'SpikeLists')
spikelist_cochlea_dir = os.path.join(spikelist_dir, cochlea_name[:-2], 'IN')

if not os.path.exists(spikelist_dir):
    os.mkdir(spikelist_dir)
if not os.path.exists(os.path.join(spikelist_dir, cochlea_name[:-2])):
    os.mkdir(os.path.join(spikelist_dir, cochlea_name[:-2]))
if not os.path.exists(spikelist_cochlea_dir):
    os.mkdir(spikelist_cochlea_dir)

stim_files = os.listdir(stim_path)
n_stims = len(stim_files)

# Load cochlea
cochlea = load_cochlea(cochlea_dirpath, cochlea_name)
print(cochlea)

chunk_pattern_id, pattern_dict, chunk_start, chunk_end = get_abs_stim_params(chunk_duration_ms/1000, n_repeat_target, n_noise_iter)


t_start = time.time()

pool = mp.Pool(processes=6)

# for i, stim_filename_i in tqdm.tqdm(enumerate(stim_files)):
target_name, wav_stim = [], []
for i, stim_filename_i in enumerate(stim_files):
    # Read stim (wavfile)
    fs_i, wav_stim_i = wavfile.read(os.path.join(stim_path, stim_filename_i))
    wav_stim.append(wav_stim_i)
    target_name.append(stim_filename_i.split('_')[1])
    if not fs_i == cochlea.fs:
        raise ValueError('Sampling rate of the stimulus is not equal to the sampling rate of the cochlea')

output = [pool.apply_async(cochlea.process_input, args=(wav_stim_i,)) for wav_stim_i in wav_stim]
spike_list_all = [p.get()[0] for p in output]

for spike_list in spike_list_all:
    spike_list = spike_list.sort('time')
    spike_list.set_pattern_id_from_time_limits(chunk_start, chunk_end, chunk_pattern_id, pattern_dict)
    spike_list.export(spikelist_cochlea_dir, export_name='{}_spike_list_0_{}.mat'.format(i, target_name_i))

print('Elasped time : {}'.format(time.time() - t_start))
