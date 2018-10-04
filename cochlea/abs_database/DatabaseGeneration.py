from generate_signals import *
from scipy.io import wavfile
import os
import csv

N_stim = 100

abs_dirpath = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS\ABS_sequences_export'
database_dirpath = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases'

chunk_duration = 0.020
n_repeat_target = 6
n_noise_iter = 1

# Database name :
database_name = 'Database_{}_{}ms_{}'.format(n_repeat_target, int(1000*chunk_duration), n_noise_iter)

# Create directory structure
database_path = os.path.join(database_dirpath, database_name)
if os.path.exists(database_path):
    raise ValueError('Path already exist')
os.mkdir(database_path)
stim_dirpath = os.path.join(database_path, 'STIM')
target_dirpath = os.path.join(database_path, 'Target')
noise_dirpath = os.path.join(database_path, 'Noise')
os.mkdir(stim_dirpath)
os.mkdir(target_dirpath)
os.mkdir(noise_dirpath)
# Open csv file
csv_filename = database_name+'.csv'
f = open(os.path.join(database_path, csv_filename), 'w')
csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

for i_stim in range(N_stim):
    fs, stim_i, sound_order, sound_names, target_seg, noise_segs, target_name, noise_names = \
        generate_abs_stim(abs_dirpath, chunk_duration, n_repeat_target, n_noise_iter)
    # Save stimuli
    stim_name_i = '{}_{}_{}_{}ms_{}.wav'.format(i_stim, target_name[:-4], n_repeat_target, int(chunk_duration*1000),
                                                n_noise_iter)
    wavfile.write(os.path.join(stim_dirpath, stim_name_i), fs, stim_i)
    # Write target
    target_name_i = '{}_{}'.format(i_stim, target_name)
    wavfile.write(os.path.join(target_dirpath, target_name_i), fs, target_seg)
    # Write noise segments
    for j in range(noise_names.size):
        noise_name_i_j = '{}_{}_{}'.format(i_stim, j, noise_names[j])
        wavfile.write(os.path.join(noise_dirpath, noise_name_i_j), fs, noise_segs[j])
    csv_writer.writerow(np.array([i_stim, stim_name_i, target_name_i]))

f.close()


