import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

res_dir = r'C:\Users\deudon\Desktop\M4\_Results\ABS_Exp_Results'
subject_dirpath = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_ExperienceManu\#6'

subject_id = 'GO'
sound_ext = '.wav'
subject_num = int(re.search('\d+$', subject_dirpath)[0])

subject_files = os.listdir(subject_dirpath)
# Keep only results log files starting by MD_2 where (2 is the block number)
sel = [re.match('{}_\d+'.format(subject_id), subject_file_i) is not None for subject_file_i in subject_files]
subject_files_res = np.array(subject_files)[sel]


def get_params_from_stim_name(stim_name):
    seq = int(re.search('seq_\d+', stim_name)[0][4:])
    snip_duration = float(re.search('SnipSz_[\d.]+', stim_name)[0][7:])
    n_rep = int(re.search('nbrep_\d+', stim_name)[0][6:])
    n_dis = int(re.search('nbdis_\d+', stim_name)[0][6:])
    return seq, snip_duration, n_rep, n_dis

# Go through result file. Get the name of the stimulus. Until this name is unchanged, the subject has not decided yet
# which sound is the target. If the name changes, this is the next trial. To know if the answer is correct, look at the
# last name before the change, if it contains target, it's a good response.
# Columns of interest are 'Code' which contain the name of the stim and the names of the sounds playes
# Stim names must end with 'gap0' for the script to work

block_num, stim_name, seq_num, snip_duration, n_rep, n_dis, last_sound_played, good_response = [], [], [], [], [], [], [], []


for i_block, filename_i in enumerate(subject_files_res):
    # Read file of block i
    df = pd.read_csv(os.path.join(subject_dirpath, filename_i), sep='\t', header=2)

    i_stim, last_stim_name, last_sound_played_i = 0, '', ''
    n_rows = df.shape[0]
    for i in range(n_rows):
        code_i = df['Code'][i]
        if type(code_i) is not str or re.search('^ABS_seq_', code_i) is None:
            continue
        else:
            stim_name_i = code_i[:re.search('gap0', code_i).end()]+sound_ext
            # If new stim
            if stim_name_i != last_stim_name:
                # Get the params of this stim, add them to the lists
                seq_i, snip_duration_i, n_rep_i, n_dis_i = get_params_from_stim_name(stim_name_i)
                stim_name.append(stim_name_i)
                seq_num.append(seq_i)
                snip_duration.append(snip_duration_i)
                n_rep.append(n_rep_i)
                n_dis.append(n_dis_i)
                block_num.append(i_block+1)
                # If this is not the first stim, get the response of the previous stim
                if i_stim > 0:
                    last_sound_played.append(last_sound_played_i)
                    good_response.append('target' in last_sound_played_i)
                i_stim += 1
                last_stim_name = stim_name_i
            last_sound_played_i = code_i
    # Check last trial
    last_sound_played.append(last_sound_played_i)
    good_response.append('target' in last_sound_played_i)

df_res = pd.DataFrame({'block_num': block_num, 'stim_name': stim_name, 'seq': seq_num, 'snip_duration':snip_duration,
                       'n_rep': n_rep, 'n_dis': n_dis, 'last_sound_played': last_sound_played,
                       'good_response': np.array(good_response).astype(int)},
                      columns=['block_num', 'stim_name', 'seq', 'snip_duration', 'n_rep', 'n_dis', 'last_sound_played', 'good_response'])

# Save results
df_res.to_csv(os.path.join(res_dir, '{}_results.csv'.format(subject_num)))

