import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns
import h5py
from scipy.io import wavfile
from freq_analysis_utils import *
import tqdm
sns.set()
sns.set_context('paper')


## Analyze the results of the ABS experiment
# Define a meta-subject, composed of all subject who did the experiment with the sames stimuli (it changes every 4 subjects)
# Compute a performance score for each stimulus, i.e. the number of subjects that have detected the target in the stimulus
# Compute spectral characteristics of each stimulus
wav_stim_dir = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_ExperienceManu\EXP_stim_set1\stimuli'
res_mat_path = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_ExperienceManu\EXP_stim_set1\ABS_MakeSequence_cfg.mat'
res_dir = r'C:\Users\deudon\Desktop\M4\_Results\ABS_Exp_Results'
meta_sub_num = [1, 2, 3, 4]


def plot_histograms(df, subject_desc=''):
    df_sum = df.groupby(['snip_duration', 'n_rep', 'n_dis'])['good_response'].sum().reset_index()
    df_snipduration = df_sum.groupby('snip_duration').sum().reset_index()
    df_snipduration['good_response_prop'] = np.array(df_snipduration['good_response']) / np.array(df['snip_duration'].value_counts())
    df_nrep = df_sum.groupby('n_rep').sum().reset_index()
    df_nrep['good_response_prop'] = np.array(df_nrep['good_response']) / np.array(df['n_rep'].value_counts())
    df_ndis = df_sum.groupby('n_dis').sum().reset_index()
    df_ndis['good_response_prop'] = np.array(df_ndis['good_response']) / np.array(df['n_dis'].value_counts())

    prop_good_response = 100*df['good_response'].sum()/df.shape[0]
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    sns.barplot(x='snip_duration', y='good_response_prop', data=df_snipduration, ax=ax1)
    sns.barplot(x='n_rep', y='good_response_prop', data=df_nrep, ax=ax2)
    sns.barplot(x='n_dis', y='good_response_prop', data=df_ndis, ax=ax3)
    ax2.set(title='Subject {} ({:.1f}% good responses)'.format(subject_desc, prop_good_response))

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    df_snipdur_nrep_prop = np.array(df.groupby(['snip_duration', 'n_rep'])['good_response'].mean())
    df_snipdur_nrep_ndis_prop = np.array(df.groupby(['snip_duration', 'n_rep', 'n_dis'])['good_response'].mean())
    norm = mpl.colors.Normalize(vmin=df_snipdur_nrep_ndis_prop.min(), vmax=df_snipdur_nrep_ndis_prop.max())
    cmap_str = "RdBu_r"
    ax1.imshow(np.atleast_2d(df_snipduration['good_response_prop']).T, origin='lower', aspect='auto', cmap=cmap_str, norm=norm)
    ax2.imshow(np.atleast_2d(df_snipdur_nrep_prop).T, origin='lower', aspect='auto', cmap=cmap_str, norm=norm)
    im3=ax3.imshow(np.atleast_2d(df_snipdur_nrep_ndis_prop).T, origin='lower', aspect='auto', cmap=cmap_str, norm=norm)
    plt.colorbar(im3)
    ax1.grid(False), ax2.grid(False), ax3.grid(False)
    ax1.set_xticks([0]), ax2.set_xticks([0]), ax3.set_xticks([0])
    ax1.set_xticklabels(['Snip Duration (s)']), ax2.set_xticklabels(['N rep']), ax3.set_xticklabels(['N dis'])
    ax1.set_yticks(np.arange(df['snip_duration'].nunique())), ax1.set_yticklabels(df_snipduration['snip_duration'])
    ax2.set_yticks(df.groupby(['snip_duration', 'n_rep'])['good_response'].mean().reset_index()['n_rep'].index)
    ax2.set_yticklabels(df.groupby(['snip_duration', 'n_rep'])['good_response'].mean().reset_index()['n_rep'])
    ax3.set_yticks(df.groupby(['snip_duration', 'n_rep', 'n_dis'])['good_response'].mean().reset_index()['n_dis'].index)
    ax3.set_yticklabels(df.groupby(['snip_duration', 'n_rep', 'n_dis'])['good_response'].mean().reset_index()['n_dis'])
    ax2.set(title='Subject {} ({:.1f}% good responses)'.format(subject_desc, prop_good_response))


for subject_num in meta_sub_num:
    df = pd.read_csv(os.path.join(res_dir, '{}_results.csv'.format(subject_num)))
    if 'Unnamed: 0' in df.keys():
        df = df.drop('Unnamed: 0', axis=1)
    plot_histograms(df, str(subject_num))

df_all = []
i_trial = 0
for sub_num in meta_sub_num:
    df_i = (pd.read_csv(os.path.join(res_dir, '{}_results.csv'.format(sub_num))))
    df_i.columns = ['trial'] + list(df_i.columns[1:])
    df_i['subject'] = sub_num
    df_i.set_index(pd.RangeIndex(start=i_trial, stop=i_trial+df_i.shape[0]), inplace=True)
    if df_i.shape[0] != 480:
        print('Warning : Number of trials is {} for subject {}'.format(df_i.shape[0], sub_num))
    i_trial += df_i.shape[0]
    df_all.append(df_i)

df_metasub = pd.concat(df_all)
plot_histograms(df_metasub, 'Meta-Subject')

# Look at the good response rate for each subject
win_size = 30
plt.figure()
for df_i in df_all:
    df_i.set_index(pd.to_datetime(df_i['trial'], unit='s'), inplace=True)
    plt.plot(np.array(df_i['good_response'].rolling(window=win_size).mean()), alpha=0.9)
plt.autoscale(axis='x', tight=True)
plt.legend(['subject {}'.format(sub_i) for sub_i in meta_sub_num])

df_stim_params = df_all[0].loc[:, ['snip_duration', 'n_rep', 'n_dis']]
df_stim_params.set_index(df_all[0]['stim_name'], inplace=True)
n_stim = df_stim_params.shape[0]

# Look at each stimulus
df_stim_grp = df_metasub.groupby('stim_name')['good_response'].sum()
df_stim_grp_sorted = df_stim_grp.sort_values(ascending=False)

df_stim_param_perf = df_stim_params.loc[df_stim_grp_sorted.index, :]
df_stim_param_perf['perf'] = np.array(df_stim_grp_sorted)
df_stim_param_perf = df_stim_param_perf.reset_index()
print(df_stim_param_perf[df_stim_param_perf.perf == 4])
print(df_stim_param_perf[df_stim_param_perf.perf == 3].mean())

df_mean_param_per_perf = df_stim_param_perf.groupby('perf').mean()

# Look at best/worst stim caracteristics
stim_perf_4 = df_stim_param_perf[df_stim_param_perf.perf == 4]['stim_name']
stim_perf_3 = df_stim_param_perf[df_stim_param_perf.perf == 3]['stim_name']
stim_perf_2 = df_stim_param_perf[df_stim_param_perf.perf == 2]['stim_name']
stim_perf_1 = df_stim_param_perf[df_stim_param_perf.perf == 1]['stim_name']
stim_perf_0 = df_stim_param_perf[df_stim_param_perf.perf == 0]['stim_name']

best_stim = stim_perf_4
worst_stim = stim_perf_0

# Compute Spectral features for all stim
spect_centroid, spect_rolloff, spect_npeaks = np.zeros((3, n_stim))
peaks_freq = []
n_fft, fmin, fmax = 1024, 20, 15000
pxx_db = np.zeros((n_stim, 1+n_fft//2))
# Compute spectral feature
for i, stim_name_i in tqdm.tqdm(enumerate(df_stim_param_perf.stim_name)):
    # Read target (wavfile)
    fs_i, wav_target_i = wavfile.read(os.path.join(wav_stim_dir, '{}_target.wav'.format(stim_name_i[:-4])))
    if wav_target_i.ndim > 1:
        print('Stereo wav file. Keep only the first channel')
        wav_target_i = wav_target_i[:, 1]
    spect_centroid[i], spect_rolloff[i], peaks_freq_i, pxx_db[i, :], freqs = get_spectral_features(wav_target_i, fs_i, do_plot=0, nfft=n_fft)
    peaks_freq.append(peaks_freq_i)
    spect_npeaks[i] = peaks_freq_i.size

# Read the ABS_MakeSequence_cfg.mat matlab matrix to get the time of the target apparition and add it to the data frame
f_mat = h5py.File(res_mat_path, 'r')
dataset_tpos = f_mat.get('stream/seq/TposSec')
dataset_filename = f_mat.get('stream/seq/filename')
target_pos_sec = []
filename_list = []
filename_order = np.zeros(dataset_tpos.shape[0], dtype=int)
stim_name_arr = np.array([stim_name[:-4] for stim_name in np.array(df_stim_param_perf.stim_name)])
for i in range(dataset_tpos.shape[0]):
    target_pos_sec.append(np.array(f_mat[dataset_tpos[i, 0]]).tolist())
    filename_list.append([''.join(chr(c) for c in f_mat[dataset_filename[i, 0]][1:])][0])

for i in range(dataset_tpos.shape[0]):
    filename_order[i] = int(np.where(stim_name_arr[i] == np.array(filename_list))[0])


target_pos_sec_ord = [np.atleast_1d(np.array(target_pos_sec[i]).squeeze()) for i in filename_order]
df_stim_param_perf['target_pos_sec'] = target_pos_sec_ord

# Add spectral characteristics to the data frame
df_stim_param_perf['spect_centroid'] = spect_centroid
df_stim_param_perf['spect_rolloff'] = spect_rolloff
df_stim_param_perf['spect_npeaks'] = spect_npeaks
df_mean_param_per_perf = df_stim_param_perf.groupby('perf').mean()
stim_set_num = (max(meta_sub_num)-1) // 4 + 1
df_stim_param_perf.to_csv(os.path.join(res_dir, 'stim_set_#{}_carac.csv'.format(stim_set_num)), sep=';')


def get_pxx(stim_names, n_fft=1048):
    pxx_target = np.zeros((stim_names.size, 1+n_fft//2))
    for i, stim_name_i in enumerate(stim_names):
        # Read target (wavfile)
        fs_i, wav_target_i = wavfile.read(os.path.join(wav_stim_dir, '{}_target.wav'.format(stim_name_i[:-4])))
        if wav_target_i.ndim > 1:
            print('Stereo wav file. Keep only the first channel')
            wav_target_i = wav_target_i[:, 1]
        _, _, _, pxx_target[i, :], freqs = get_spectral_features(wav_target_i, fs_i, do_plot=0, nfft=n_fft)
    return pxx_target, freqs

n_fft = 512
pxx_best, freqs = get_pxx(best_stim, n_fft)
pxx_worst, _ = get_pxx(worst_stim, n_fft)

f, ax = plt.subplots()
ax.plot(freqs, pxx_best.mean(0))
ax.plot(freqs, pxx_worst.mean(0))
ax.set_xscale('log')
plt.autoscale(axis='x', tight=True)
plt.legend(['Best stims', 'Worst stims'])

# f, ax = plt.subplots()
# ax.plot(freqs, pxx_best.T)
# ax.legend(list(pxx_best))
# plt.autoscale(axis='x', tight=True)
# ax.set_xscale('log')

