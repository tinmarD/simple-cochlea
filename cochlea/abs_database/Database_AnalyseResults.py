## After STDP has been applied on the input spikelist and results scores have been computed, this script analyse
# the results :
#   - Which stimuli were not learned
#   - Difference between learned stimuli and not learned ones
#   - Influence of the spectrum

import numpy as np
import os
from freq_analysis_utils import *
from scipy.io import wavfile
import seaborn as sns
import pandas as pd
sns.set()

database_root_path = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases\Database_10_50ms_1'
# result_dirpath = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases\Database_10_50ms_1\SpikeLists\cochlea_model_inhib_160418_1324\OUT_24092018_16h14'
# database_root_path = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases\Database_10_50ms_1'
# result_dirpath = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases\Database_10_50ms_1\SpikeLists\cochlea_model_adapt_thresh_FS_inhib_1k_8kHz\OUT_24092018_15h16'
# database_root_path = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases\Database_10_50ms_1'
result_dirpath = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases\Database_10_50ms_1\SpikeLists\cochlea_model_adapt_thresh_FS_inhib_1k_8kHz_lowomega\OUT_25092018_12h11'
fmin, fmax = 1000, 15000

# database_root_path = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases\Database_10_50ms_1'
# result_dirpath = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases\Database_6_20ms_1\SpikeLists\cochlea_model_adapt_thresh_FS_inhib_1k_8kHz_lowomega\OUT_25092018_15h59'

csv_res_filepath = os.path.join(result_dirpath, 'JAST2_results.csv')
target_dirpath = os.path.join(database_root_path, 'Target')
target_files = os.listdir(target_dirpath)
n_stim = len(target_files)

spect_centroid, spect_rolloff = np.zeros((2, n_stim))
peaks_freq = []
# Get size of pxx and freqs
fs_0, wav_target_0 = wavfile.read(os.path.join(target_dirpath, target_files[0]))
_, _, _, pxx_0, _ = get_spectral_features(wav_target_0, fs_0, fmin, fmax)
pxx_db = np.zeros((n_stim, pxx_0.size))

# Read csv results file with pandas
df_res = pd.read_csv(csv_res_filepath, sep=',',  names=['pos', 'target_name', 'n_spikes_in', 'n_spikes_out',
                                                        'learning_ratio','active_channel_ratio', 'sound_learned'])

learned_ind = df_res['sound_learned'] == 1
notlearned_ind = df_res['sound_learned'] == 0
n_learned, n_not_learned = sum(learned_ind), sum(notlearned_ind)
print('{} stim learned'.format(n_learned))

f = plt.figure()
sns.violinplot(x='sound_learned', y='n_spikes_in',  data=df_res)

f = plt.figure()
sns.scatterplot(x='n_spikes_in', y='n_spikes_out', hue='sound_learned', data=df_res)


# Compute spectral feature of audio target
for i, target_file_i in enumerate(target_files):
    # Read target (wavfile)
    fs_i, wav_target_i = wavfile.read(os.path.join(target_dirpath, target_file_i))
    spect_centroid[i], spect_rolloff[i], peaks_freq_i, pxx_db[i, :], freqs = get_spectral_features(wav_target_i, fs_i, fmin, fmax, do_plot=0)
    peaks_freq.append(peaks_freq_i)


# Plot mean spectral feature of learned and not learned stim
colors = sns.color_palette(n_colors=2)
f = plt.figure()
ax = f.add_subplot(111)
pxx_learn_mean, pxx_learn_std = pxx_db[learned_ind, :].mean(0), pxx_db[learned_ind, :].std(0)
pxx_notlearn_mean, pxx_notlearn_std = pxx_db[notlearned_ind, :].mean(0), pxx_db[notlearned_ind, :].std(0)
ax.plot(freqs, pxx_learn_mean, c=colors[0])
ax.fill_between(freqs, pxx_learn_mean-pxx_learn_std, pxx_learn_mean+pxx_learn_std, color=colors[0], alpha=0.2)
ax.plot(freqs, pxx_notlearn_mean, c=colors[1])
ax.fill_between(freqs, pxx_notlearn_mean-pxx_notlearn_std, pxx_notlearn_mean+pxx_notlearn_std, color=colors[1], alpha=0.2)
ax.axvline(spect_centroid[learned_ind].mean(), color=colors[0])
ax.axvline(spect_centroid[notlearned_ind].mean(), color=colors[1])
ax.autoscale(axis="x", tight=True)
ax.set(xlabel='Frequency (Hz)', ylabel='Gain (dB)', title='Spectral Features')
ax.set_xscale('log')
ax.grid(True, which="both", ls="-")
plt.legend(['Learned sound (N={})'.format(n_learned), 'Not-learned sound (N={})'.format(n_not_learned)])


# Plot spectral feature for all not learned sounds
for notlearned_target_i in np.array(target_files)[notlearned_ind]:
    fs, wav_target = wavfile.read(os.path.join(target_dirpath, notlearned_target_i))
    get_spectral_features(wav_target, fs, fmin, fmax, do_plot=1)

f = plt.figure()
ax = f.add_subplot(111)
# Plot spectral feature for all not learned sounds
for notlearned_target_i in np.array(target_files)[notlearned_ind]:
    fs, wav_target = wavfile.read(os.path.join(target_dirpath, notlearned_target_i))
    _, _, _, pxx_i, freqs = get_spectral_features(wav_target, fs, fmin, fmax, do_plot=0)
    ax.plot(freqs, pxx_i, c=colors[1], alpha=0.5)
ax.set_xscale('log')
ax.grid(True, which="both", ls="-")
ax.autoscale(axis="x", tight=True)
ax.set(xlabel='Frequency (Hz)', ylabel='Gain (dB)', title='Spectral Features')
ax.plot(freqs, pxx_learn_mean, c=colors[0])

