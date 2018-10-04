import numpy as np
import seaborn as sns
from multiprocessing import Pool
from cochlea import *


def plot_test_sound(signal, fs, pattern_pos, t_chunk_start, chunk_duration):
    plt.figure()
    n_chunks = len(t_chunk_start)
    t_vect = np.linspace(0, len(signal)/fs, len(signal))
    for i in range(0, n_chunks):
        color = 'g' if i in pattern_pos else 'b'
        ind_i = (t_vect >= t_chunk_start[i]) & (t_vect <= (t_chunk_start[i] + chunk_duration))
        plt.plot(t_vect[ind_i], signal[ind_i], c=color)

if __name__ == '__main__':
    # Input sig
    abs_dirpath = 'C:\\Users\\deudon\\Desktop\\M4\\_Data\\audioStim\\ABS\\ABS_sequences_export'
    results_filepath = 'C:\\Users\\deudon\\Desktop\\M4\\_Results\\Python_cochlea\\ParamSearch_results_vthreshmidtest.csv'
    chunk_duration = 0.050
    n_repet_target = 3
    n_iter = 10

    fs, _ = wavfile.read(os.path.join(abs_dirpath, os.listdir(abs_dirpath)[0]))
    cochlea_estimator = CochleaEstimator(fs=fs, n_filters=1024, freq_scale='erbscale', fmin=[200, 1000], fmax=8000,
                                         forder=[2], comp_gain=[0.8, 1])
    cochlea_estimator = CochleaEstimator(fs=fs, n_filters=1024, freq_scale='erbscale', fmin=1000, fmax=8000,
                                         fbank_type='bessel', forder=[2], lif_tau_coeff=1, comp_gain=0.8,
                                         lif_v_thresh_start=[0.4, 0.5], lif_v_thresh_stop=[0.1, 0.2, 0.3],
                                         lif_v_thresh_mid=[0.1, 0.2, 0.3, 0.4], comp_factor=0.3)
    # cochlea_estimator = CochleaEstimator(fs=fs, n_filters=1024, freq_scale='erbscale', fmin=[200, 1000], fmax=8000,
    #                                      forder=[2], fbank_type='bessel', comp_factor=1/3.0, comp_gain=[0.8, 1, 1.2],
    #                                      lif_tau_coeff=[0.5], lif_v_thresh_start=[0.4, 0.5], lif_v_thresh_stop=[0.15])
    print(cochlea_estimator)
    n_comb = cochlea_estimator.n_combinations

    global_score, vr_score, spkchanvar_score, channel_with_spikes_ratio, n_spikes_mean = np.zeros((5, n_iter, n_comb))
    target_sound_name = []
    for i_iter in tqdm.tqdm(range(0, n_iter)):
        # Generate the signal
        target_found = 1
        while target_found:
            _, signal_i, sound_order, sound_names = generate_signals.merge_wav_sound_from_dir\
                (abs_dirpath, chunk_duration, 1+n_repet_target, np.hstack([n_repet_target, np.ones(n_repet_target)]),
                 max_sound_repet=0)
            target_found = 1 if sound_names[0] in target_sound_name else 0
        target_sound_name.append(sound_names[0])
        pattern_pos = np.where(sound_order == 0)[0]
        t_chunk_start = np.linspace(0, chunk_duration * (n_repet_target*2 - 1), int(n_repet_target*2))
        t_chunk_end = t_chunk_start + 1.0 * chunk_duration
        # Find best parameters
        global_score[i_iter, :], vr_score[i_iter, :], spkchanvar_score[i_iter, :], \
        channel_with_spikes_ratio[i_iter, :], n_spikes_mean[i_iter, :], param_grid, df = \
            cochlea_estimator.grid_search(signal_i, t_chunk_start, t_chunk_end, pattern_pos,
                                          input_sig_name=sound_names[0])
        # Save results
        try:
            write_header = True if not os.path.isfile(results_filepath) or os.path.getsize(results_filepath) == 0 else False
            df.to_csv(results_filepath, header=write_header, mode='a+', float_format='%.3f')
        except Exception as e:
            print('Could not write results : {}'.format(e))
    global_score_mean, vr_score_mean = np.nanmean(global_score, axis=0), np.nanmean(vr_score, axis=0)
    spkchanvar_score_mean, n_spikes_mean_mean = np.nanmean(spkchanvar_score, axis=0), np.nanmean(n_spikes_mean, axis=0)
    channel_with_spikes_ratio_mean = np.nanmean(channel_with_spikes_ratio, axis=0)
    # Plot the mean results
    cochlea_estimator.plot_results(global_score_mean, vr_score_mean, spkchanvar_score_mean,
                                   channel_with_spikes_ratio_mean, n_spikes_mean_mean, param_grid, order='global_score')
    cochlea_estimator.plot_param_scores(global_score_mean, vr_score_mean, spkchanvar_score_mean,
                                        channel_with_spikes_ratio_mean, n_spikes_mean_mean, param_key=['lif_v_thresh_mid'])

