from sklearn.model_selection import ParameterGrid
import generate_signals
from cochlea import *
import seaborn as sns
import pandas as pd
from datetime import datetime
import tqdm
import os
import _pickle


def stdp_params_grid_search(spikelist_dir, fs, stdp_params_all, learn_ratio_threshold=2, active_chan_threshold=1,
                            save_dir='./results'):
    """ Apply STDP on all the spikelists in ``spikelist_dir``, compute several scores. Do this for each set of STDP
    parameters.

    First score is the learning ratio, defined as the ratio between number of spikes occuring during the target sound
    and the number of spikes occuring during the noise periods.
    Second score is the active channel ratio, defined as the number of active channels during target over the number
    of active channel during noise. An active channel is a channel with at least one spike.

    For each set of STDP parameters, the median and the standard deviation of each score is returned. A final score
    is incrementer each time the learning ratio and the active channel ratio are above a their respective thresholds
    (2, by default). We consider that in this case the STDP has learn to detect the target sound. Thus this measure
    the number of sound correctly learned.

    Parameters
    ----------
    spikelist_dir : str
        Path of the directory containing the spike-lists
    fs : float
        Sampling rate (Hz)
    stdp_params_all : dict
        Dictionnary containing values for the STDP parameters

    Returns
    -------

    """
    spikelist_names = os.listdir(spikelist_dir)
    n_spikelist = len(spikelist_names)
    if not type(stdp_params_all) == list:
        stdp_params_all = [stdp_params_all]
    stdp_grid = ParameterGrid(stdp_params_all)
    learn_ratio_med, learn_ratio_std, active_chan_med, active_chan_std = np.zeros((4, len(stdp_grid)))
    n_target_spikes_mean = np.zeros(len(stdp_grid))
    n_stim_learned = np.zeros(len(stdp_grid), dtype=int)
    learn_ratio_all, active_chan_all, n_target_spikes_all = np.zeros((3, len(stdp_grid), n_spikelist))
    stdp_param_list = []
    for i_param in tqdm.tqdm(range(len(stdp_grid))):
        stdp_param_i = stdp_grid[i_param]
        stdp_param_list.append(stdp_param_i)
        for i_spklist, spikelist_name_i in enumerate(spikelist_names):
            spikelist_i = import_spikelist_from_mat(os.path.join(spikelist_dir, spikelist_name_i), stdp_param_i['M'])
            learn_ratio_all[i_param, i_spklist], active_chan_all[i_param, i_spklist], \
            n_target_spikes_all[i_param, i_spklist] = apply_stdp_on_spikelist(spikelist_i, fs, stdp_param_i)
            if (learn_ratio_all[i_param, i_spklist] > learn_ratio_threshold) and \
                    (active_chan_all[i_param, i_spklist] > active_chan_threshold):
                n_stim_learned[i_param] += 1
        learn_ratio_med[i_param], learn_ratio_std[i_param] = np.nanmedian(learn_ratio_all[i_param]), np.nanstd(learn_ratio_all[i_param])
        active_chan_med[i_param], active_chan_std[i_param] = np.nanmedian(active_chan_all[i_param]), np.nanstd(active_chan_all[i_param])
        n_target_spikes_mean[i_param] = np.nanmean(n_target_spikes_all[i_param])
    df_stdp_params = pd.DataFrame(stdp_param_list, columns=['M', 'P', 'N', 'W', 'T_i', 'T_firing', 'T_f', 'dT', 'd_n_swap'])
    df_scores = pd.DataFrame({'Learn Ratio - median': learn_ratio_med, 'Learn ratio - std': learn_ratio_std,
                              'Active Chan - median': active_chan_med, 'Active Chan - std': active_chan_std,
                              'Sounds learned (%)': np.array(100*n_stim_learned/n_spikelist).astype(int),
                              'N Spikes Target': n_target_spikes_mean},
                             columns=['Learn Ratio - median', 'Learn ratio - std', 'Active Chan - median',
                                      'Active Chan - std', 'Sounds learned (%)', 'N Spikes Target'])
    df = pd.concat([df_stdp_params, df_scores], axis=1)
    # Create save directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir_date = os.path.join(save_dir, datetime.strftime(datetime.now(), '%d%m%y_%H%M'))
    os.mkdir(save_dir_date)
    # Save results pandas data frame
    df.to_csv(os.path.join(save_dir_date, 'results.csv'))
    # Create a dict with main results data
    res_dict = {'learn_ratio_all': learn_ratio_all, 'active_chan_all': active_chan_all,
                'n_target_spikes_all': n_target_spikes_all}
    with open(path.join(save_dir_date, 'res_dict.p'), 'wb') as f:
        _pickle.dump(res_dict, f)
    # Figures
    plot_results(stdp_params_all, learn_ratio_all, active_chan_all, n_target_spikes_all, save_dir=save_dir_date)


def plot_results(stdp_params_all, learn_ratio_all, active_chan_all, n_target_spikes_all, save_dir=[]):
    stdp_grid = ParameterGrid(stdp_params_all)
    f = plt.figure()
    n_param_sets = len(stdp_grid)
    n_rows = int(np.ceil(n_param_sets/5))
    for i_param in range(n_param_sets):
        if i_param == 0:
            ax0 = f.add_subplot(n_rows, 5, i_param+1)
            ax0.set(xlabel='Learning ratio', ylabel='Count', title='Set 0')
        else:
            ax = f.add_subplot(n_rows, 5, i_param + 1, sharex=ax0, sharey=ax0)
            ax.set(xlabel='Learning ratio', title='Set {}'.format(i_param))
        learn_ratio_i = learn_ratio_all[i_param, ~np.isnan(learn_ratio_all[i_param])]
        learn_ratio_i[learn_ratio_i > 10] = 10
        plt.hist(learn_ratio_i, bins=[0, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10], alpha=0.8, rwidth=0.8)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    f.show()
    if save_dir:
        f.savefig(os.path.join(save_dir_date, 'learning_ratio.png'), dpi=400)

    f = plt.figure()
    n_param_sets = len(stdp_grid)
    n_rows = int(np.ceil(n_param_sets/5))
    for i_param in range(n_param_sets):
        if i_param == 0:
            ax0 = f.add_subplot(n_rows, 5, i_param+1)
            ax0.set(xlabel='Active channel ratio', ylabel='Count', title='Set 0')
        else:
            ax = f.add_subplot(n_rows, 5, i_param + 1, sharex=ax0, sharey=ax0)
            ax.set(xlabel='Active channel ratio', title='Set {}'.format(i_param))
        active_chan_i = active_chan_all[i_param, ~np.isnan(active_chan_all[i_param])]
        active_chan_i[active_chan_i > 4] = 4
        plt.hist(active_chan_i, bins=[0, 0.2, 0.5, 1, 1.5, 2, 3, 4], alpha=0.8, rwidth=0.8)
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()
    if save_dir:
        plt.savefig(os.path.join(save_dir_date, 'active_channel_ratio.png'), dpi=400)

    f = plt.figure()
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    for i_param in range(n_param_sets):
        if i_param == 0:
            ax0 = f.add_subplot(n_rows, 5, i_param+1)
            ax0.set(xlabel='Learning ratio', ylabel='Active channel Ratio', title='Set 0')
        else:
            ax = f.add_subplot(n_rows, 5, i_param + 1, sharex=ax0, sharey=ax0)
            ax.set(xlabel='Learning ratio', ylabel='Active channel Ratio', title='Set {}'.format(i_param))
        learn_ratio_i = learn_ratio_all[i_param, ~np.isnan(learn_ratio_all[i_param])]
        learn_ratio_i[learn_ratio_i > 10] = 10
        active_chan_i = active_chan_all[i_param, ~np.isnan(active_chan_all[i_param])]
        active_chan_i[active_chan_i > 10] = 10
        n_target_spikes_i = n_target_spikes_all[i_param, ~np.isnan(n_target_spikes_all[i_param])]
        mean_firing_rate_target = n_target_spikes_i / 1024 / 0.05
        im = plt.scatter(learn_ratio_i, active_chan_i, c=mean_firing_rate_target, cmap=plt.cm.coolwarm)
        plt.colorbar(im)
    if save_dir:
        f.savefig(os.path.join(save_dir_date, 'learn_active_chan_joint_plot.png'), dpi=400)


def apply_stdp_on_spikelist(in_spikelist, fs, stdp_params, plot_results=0, cochlea=[]):
    """ Given an input spike list ``in_spikelist`` and STDP parameters in ``stdp_params``, compute a learning score.
    Parameters
    ----------
    in_spikelist :
    fs :
    stdp_params :
    plot_results :

    Returns
    -------

    """
    if not type(stdp_params) == dict:
        raise ValueError('Parameter stdp_param must be a dictionnary')
    params_needed = ['M', 'P', 'N', 'W', 'T_i', 'T_firing']
    param_opt = {'dT': 0.5, 'n_swap_i': [], 'd_n_swap': [], 'min_n_swap': 1}
    for param in params_needed:
        if param not in list(stdp_params.keys()):
            raise ValueError('Missing key in stdp_params : {}'.format(param))
        if not np.isscalar(stdp_params[param]):
            raise ValueError('Values in stdp_param should be scalars')
    for param_o, value_o in param_opt.items():
        if param_o not in list(stdp_params.keys()):
            stdp_params[param_o] = value_o
    if 'T_f' not in list(stdp_params.keys()):
        stdp_params['T_f'] = stdp_params['T_firing']
    if 'same_chan_in_buffer_max' not in list(stdp_params.keys()):
        stdp_params['same_chan_in_buffer_max'] = 1
    if stdp_params['W'] > stdp_params['N'] or stdp_params['T_firing'] > stdp_params['W']:
        return np.nan, np.nan, np.nan
    else:
        try:
            _, weights, _ = STDP_v2(in_spikelist, fs, stdp_params['N'], stdp_params['W'], stdp_params['M'],
                                    stdp_params['P'], dT=stdp_params['dT'], n_swap_i=stdp_params['n_swap_i'],
                                    d_n_swap=stdp_params['d_n_swap'], min_n_swap=stdp_params['min_n_swap'],
                                    T_i=stdp_params['T_i'], T_f=stdp_params['T_f'], T_firing=stdp_params['T_firing'],
                                    same_chan_in_buffer_max=stdp_params['same_chan_in_buffer_max'])
        except:
            print('Error in JAST2')
            return np.nan, np.nan, np.nan

        try:
            out_spikelist, _, _ = STDP_v2(in_spikelist, fs, stdp_params['N'], stdp_params['W'], stdp_params['M'],
                                          stdp_params['P'], dT=stdp_params['dT'], n_swap_i=stdp_params['n_swap_i'],
                                          d_n_swap=stdp_params['d_n_swap'], min_n_swap=stdp_params['min_n_swap'],
                                          T_i=stdp_params['T_i'], T_f=stdp_params['T_f'], T_firing=stdp_params['T_firing'],
                                          weight_init=weights, freeze_weight=True,
                                          same_chan_in_buffer_max=stdp_params['same_chan_in_buffer_max'])
        except:
            print('Error in JAST2')
            return np.nan, np.nan, np.nan

        noise_spikes_ind, target_spikes_ind = out_spikelist.pattern_id == 0, out_spikelist.pattern_id == 1
        n_noise_spikes, n_target_spikes = np.sum(noise_spikes_ind), np.sum(target_spikes_ind)
        learning_ratio = n_target_spikes / max(1, n_noise_spikes)
        active_channel_ratio = np.unique(out_spikelist.channel[target_spikes_ind]).size / \
                                         max(1, np.unique(out_spikelist.channel[noise_spikes_ind]).size)
        if plot_results:
            cochlea_tau_lif = cochlea.lifbank.tau if cochlea else []
            ax_list = dual_spikelist_plot(spike_list, out_spikelist, tau_lif=cochlea_tau_lif, pattern_id_sel_in=1)
            ax_list[4].set(title='Raster plot - Learn_ratio={:.2f}, Active_chan_ratio={:.2f}'.format(learning_ratio, active_channel_ratio))

        return learning_ratio, active_channel_ratio, n_target_spikes


def load_results(filepath):
    with open(filepath, 'rb') as f:
        res_dict = _pickle.load(f)
    learn_ratio_all = res_dict['learn_ratio_all']
    active_chan_all = res_dict['active_chan_all']
    n_target_spikes_all = res_dict['n_target_spikes_all']
