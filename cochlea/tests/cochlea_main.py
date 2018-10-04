from cochlea import *
import seaborn as sns
import matplotlib.patches as mpatches


sns.set()
sns.set_context('paper')
fs = 44100
spikelist_dir = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_spikelist_base_2_inhib'
spikelist_name = r'spike_list_0_A0000006208.mat'
cochlea_dir = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_models'
# cochlea_name = 'cochlea_model_270218_1358.p'
cochlea_name = 'cochlea_model_inhib_160418_1324.p'
abs_dirpath = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS\ABS_sequences_export'
save_dir = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\JAST2_param_search'

# learning_ratio = -1
# while learning_ratio < 2 or learning_ratio > 3:
# Import spikelist and cochlea
spike_list = import_spikelist_from_mat(os.path.join(spikelist_dir, spikelist_name), 1024)
spike_list.pattern_names[0] = 'Noise'
spike_list.pattern_names[1] = 'Target'
cochlea = load_cochlea(cochlea_dir, cochlea_name)

kMax = 1
M, P = 1024, 1024
# N, W, T_i, T_f, T_firing = 64, 30, 7, 25, 18
N, W, T_i, T_f, T_firing = 32, 16, 5, 13, 11
refract_period_s = 0.002
dT, d_n_swap = 0.1, 1
# out_spikelist, weights, neu_thresh, afferents = STDP_v2(spike_list, fs, N, W, M, P, dT=0.5, n_swap_i=[], d_n_swap=[],
#                                                         min_n_swap=1, T_i=T_i, T_f=T_f, T_firing=T_firing,
#                                                         same_chan_in_buffer_max=kMax, full_mode=0)
#
# dual_spikelist_plot(spike_list, out_spikelist)

out_spikelist, weight, thresh_neu, weightset_spk, learn_spikelist, T_all, n_swap_all, weightset_learn = STDP_v2\
    (spike_list, fs, N, W, M, P, dT=dT, n_swap_i=[], d_n_swap=d_n_swap, min_n_swap=1, T_i=T_i, T_f=T_f,
     T_firing=T_firing, refract_period_s=refract_period_s, same_chan_in_buffer_max=kMax, full_mode=1)

# out_spikelist_freezed, weight_freezed, thresh_neu_freezed, weightset_spk_freezed, learn_spikelist_freezed, \
# T_all_freezed, n_swap_all_freezed, weightset_learn_freezed = STDP_v2\
#     (spike_list, fs, N, W, M, P, T_i=T_i, T_f=T_f, T_firing=T_firing, refract_period_s=refract_period_s,
#      same_chan_in_buffer_max=kMax, full_mode=1, weight_init=weight, freeze_weight=True)

dual_spikelist_plot(spike_list, out_spikelist)
# dual_spikelist_plot(spike_list, out_spikelist_freezed)


def plot_spikes_repartition(spikelist):
    n_channels = spikelist.n_channels
    n_spike_target, n_spike_noise = np.zeros((2, spikelist.n_channels), dtype=int)
    mean_pot_target, mean_pot_noise = np.zeros((2, spikelist.n_channels), dtype=float)
    for i in range(n_channels):
        spike_pos_i = np.where(spikelist.channel == i)[0]
        target_spikes_pos = spike_pos_i[spikelist[spike_pos_i].pattern_id == 1]
        noise_spikes_pos = spike_pos_i[spikelist[spike_pos_i].pattern_id == 0]
        n_spike_target[i], n_spike_noise[i] = target_spikes_pos.size, noise_spikes_pos.size
        mean_pot_target[i] = np.mean(spikelist[target_spikes_pos].potential)
        mean_pot_noise[i] = np.mean(spikelist[noise_spikes_pos].potential)
    spiking_channel = (n_spike_noise > 0) | (n_spike_target > 0)
    n_spiking_channels = np.sum(spiking_channel)
    score = -1*n_spike_noise + 1*n_spike_target
    df = pd.DataFrame({'pos': range(n_channels), 'n_spikes': n_spike_target+n_spike_noise, 'n_target': n_spike_target,
                       'n_noise': n_spike_noise, 'score': score},
                      columns=['pos', 'n_spikes', 'n_target', 'n_noise','score'])
    df_spiking = df[df['n_spikes'] > 0]
    df_spiking_sorted = df_spiking.sort_values(['score'])
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.bar(range(n_spiking_channels), df_spiking_sorted['n_target'], width=1)
    ax.bar(range(n_spiking_channels), df_spiking_sorted['n_noise'], width=1)
    ax.set_xticks(range(n_spiking_channels))
    ax.set_xticklabels(df_spiking_sorted['pos'])
    ax.set(xlabel='Neuron number', ylabel='Count')
    ax.legend(['Target Spikes', 'Noise Spikes'])
    plt.autoscale(axis='x', tight=True)


def plot_learning_ev(pos, spike_list, learn_list, weightset_spk, weightset_learn, W):

    spike_pos = np.where(spike_list.channel == pos)[0]
    learn_pos = np.where(learn_list.channel == pos)[0]
    colors, labels = sns.color_palette(n_colors=2), ['Spiking Event', 'Learning Event']
    colors_2, labels_2 = sns.color_palette(n_colors=3), ['Spike Potential', 'Learning Threshold', 'n_swap']
    f = plt.figure()
    ax0 = plt.subplot2grid((4, 7), (0, 0), rowspan=3, colspan=7)
    for i_spk in spike_pos:
        ax0.plot(spike_list.time[i_spk] * np.ones(W), weightset_spk[i_spk, :], '.', color=colors[0], markersize=3)
    for i_learn in learn_pos:
        ax0.plot(learn_list.time[i_learn] * np.ones(W), weightset_learn[i_learn, :], '.', color=colors[1], markersize=3)

    ax1 = plt.subplot2grid((4, 7), (3, 0), rowspan=1, colspan=7, sharex=ax0)
    ax2 = ax1.twinx()
    ax1.plot(spike_list.time[spike_pos], spike_list.potential[spike_pos], marker='o', color=colors_2[0], markersize=3)
    # ax1.plot(learn_list.time[learn_pos], learn_list.potential[learn_pos], marker='o', color=colors_2[0], markersize=3)
    ax1.plot(learn_list[learn_pos].time, T_all[learn_pos], marker='o', color=colors_2[1], markersize=3)
    ax2.plot(learn_list[learn_pos].time, n_swap_all[learn_pos], marker='o', color=colors_2[2], markersize=3)
    ax0.set_xlim((spike_list.tmin, spike_list.tmax))
    ax0.set_ylim((0, spike_list.n_channels))
    ax0.set(ylabel='Channels', title='JAST neuron {} - Weights equal to 1'.format(pos))
    ax0.legend(handles=[mpatches.Patch(color=colors[i], label='{}'.format(labels[i])) for i in range(2)])
    ax1.set(xlabel='Time (s)', ylabel='Potential / Threshold')
    ax2.set(ylabel='n_swap')
    ax2.grid(False)
    ax2.legend(handles=[mpatches.Patch(color=colors_2[i], label='{}'.format(labels_2[i])) for i in range(3)])
    ax1.plot([spike_list.tmin, spike_list.tmax], [T_firing, T_firing], 'k', ls=':')


# plot_spikes_repartition(out_spikelist)
# # plot_spikes_repartition(out_spikelist_freezed)
#
# jast_neu_pos = 877
# plot_learning_ev(jast_neu_pos, out_spikelist, learn_spikelist, weightset_spk, weightset_learn, W)
# plot_learning_ev(jast_neu_pos, out_spikelist_freezed, learn_spikelist_freezed, weightset_spk_freezed,
#                  weightset_learn_freezed, W)






    # spklist = out_spikelist
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(spklist.time, spklist.channel, '.', color='k', markersize=4,  picker=2)
#
# def onpick(event):
#     spike = event.artist
#     ind = event.ind[0]
#     print(ind)
#     print('Pattern id : {}'.format(int(spklist[ind].pattern_id)))
#
# fig.canvas.mpl_connect('pick_event', onpick)