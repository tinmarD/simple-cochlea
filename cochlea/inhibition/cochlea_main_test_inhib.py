from cochlea import *
import seaborn
import generate_signals

sns.set()
sns.set_context('paper')

abs_dirpath = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS\ABS_sequences_export'
cochlea_dir = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_models'
cochlea_name = 'cochlea_model_270218_1358.p'

fs = 44100
# Cochlea parameters
n_channels = 1024
freq_scale = 'erbscale'
fbank_type = 'bessel'
# # fs = 44100
fmin = 1000
fmax = 8000
forder = 2
rectifier_type = 'full'
rectifier_lp_freq = []
compression_factor = 0.2
compression_gain = 0.8
lif_v_spike = 0.35
lif_t_refract = 0.001
lif_v_reset = -1
if freq_scale == 'erbscale':
    _, cf = erbscale(fs, fmin, fmax, n_channels)
elif freq_scale == 'linearscale':
    _, cf = linearscale(fs, fmin, fmax, n_channels)
elif freq_scale == 'musicscale':
    _, cf = musicscale(fs, fmin, fmax, n_channels)
lif_tau = 1 * 1 / cf
lif_tau = np.array([max(0.0004, lif_tau_i) for lif_tau_i in lif_tau])
lif_v_thresh = np.linspace(0.5, 0.15, n_channels)
lif_v_thresh = np.array([max(0.2, v_i) for v_i in lif_v_thresh])
# lif_v_thresh = np.array([min(0.35, v_i) for v_i in lif_v_thresh])

cochlea_2 = Cochlea(n_channels, fs, fmin, fmax, freq_scale, forder, fbank_type=fbank_type, rect_type=rectifier_type,
                    rect_lowpass_freq=rectifier_lp_freq, comp_factor=compression_factor,
                    comp_gain=compression_gain, lif_tau=lif_tau, lif_v_thresh=lif_v_thresh,
                    lif_v_spike=lif_v_spike, lif_t_refract=lif_t_refract, lif_v_reset=lif_v_reset)

cochlea = load_cochlea(cochlea_dir, cochlea_name)

chunk_duration = 0.050
n_repet_target = 3
fs, signal_abs, sound_order, sound_names, target_sig = generate_signals.generate_abs_stim \
    (abs_dirpath, chunk_duration, n_repet_target)
signal_sinus = generate_signals.generate_sinus(44100, [2000, 3000], t_offset=[0, 0.05], t_max=0.2, amplitude=1)

signal_in = signal_abs
# signal_in = signal_sinus

spike_list, _ = cochlea_2.process_input(signal_in, do_plot=0)
# spike_list_sub, _ = cochlea.process_input_with_inhib(signal_in, inhib_type='sub_for',
#                                                      inhib_vect=np.array([-0.025, 0.01, -0.025]))
spike_list_shunt, _ = cochlea_2.process_input_with_inhib(signal_in, inhib_type='shunt_for',
                                                         inhib_vect=np.array([0.003, 0.006, 0.009, 0, 0.009, 0.006, 0.003]))
spike_list_shunt_cur, _ = cochlea_2.process_input_with_inhib(signal_in, inhib_type='shunt_for_current',
                                                           inhib_vect=np.array([0.05, 0.1, 0.15, 0, 0.15, 0.1, 0.05]))
# spike_list_spikeinhib, _ = cochlea_2.process_input_with_inhib(signal_in, inhib_type='spike',
#                                                              inhib_vect=np.array([-0.1, -0.2, 0, -0.2, -0.1]))

print(spike_list)
# print(spike_list_sub)
# print(spike_list_shunt)
# print(spike_list_shunt_cur)
# print(spike_list_spikeinhib)

# dual_spikelist_plot(spike_list, spike_list_sub)
# dual_spikelist_plot(spike_list, spike_list_shunt)
# dual_spikelist_plot(spike_list, spike_list_shunt_cur)
# dual_spikelist_plot(spike_list, spike_list_spikeinhib)


# STDP
# M, P = 1024, 1024
# N, W, T_i, T_f, T_firing = 32, 16, 5, 11, 11
# t_refract = 0
# out_spikelist, weights, neu_thresh = STDP_v2(spike_list, fs, N, W, M, P, dT=0.5, n_swap_i=[], d_n_swap=[],
#                                              min_n_swap=1, T_i=T_i, T_f=T_f, T_firing=T_firing, refract_period_s=t_refract)
# # out_spikelist_sub, weights, neu_thresh = STDP_v2(spike_list_sub, fs, N, W, M, P, dT=0.5, n_swap_i=[], d_n_swap=[],
# #                                                  min_n_swap=1, T_i=T_i, T_f=T_f, T_firing=T_firing, refract_period_s=t_refract)
# out_spikelist_shunt, weights, neu_thresh = STDP_v2(spike_list_shunt, fs, N, W, M, P, dT=0.5, n_swap_i=[], d_n_swap=[],
#                                                    min_n_swap=1, T_i=T_i, T_f=T_f, T_firing=T_firing, refract_period_s=t_refract)
# out_spikelist_spikeinhib, weights, neu_thresh = STDP_v2(spike_list_spikeinhib, fs, N, W, M, P, dT=0.5, n_swap_i=[], d_n_swap=[],
#                                                    min_n_swap=1, T_i=T_i, T_f=T_f, T_firing=T_firing, refract_period_s=t_refract)
# # Figures
# dual_spikelist_plot(spike_list, out_spikelist)
# # dual_spikelist_plot(spike_list_sub, out_spikelist_sub)
# dual_spikelist_plot(spike_list_shunt, out_spikelist_shunt)
# dual_spikelist_plot(spike_list_spikeinhib, out_spikelist_spikeinhib)
