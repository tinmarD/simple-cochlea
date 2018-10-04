from cochlea import *
from scipy.io import wavfile
from LIF_AdaptiveThreshold_test import *
import cochlea_utils
import generate_signals
import re


# abs_dirpath = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_ExperienceManu\EXP_stim_set1\stimuli'
# abs_filename = r'ABS_seq_2_SnipSz_0.1_nbrep_2_nbdis_2_gap0.wav'
database_dirpath = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases'
database_name = r'Database_5_20ms_1'
database_stim_path = os.path.join(database_dirpath, database_name, 'STIM')
stim_name = r'14_A0000017602_5_20ms_1.wav'
[n_repet_target, chunk_duration_ms, n_noise_iter] = [int(param) for param in re.findall('\d+', database_name)]
chunk_pattern_id, pattern_dict, chunk_start, chunk_end = generate_signals.get_abs_stim_params(chunk_duration_ms/1000, n_repet_target, n_noise_iter)

# Cochlea
cochlea_dir = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_models'
cochlea_name = r'cochlea_model_inhib_160418_1324.p'
# cochlea_name = r'cochlea_model_270218_1358.p'
cochlea_std = load_cochlea(cochlea_dir, cochlea_name)

fs, fmin, fmax, freq_scale = 44100, 1000, 8000, 'erbscale'
n_channels = 1000
comp_factor, comp_gain = 0.3, 1.5
tau, v_thresh = np.linspace(0.001, 0.0004, n_channels), np.linspace(0.3, 0.17, n_channels)
refract_period, v_spike, t_refract, v_reset = 0, 0.5, 0.002, 0
# Adaptive Threshold
alpha_rs, alpha_ib, alpha_fs = np.array([0.037, 0.002]), np.array([0.0017, 0.002]), np.array([0.010, 0.000002])
# w_rs, w_ib, w_fs = 0.019, 0.026, 0.011
tau_j, alpha_j = np.array([0.010, 0.200]), alpha_fs
# alpha_j = np.array([0.08, 0.0])
w = np.linspace(0.15, 0.2, n_channels)
t_refract_mat = 0
# Inhibition :
N, inhib_sum = 50, 2
inhib_vect = signal.gaussian(2*N+1, std=15)
inhib_vect[N] = -2
inhib_vect_norm = inhib_sum * inhib_vect / inhib_vect.sum()

cochlea_adapt = Cochlea(n_channels, fs, fmin, fmax, freq_scale, comp_factor=comp_factor, comp_gain=comp_gain,
                  lif_tau=tau, lif_v_thresh=v_thresh, lif_v_spike=v_spike, lif_t_refract=t_refract_mat, lif_v_reset=v_reset,
                  tau_j=tau_j, alpha_j=alpha_j, omega=w)
cochlea_adapt_inhib = Cochlea(n_channels, fs, fmin, fmax, freq_scale, comp_factor=comp_factor, comp_gain=comp_gain,
                  lif_tau=tau, lif_v_thresh=v_thresh, lif_v_spike=v_spike, lif_t_refract=t_refract_mat, lif_v_reset=v_reset,
                  tau_j=tau_j, alpha_j=alpha_j, omega=w, inhib_vect=inhib_vect_norm, inhib_type='shunt_for_current')

# Input signal
step = generate_signals.generate_step(fs, t_offset=0.05, t_max=0.15)
sin = generate_signals.generate_sinus(fs, 2200, 0.05, t_max=0.3)
_, abs_sig = wavfile.read(os.path.join(database_stim_path, stim_name))
if abs_sig.ndim == 2:
    abs_sig = abs_sig[:, 0]
input_sig = abs_sig
input_sig = cochlea_utils.normalize_vector(input_sig)

# Run Cochlea
spike_list_inhib = cochlea_std.process_input(input_sig)
spike_list_adapt = cochlea_adapt.process_input(input_sig)
spike_list_adapt_inhib = cochlea_adapt_inhib.process_input(input_sig)
# Set pattern id
spike_list_inhib.set_pattern_id_from_time_limits(chunk_start, chunk_end, chunk_pattern_id, pattern_dict)
spike_list_adapt.set_pattern_id_from_time_limits(chunk_start, chunk_end, chunk_pattern_id, pattern_dict)
spike_list_adapt_inhib.set_pattern_id_from_time_limits(chunk_start, chunk_end, chunk_pattern_id, pattern_dict)
# Plot spikelists
spike_list_inhib.plot()
spike_list_adapt.plot()
spike_list_adapt_inhib.plot()


######## Single channel ########

# sig_filtered, _ = cochlea.filterbank.filter_signal(input_sig, do_plot=0)
# sig_rectified = cochlea.rectifierbank.rectify_signal(sig_filtered, do_plot=0)
# sig_comp = cochlea.compressionbank.compress_signal(sig_rectified, do_plot=0)
# i_chan = 200
# t_spikes, v_out, sig_compressed, threshold = cochlea.process_one_channel_adaptive_threshold(input_sig, i_chan)
# tmax = input_sig.size / fs
# t_vect = np.linspace(0, tmax, input_sig.size)
# f = plt.figure()
# ax = f.add_subplot(211)
# ax.plot(t_vect, sig_compressed)
# ax2 = f.add_subplot(212, sharex=ax)
# ax2.plot(t_vect, v_out)
# ax2.plot(t_vect, threshold, 'k')

# input_x = sig_comp[i_chan, :]
# tmax = input_x.size / fs
# v_out, t_spikes = lif_standard(fs, refract_period, input_x, tau[i_chan], v_thresh[i_chan], v_spike, t_refract, v_reset)
# v_out_mat, t_spikes_mat, v_thresh_mat = lif_mat(fs, input_x, 1, t_refract, tau[i_chan], tau_j, alpha_j, omega)
# t_vect = np.linspace(0, tmax, input_x.size)
#
# f = plt.figure()
# ax = f.add_subplot(211)
# ax.plot(t_vect, input_x)
# # ax2 = f.add_subplot(312, sharex=ax)
# # ax2.plot(t_vect, v_out)
# # ax2.hlines(v_thresh[i_chan], 0, tmax)
# ax3 = f.add_subplot(212, sharex=ax)
# ax3.plot(t_vect, v_out_mat)
# ax3.plot(t_vect, v_thresh_mat, 'k')
# ax.autoscale(axis='x', tight=True)
