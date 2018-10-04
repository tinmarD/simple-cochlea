from cochlea import *
import cochlea_utils


abs_stim_path = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases\Database_5_20ms_1\STIM\14_A0000017602_5_20ms_1.wav'
fs, fmin, fmax, freq_scale, n_channels = 44100, 200, 8000, 'erbscale', 1000
comp_factor, comp_gain = 0.3, 1.5
tau, v_thresh = np.linspace(0.001, 0.0004, n_channels), np.linspace(0.3, 0.17, n_channels)
refract_period, v_spike, t_refract, v_reset = 0, 0.5, 0.002, 0
# Adaptive Threshold
alpha_rs, alpha_ib, alpha_fs = np.array([0.037, 0.002]), np.array([0.0017, 0.002]), np.array([0.010, 0.000002])
# w_rs, w_ib, w_fs = 0.019, 0.026, 0.011
tau_j, alpha_j = np.array([0.010, 0.200]), alpha_ib
w = np.linspace(0.25, 0.2, n_channels)
t_refract_mat = 0

cochlea_adapt_rs = Cochlea(n_channels, fs, fmin, fmax, freq_scale, comp_factor=comp_factor, comp_gain=comp_gain,
                           lif_tau=tau, lif_v_thresh=v_thresh, lif_v_spike=v_spike, lif_t_refract=[],
                           lif_v_reset=v_reset, tau_j=tau_j, alpha_j=alpha_rs, omega=w)
cochlea_adapt_ib = Cochlea(n_channels, fs, fmin, fmax, freq_scale, comp_factor=comp_factor, comp_gain=comp_gain,
                           lif_tau=tau, lif_v_thresh=v_thresh, lif_v_spike=v_spike, lif_t_refract=[],
                           lif_v_reset=v_reset, tau_j=tau_j, alpha_j=alpha_ib, omega=w)
cochlea_adapt_fs = Cochlea(n_channels, fs, fmin, fmax, freq_scale, comp_factor=comp_factor, comp_gain=comp_gain,
                           lif_tau=tau, lif_v_thresh=v_thresh, lif_v_spike=v_spike, lif_t_refract=[],
                           lif_v_reset=v_reset, tau_j=tau_j, alpha_j=alpha_fs, omega=w)

v_thresh = np.linspace(0.5, 0.4, n_channels)
cochlea_std = Cochlea(n_channels, fs, fmin, fmax, freq_scale, comp_factor=comp_factor, comp_gain=comp_gain,
                      lif_tau=tau, lif_v_thresh=v_thresh, lif_v_spike=v_spike, lif_t_refract=[],
                      lif_v_reset=-1)


# Inhibition :
N, inhib_sum = 50, 2
inhib_vect = signal.gaussian(2*N+1, std=15)
inhib_vect[N] = -2
inhib_vect_norm = inhib_sum * inhib_vect / inhib_vect.sum()


# Input signal
step = generate_signals.generate_step(fs, t_offset=0.05, t_max=0.15)
sin = generate_signals.generate_sinus(fs, f_sin=268, t_offset=0.05, t_max=0.2)
sin_dur, sin_offset = 0.1*np.random.rand(5), 0.1*np.random.rand(5)
t_max = sin_offset+sin_dur
t_start = np.hstack([0, t_max[:-1]])
sin_serie = np.hstack([generate_signals.generate_sinus(fs, f_sin=800, t_offset=t_offset_i, t_max=t_max_i)
             for (t_start_i, t_offset_i, t_max_i) in zip(t_start, sin_offset, t_max)])
# t_vect = np.linspace(0, np.hstack(sin_serie).size / fs, np.hstack(sin_serie).size)
# plt.plot(t_vect, np.hstack(sin_serie))

_, abs_sig = wavfile.read(r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases\Database_5_20ms_1\STIM\14_A0000017602_5_20ms_1.wav')
if abs_sig.ndim == 2:
    abs_sig = abs_sig[:, 0]
input_sig = sin_serie
input_sig = cochlea_utils.normalize_vector(input_sig)

ichan = 300
cochlea_std.plot_channel_evolution(input_sig, ichan)
cochlea_adapt_rs.plot_channel_evolution(input_sig, ichan)
cochlea_adapt_ib.plot_channel_evolution(input_sig, ichan)
cochlea_adapt_fs.plot_channel_evolution(input_sig, ichan)

spklist_sin_std = cochlea_std.process_input(input_sig)
spklist_sin_adapt_rs = cochlea_adapt_rs.process_input(input_sig)
spklist_sin_adapt_ib = cochlea_adapt_ib.process_input(input_sig)
spklist_sin_adapt_fs = cochlea_adapt_fs.process_input(input_sig)
spklist_sin_std.plot()
spklist_sin_adapt_rs.plot()
spklist_sin_adapt_ib.plot()
spklist_sin_adapt_fs.plot()

