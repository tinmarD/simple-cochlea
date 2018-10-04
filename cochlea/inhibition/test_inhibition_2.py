from cochlea import *
import generate_signals
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tqdm
sns.set()
sns.set_context('paper')

cochlea_dir = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_models'
cochlea_name = 'cochlea_model_inhib_160418_1324.p'
abs_dirpath = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS\ABS_sequences_export'
export_dirpath = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_spikelist_base_2'
export_dirpath_inhib = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_spikelist_base_2_inhib'


fs = 44100
# # Cochlea parameters
n_channels = 1024
freq_scale = 'erbscale'
fbank_type = 'bessel'
# # fs = 44100
fmin = 500
fmax = 8000
forder = 2
rectifier_type = 'full'
rectifier_lp_freq = []
compression_factor = 0.3
compression_gain = 1.5
lif_v_spike, lif_t_refract, lif_v_reset = 0.35, [], -1
if freq_scale == 'erbscale':
    _, cf = erbscale(fs, fmin, fmax, n_channels)
elif freq_scale == "linearscale":
    _, cf = linearscale(fs, fmin, fmax, n_channels)
elif freq_scale == 'musicscale':
    _, cf = musicscale(fs, fmin, fmax, n_channels)
lif_tau = 1 * 1 / cf
lif_tau = np.array([max(0.0004, lif_tau_i) for lif_tau_i in lif_tau])
# lif_v_thresh = np.linspace(0.3, 0.17, n_channels)
lif_v_thresh = np.linspace(0.3, 0.17, n_channels)
# lif_v_thresh = np.array([max(0.2, v_i) for v_i in lif_v_thresh])
# Inhibition
N = 50
inhib_sum = 1.6
inhib_vect = signal.gaussian(2*N+1, std=15)
inhib_vect[N] = -2
inhib_vect_norm = inhib_sum * inhib_vect / inhib_vect.sum()
# plt.figure()
# plt.plot(inhib_vect_norm)

cochlea = Cochlea(n_channels, fs, fmin, fmax, freq_scale, forder, fbank_type=fbank_type, rect_type=rectifier_type,
                  rect_lowpass_freq=rectifier_lp_freq, comp_factor=compression_factor,
                  comp_gain=compression_gain, lif_tau=lif_tau, lif_v_thresh=lif_v_thresh,
                  lif_v_spike=lif_v_spike, lif_t_refract=lif_t_refract, lif_v_reset=lif_v_reset,
                  inhib_type='shunt_for_current', inhib_vect=inhib_vect_norm)

print(cochlea)
cochlea_ori = load_cochlea(cochlea_dir, cochlea_name)

# for i in tqdm.tqdm(range(10)):
chunk_duration = 0.050
n_repet_target = 10
sig = generate_signals.generate_sinus(fs=44100, f_sin=[2000, 2100, 6000, 6500], t_offset=[0, 0.1, 0.05, 0.15])
fs, sig, sound_order, sound_names, target_sig = generate_signals.generate_abs_stim(abs_dirpath, chunk_duration, n_repet_target)

spikelist, _ = cochlea_ori.process_input(sig)
spikelist_inhib, _ = cochlea.process_input(sig)

t_chunk_start = np.linspace(0, chunk_duration * (2*n_repet_target - 1), int(2*n_repet_target))
t_chunk_end = t_chunk_start + 1.0*chunk_duration

spikelist.set_pattern_id_from_time_limits(t_chunk_start, t_chunk_end, sound_order, sound_names)
spikelist_inhib.set_pattern_id_from_time_limits(t_chunk_start, t_chunk_end, sound_order, sound_names)

# Run STDP
M, P, N, W = 1024, 1024, 32, 16
T_i, T_firing, T_f = 5, 9, 9

out_spikelist, _, _ = STDP_v2(spikelist, fs, N, W, M, P, T_i=T_i, T_firing=T_firing, T_f=T_f, same_chan_in_buffer_max=1)
out_spikelist_inhib, _, _ = STDP_v2(spikelist_inhib, fs, N, W, M, P, T_i=T_i, T_firing=T_firing, T_f=T_f, same_chan_in_buffer_max=1)

# Plot
dual_spikelist_plot(spikelist, out_spikelist)
dual_spikelist_plot(spikelist_inhib, out_spikelist_inhib)

# spikelist.export(export_dirpath, export_name='spike_list_0_{}'.format(sound_names[1][:-4]))
# spikelist_inhib.export(export_dirpath_inhib, export_name='spike_list_0_{}'.format(sound_names[1][:-4]))

# print(spikelist.n_spikes)
# print(spikelist_inhib.n_spikes)

# spikelist.plot()
# spikelist_inhib.plot()
dual_spikelist_plot(spikelist, spikelist_inhib)

