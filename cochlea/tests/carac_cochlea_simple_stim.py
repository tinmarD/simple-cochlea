from cochlea import *

cochlea_dir = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_models'
cochlea_name = 'cochlea_model_adapt_thresh_IB_200Hz.p'
abs_stim_path = r'C:\Users\deudon\Desktop\M4\_Data\audioStim\ABS_Databases\Database_5_20ms_1\STIM\14_A0000017602_5_20ms_1.wav'


# Import  cochlea
cochlea = load_cochlea(cochlea_dir, cochlea_name)

print(cochlea)

t_max, t_offset = 0.2, 0.05
x_test_lf = 0.01 * generate_signals.generate_sinus(44100, f_sin=226, t_max=t_max, t_offset=t_offset)
x_test_hf = 0.01 * generate_signals.generate_sinus(44100, f_sin=4830, t_max=t_max, t_offset=t_offset)
x_step = 1 * generate_signals.generate_step(44100, t_offset=t_offset, t_max=t_max)
print(cochlea)
cochlea.plot_channel_evolution(x_test_lf, channel_pos=20)
cochlea.plot_channel_evolution(x_test_hf, channel_pos=300)

spike_list_lf = cochlea.process_input(x_test_lf)
spike_list_hf = cochlea.process_input(x_test_hf)
spike_list_lf.plot()
spike_list_hf.plot()

spike_list_step = cochlea.process_input(x_step)
spike_list_step.plot()

# Abs stim
_, abs_sig = wavfile.read(abs_stim_path)
spike_list_abs = cochlea.process_input(abs_sig)
spike_list_abs.plot()
cochlea.plot_channel_evolution(abs_sig, channel_pos=40)
cochlea.plot_channel_evolution(abs_sig, channel_pos=800)

