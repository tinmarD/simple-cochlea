from sklearn.model_selection import ParameterGrid
from cochlea import *
import seaborn as sns
import pandas as pd
import tqdm
import os
import generate_signals
sns.set()


# Question : What is the mean firing rate of jast neurons on noise ?
# Generate a noise signal (no repetition) from ABS sounds
# Pass it through the cochlea
# Run JAST2
# See the mean firing rate

# ISSUE : depends on the JAST parameters

abs_dirpath = 'C:\\Users\\deudon\\Desktop\\M4\\_Data\\audioStim\\ABS\\ABS_sequences_export'
cochlea_dir = r'C:\Users\deudon\Desktop\M4\_Results\Python_cochlea\Cochlea_models'
cochlea_name = 'cochlea_model_270218_1358.p'

N = 50
n_out_spikes = np.zeros(N)
for i in tqdm.tqdm(range(0, N)):
    # Noise signal generation
    fs, noise_sig, sound_order, sound_names = generate_signals.merge_wav_sound_from_dir(abs_dirpath, 0.050, 20, 1)

    # Load cochlea
    cochlea = load_cochlea(cochlea_dir, cochlea_name)
    spike_list, _ = cochlea.process_input(noise_sig)

    # Run JAST2
    M, P = 1024, 1024
    N, W, T_i, T_f, T_firing = 64, 20, 5, 15, 15
    refract_T = 0.001
    out_spikelist, weights, neu_thresh = STDP_v2(spike_list, fs, N, W, M, P, dT=0.5, n_swap_i=[], d_n_swap=[],
                                                 min_n_swap=1, T_i=T_i, T_f=T_f, T_firing=T_firing,
                                                 refract_period_s=refract_T, )

    n_out_spikes[i] = out_spikelist.n_spikes


