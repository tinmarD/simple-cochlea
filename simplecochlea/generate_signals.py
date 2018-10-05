import numpy as np
import os
import random
from scipy.io import wavfile
import matplotlib.pyplot as plt
import tqdm

from .utils import utils_cochlea


def generate_sinus(fs, f_sin, t_offset=0, t_max=1, amplitude=1):
    if np.isscalar(f_sin):
        f_sin = np.array([f_sin])
    if np.isscalar(t_offset):
        t_offset = np.array([t_offset])
    if np.isscalar(amplitude):
        amplitude = amplitude * np.ones(len(f_sin))
    if not len(f_sin) == len(t_offset) == len(amplitude):
        raise ValueError('Arguments f_sin and t_offset must have the same size')
    if not np.isscalar(fs):
        raise ValueError('Argument fs must be a scalar')
    if not np.isscalar(t_max):
        raise ValueError('Argument t_max must be a scalar')
    n_sin = len(f_sin)
    n_pnts = int(t_max * fs)
    signals = np.zeros((n_sin, n_pnts))
    for i in range(0, n_sin):
        t_pre = np.arange(0, t_offset[i], 1.0 / fs)
        t_post = np.linspace(t_offset[i], t_max, n_pnts-len(t_pre))
        signals[i, :] = np.hstack([np.zeros(len(t_pre)), amplitude[i] * np.sin(2 * np.pi * f_sin[i] * t_post)])
    signals = signals.squeeze()
    if n_sin > 1:
        signals = signals.sum(0)
    return signals


def generate_dirac(fs, t_offset=0.2, t_max=1, amplitude=1):
    t_offset, t_max, amplitude = np.atleast_1d(t_offset), np.atleast_1d(t_max), np.atleast_1d(amplitude)
    if not len(t_offset) == len(amplitude):
        raise ValueError('Arguments t_offset and amplitude must have the same size')
    if not np.isscalar(fs):
        raise ValueError('Argument fs must be a scalar')
    if t_max.size > 1:
        raise ValueError('Argument t_max must be a scalar')
    sig_dirac = np.zeros(int(np.ceil(t_max*fs)))
    for i in range(0, len(t_offset)):
        sig_dirac[int(np.round(t_offset[i]*fs))] = amplitude[i]
    return sig_dirac


def generate_step(fs, t_offset=0.2, t_max=1, amplitude=1):
    t_offset, t_max, amplitude = np.atleast_1d(t_offset), np.atleast_1d(t_max), np.atleast_1d(amplitude)
    if not len(t_offset) == len(amplitude):
        raise ValueError('Arguments t_offset and amplitude must have the same size')
    if not np.isscalar(fs):
        raise ValueError('Argument fs must be a scalar')
    if t_max.size > 1:
        raise ValueError('Argument t_max must be a scalar')
    sig_step = np.zeros(int(np.ceil(t_max*fs)))
    for i in range(0, len(t_offset)):
        sig_step[int(np.round(t_offset[i]*fs)):-1] += amplitude[i]
    return sig_step


def merge_wav_sound_from_dir(dirpath, chunk_duration, n_sounds, n_repeat_per_sound=1, max_sound_repet=1):
    """

    Parameters
    ----------
    dirpath : str
        Path of the directory containing the ABS stimuli
    chunk_duration : float
        Duration of each segment (s)
    n_sounds : int
        Number of different sound in the sequence
    n_repeat_per_sound : int | array (default: 1)
        Number of repetition for each sound
    max_sound_repet : int
        Maximal number of times a sound can appears in a row

    Returns
    -------
    fs : float
        Sampling rate(Hz)
    sound_merged : array
        Output sound sequence
    sound_order : array
        Number of the sound in the sequence
    sound_names : array
        Name of the sound in the sequence

    """
    n_repeat_per_sound = np.array(n_repeat_per_sound)
    if n_repeat_per_sound.size == 0:
        n_repeat_per_sound = np.ones(n_sounds)
    if n_repeat_per_sound.size == 1:
        n_repeat_per_sound = n_repeat_per_sound * np.ones(n_sounds)
    n_repeat_per_sound = n_repeat_per_sound.astype(int)
    file_list = os.listdir(dirpath)
    n_files = len(file_list)
    if n_files < n_sounds:
        raise ValueError('The directory {} contain less than {} sounds'.format(dirpath, n_sounds))
    file_index = np.sort(random.sample(range(0, n_files), n_sounds))
    file_names = [file_list[index] for index in file_index]
    fs_arr = np.zeros(n_sounds)
    # Convert multi-channel files to mono and check the files are long enough
    for i in range(0, n_sounds):
        fs_arr[i], signal_i = wavfile.read(os.path.join(dirpath, file_names[i]))
        chunk_i_n_pnts = int(np.ceil(chunk_duration * fs_arr[i]))
        if len(signal_i) < chunk_i_n_pnts:
            raise ValueError('Signal {} is too shorter than chkun_duration argument : {}'.format(file_names[i],
                                                                                                 chunk_duration))
    # Check that all files have the same sampling frequency
    fs = np.unique(fs_arr)
    if len(fs) > 1:
        raise ValueError('Signal have differents sampling frquencies')
    else:
        fs = int(fs)
    # Construct signal
    n_chunks = int(np.sum(n_repeat_per_sound))
    chunks = np.zeros((n_chunks, chunk_i_n_pnts))
    sound_num_vect = np.zeros(n_chunks)
    chunk_inc = 0
    for i in range(0, n_sounds):
        _, signal_i = wavfile.read(os.path.join(dirpath, file_names[i]))
        if len(signal_i.shape) > 1:
            print('Keeping only the first channel of signal {}'.format(file_names[i]))
            signal_i = signal_i[:, 0]
        # Normalize the signal
        signal_i = utils_cochlea.normalize_vector(signal_i)
        for i_repet in range(0, n_repeat_per_sound[i]):
            chunks[chunk_inc, :] = signal_i[0:chunk_i_n_pnts]
            sound_num_vect[chunk_inc] = i
            chunk_inc += 1
    # Randomise the order
    max_repets, loop_count = max_sound_repet+2, 0
    while np.max(max_repets) > max_sound_repet and loop_count < 5000:
        loop_count += 1
        chunk_order = np.random.permutation(range(0, n_chunks))
        sound_order = sound_num_vect[chunk_order]
        max_repet, max_repets = 0, []
        for i in range(1, n_chunks):
            max_repet = max_repet+1 if sound_order[i-1] == sound_order[i] else 0
            max_repets.append(max_repet)
    if loop_count > 5000:
        raise ValueError('Cannot generate the sequence - check the parameters')
    chunks = chunks[chunk_order]
    sound_merged = np.hstack(chunks)
    sound_names = [file_names[int(sound_pos)] for sound_pos in sound_order]
    return fs, sound_merged, sound_order.astype(int), sound_names


def get_abs_stim_params(chunk_duration_s, n_repeat_target, n_noise_iter):
    n_noise_sounds = n_repeat_target*n_noise_iter
    n_chunks = n_noise_sounds + n_repeat_target     # Total number of segments
    pattern_id = np.zeros(n_chunks, dtype=int)
    pattern_name_dict = {1: 'Target', 2: 'Noise'}
    for i in range(n_chunks):
        if i % (n_noise_iter + 1) == n_noise_iter:  # Target
            pattern_id[i] = 1
        else:   # Noise
            pattern_id[i] = 2
    chunk_start = np.arange(0, (n_chunks-0.001)*chunk_duration_s, chunk_duration_s)
    chunk_end = np.arange(chunk_duration_s, (n_chunks+0.001)*chunk_duration_s, chunk_duration_s)
    return pattern_id, pattern_name_dict, chunk_start, chunk_end


def generate_abs_stim(dirpath, chunk_duration, n_repeat_target, n_noise_iter=1):
    """ Generate a stimulus used for Audio Brain Spotting (ABS). The stimulus is composed of one repeating target sound,
    between n_noise_iter noise sound. It alternate between n_noise_iter noise sound (non-repeating one) and
    the target sound, starting with a noise sound.

    Parameters
    ----------
    dirpath : str
        Path of the directory containing the ABS stimuli
    chunk_duration : float
        Duration of each segment (s)
    n_repeat_target : int
        Number of times the target sound is repeated
    n_noise_iter : int
        Number of noise segments between 2 target repetition

    Returns
    -------
    fs : float
        Sampling frequency (Hz)
    sound_merged : array
        Output ABS stimulus
    sound_order : array
        Pattern ID of each segment
    sound_names : array
        Pattern name of each segment
    sig_target_norm : array
        Target signal

    """
    n_noise_sounds = n_repeat_target*n_noise_iter
    n_sounds = n_noise_sounds + 1                   # Number of different sounds used in the stim (different than number of segments)
    n_chunks = n_noise_sounds + n_repeat_target     # Total number of segments
    file_list = os.listdir(dirpath)
    n_files = len(file_list)
    if n_files < n_noise_sounds+1:
        raise ValueError('The directory {} contain less than {} sounds'.format(dirpath, n_sounds))
    file_index = np.sort(random.sample(range(0, n_files), n_sounds))
    file_names = [file_list[index] for index in file_index]
    fs_arr = np.zeros(n_sounds, dtype=int)
    # Convert multi-channel files to mono and check the files are long enough
    for i in range(0, n_sounds):
        fs_arr[i], signal_i = wavfile.read(os.path.join(dirpath, file_names[i]))
        chunk_i_n_pnts = int(np.ceil(chunk_duration * fs_arr[i]))
        if len(signal_i) < chunk_i_n_pnts:
            raise ValueError('Signal {} is too shorter than chunk_duration argument : {}'.format(file_names[i],
                                                                                                 chunk_duration))
    # Check that all files have the same sampling frequency
    fs = np.unique(fs_arr)
    if len(fs) > 1:
        raise ValueError('Signal have differents sampling frquencies')
    else:
        fs = int(fs)
    # The target signal is the first one
    sig_target_name = file_names[0]
    _, sig_target = wavfile.read(os.path.join(dirpath, sig_target_name))
    if len(sig_target.shape) > 1:
        print('Keeping only the first channel of signal {}'.format(file_names[i]))
        sig_target = sig_target[:, 0]
    sig_target_norm = utils_cochlea.normalize_vector(sig_target)[0:chunk_i_n_pnts]
    sig_noise_norm = np.zeros((n_noise_sounds, chunk_i_n_pnts))
    chunks = np.zeros((n_chunks, chunk_i_n_pnts))
    sound_order, sound_names = np.zeros(n_chunks, dtype=int), list()
    i_noise = 0
    sig_noise_name = []
    # Construct signal, start with a noise signal
    for i in range(n_chunks):
        if i % (n_noise_iter+1) == n_noise_iter:  # Target
            chunks[i, :] = sig_target_norm
            sound_order[i] = 1
            sound_names.append(file_names[0])
        else:  # Noise
            _, sig_noise_i = wavfile.read(os.path.join(dirpath, file_names[1+i_noise]))
            sig_noise_name.append(file_names[1+i_noise])
            if len(sig_noise_i.shape) > 1:
                print('Keeping only the first channel')
                sig_noise_i = sig_noise_i[:, 0]
            sig_noise_norm[i_noise, :] = utils_cochlea.normalize_vector(sig_noise_i)[0:chunk_i_n_pnts]
            chunks[i, :] = sig_noise_norm[i_noise, :]
            sound_order[i] = 0
            sound_names.append(file_names[1+i_noise])
            i_noise += 1
    sound_merged = np.hstack(chunks)
    return fs, sound_merged, sound_order, np.array(sound_names), sig_target_norm, sig_noise_norm, sig_target_name,\
           np.array(sig_noise_name)


def plot_signal(x, fs, ax=[]):
    x = np.array(x)
    if not ax:
        f = plt.figure()
        ax = f.add_subplot(111)
    t = np.linspace(0, x.size / fs, x.size)
    ax.plot(t, x)
    ax.set(xlabel="time (s)", ylabel="Amplitude")


def delete_zero_signal(dirpath):
    """ There are some nulle signals (only zero amplitude) in the ABS directory. This function delete them.
    Delete also constant signals (only one single amplitude)
    """
    file_list = os.listdir(dirpath)
    n_files = len(file_list)
    file_names = [file_list[i] for i in range(n_files)]
    file_deleted_names = list()
    for i in tqdm.tqdm(range(n_files)):
        _, signal_i = wavfile.read(os.path.join(dirpath, file_names[i]))
        if np.unique(signal_i).size == 1:
            os.remove(os.path.join(dirpath, file_names[i]))
            file_deleted_names.append(file_names[i])
    return file_deleted_names

