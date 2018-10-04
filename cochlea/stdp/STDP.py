import numpy as np
import matplotlib.pyplot as plt

from ..spikes.spikelist import *
from jast2_cy import *
from jast2_fullmode_cy import *


def STDP_v2(spike_list, fs, N, W, M, P, dT=0.5, n_swap_i=[], d_n_swap=[], min_n_swap=1, T_i=5, T_f=11, T_firing=11,
            refract_period_s=0, weight_init=[], freeze_weight=[], same_chan_in_buffer_max=-1, full_mode=0):
    """ STDP (JAST) version 2 - timothee.masquelier@cnrs.fr - call the cython version
    Input spike-list
    Format: n x 3. Each line is: spike_time,
                                 afferent_number, (from 0 to M-1)
                                 pattern_id (number of the pattern if the spike belongs to one. Put NaN if this is
                                 unknown, which is the typical case for real applications)

    Output spike-list
    An output spike is produced whenever an input spike is integrated and the potential is above the firing threshold
    T_fire.
    Format: n x 4. Each line is: spike_time,
                                 afferent_number, (from 0 to P-1)
                                 pattern_id, (always equals to NaN)
                                 potential

    Parameters
    ----------
    spike_list : SpikeList instance
        Input spike list
    N : int
        JAST packet size
    W : int
        Number of weight set to 1
    M : int
        Number of input neurons
    P : int
        Number of output neurons
    dT : float
        Time step
    n_swap_i : int
        Initial number of swapped weights at each learning spike.
    d_n_swap : float
        Decrement of nb swapped weights at each learning spike, until min_n_swap is reached
    min_n_swap : int
        Min nb of swapp weight (put 0 to stop learning)
    T_i : int
        Initial learning threshold
    T_f : int
        Final learning threshold
    T_firing : int
        Firing threshold
    refract_period_s : float (default: 0)
        Refractory period (s). If none, no refractory period. Will be converted into samples, like the input spike-list
        time.
    weight_init : array | none
        Weight at initialisation. By default, weight are set randomly
    freeze_weight : bool | none (default: False)
        If True, freeze the weights, thus stop the learning

    Returns
    -------
    out_spikelist : SpikeList instance
        Output spike list
    weight : array (size M*P)
        Final weight matrix
    neu_thresh : array (size P)
        Output neuron's threshold
    """
    freeze_weight = 0 if not freeze_weight else 1
    spike_list = spike_list.sort('time')
    in_time, in_chan = spike_list.time.astype(float), spike_list.channel.astype(int)
    in_pattern = spike_list.pattern_id.astype(int)
    # Set nan pattern id (not defined) to value -1
    in_pattern[np.isnan(in_pattern)] = -1
    in_time_sample = (fs * in_time).astype(int)
    if not n_swap_i:
        n_swap_i = -1
    if not d_n_swap:
        d_n_swap = -1
    refract_period_samples = int(np.round(refract_period_s * fs))
    if same_chan_in_buffer_max <= 0:
        same_chan_in_buffer_max = int(2*N)

    in_time_sample, in_chan, in_pattern = in_time_sample.astype(np.int32), in_chan.astype(np.int32), in_pattern.astype(np.int32)
    # Call cython STDP
    if not full_mode:
        out_time_sample, out_chan, out_pattern_id, out_potential, weight, thresh_neu = \
            STDP_v2_cy(in_time_sample, in_chan, in_pattern, np.int32(spike_list.n_spikes), np.int32(N), np.int32(W), np.int32(M), np.int32(P),
                       np.float32(dT), np.int32(n_swap_i), np.int32(d_n_swap), np.int32(min_n_swap), np.int32(T_i), np.int32(T_f),
                       np.int32(T_firing), np.int32(refract_period_samples), weight_init, np.int32(freeze_weight), np.int32(same_chan_in_buffer_max))
    else:
        out_time_sample, out_chan, out_pattern_id, out_potential, weight, thresh_neu, weightset_spk,\
            learn_time, learn_chan, learn_pattern_id, learn_potential, T_all, n_swap_all, weightset_learn = \
            STDP_v2_fullmode_cy(in_time_sample, in_chan, in_pattern, int(spike_list.n_spikes), int(N), int(W), int(M), int(P),
                       np.float32(dT), int(n_swap_i), int(d_n_swap), int(min_n_swap), int(T_i), int(T_f),
                       int(T_firing), int(refract_period_samples), weight_init, freeze_weight, same_chan_in_buffer_max)

    # Re-set to nan the pattern_id when not defined (==-1)
    out_pattern_id[out_pattern_id == -1] = np.nan

    out_time = out_time_sample / fs
    out_spike_list = SpikeList(out_time, out_chan, out_pattern_id, out_potential, spike_list.n_channels,
                               tmin=spike_list.tmin, tmax=spike_list.tmax)
    if full_mode:
        learn_spike_list = SpikeList(learn_time / fs, learn_chan, learn_pattern_id, learn_potential, spike_list.n_channels,
                                     tmin=spike_list.tmin, tmax=spike_list.tmax)

    # Copy the pattern names from input spike list to output spike list
    for pat_id in np.unique(out_spike_list.pattern_id):
        if not pat_id == np.nan:
            out_spike_list.pattern_names[pat_id] = spike_list.pattern_names[pat_id]
            if full_mode:
                learn_spike_list.pattern_names[pat_id] = spike_list.pattern_names[pat_id]

    if full_mode:
        return out_spike_list, weight, thresh_neu, weightset_spk, learn_spike_list, T_all, n_swap_all, weightset_learn
    else:
        return out_spike_list, weight, thresh_neu


def plot_weight_matrix(weight, T=[], Tmin_sel=[]):
    """ Plot the weight matrix in the center axis. Plot on the right the total number of positive weights for each
    input neurons.

    Parameters
    ----------
    weight : array (size M*P) | None
        weight matrix
    T : array (size P) | None
        Output neurons threshold
    Tmin_sel : float
        The weight of output neurons whose threshold is higher (strictly) than ``Tmin_sel`` will be displayed in another
        color in the right and bottom graphs

    Returns
    -------

    """
    T = np.array(T)
    M, P = weight.shape
    f = plt.figure()
    ax_center = plt.subplot2grid((5, 5), (0, 0), rowspan=4, colspan=4)
    ax_center.imshow(weight, aspect='auto', origin='lower', extent=(0, M, 0, P))
    ax_right = plt.subplot2grid((5, 5), (0, 4), rowspan=4, colspan=1, sharey=ax_center)
    ax_right.barh(range(0, M), weight.sum(axis=1), height=1)
    if T.size > 0:
        ax_bottom = plt.subplot2grid((5, 5), (4, 0), rowspan=1, colspan=4, sharex=ax_center)
        ax_bottom.bar(range(P), T, width=1)
        ax_bottom.set(xlabel='Output Neurons', ylabel='Neuron Threshold')
        if Tmin_sel:
            out_neurons_sel = T > Tmin_sel
            weight_sel_sum = weight[:, out_neurons_sel].sum(axis=1)
            ax_bottom.bar(np.where(out_neurons_sel)[0], T[out_neurons_sel], width=1)
            ax_right.barh(range(0, M), weight_sel_sum, height=1)
            ax_right.legend(['All weights', 'T > {}'.format(Tmin_sel)])
    ax_center.autoscale(axis='both', tight=True)
    ax_center.set(title='Weight Matrix', xlabel='Output Neurons', ylabel='Input Neurons')


def save_weights_to_mat(weight, export_path, export_name):
    """ Export the STDP weights to a .mat file

    Parameters
    ----------
    weights : array (size: M*P)
        weights array
    export_path : str
        Export directory path
    export_name : str
        Export filename

    """
    sio.savemat(os.path.join(export_path, export_name), mdict={'weight': weight})


def import_weights_from_mat(import_path, import_name):
    mat_file = sio.loadmat(os.path.join(import_path, import_name))
    try:
        weight_data = mat_file['weight']
    except:
        try:
            weight_data = mat_file[list(mat_file.keys())[-1]]
        except:
            raise ValueError('Could not read weight')
    return weight_data
