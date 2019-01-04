import numpy as np
from cython.parallel import prange
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double exp(double x)

##############################################################
#  - Multi-timescale adaptive threshold for LIF neurons -
#  Adapted from :
#    Kobayashi Ryota, Tsubo Yasuhiro, Shinomoto Shigeru. Made-to-order spiking neuron model equipped with a
#    multi-timescale adaptive threshold. Frontiers in Computational Neuroscience. 2009
##############################################################


@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_threshold(double[:] t_spikes, double t, double delay_max, double[:] tau_j, double[:] alpha_j, double omega):
    cdef int n_spikes = t_spikes.shape[0]
    cdef int n_tau = tau_j.shape[0]
    cdef double threshold = omega
    cdef double delay_i
    cdef Py_ssize_t i, j
    if n_spikes > 0:
        for i in range(n_spikes-1, -1, -1):
            delay_i = t - t_spikes[i]
            if delay_i > delay_max:
                break
            for j in range(n_tau):
                threshold = threshold + alpha_j[j] * exp(- delay_i / tau_j[j])
    return threshold


@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_threshold_multichan_v2(int n_chan, double[:] t_spikes, int[:] chan_spikes, double t,
                             double[:] tau_j, double[:] alpha_j, double[:] omega):
    cdef int n_spikes = t_spikes.shape[0]
    cdef int n_tau = tau_j.shape[0]
    cdef threshold = omega * np.ones(n_chan)
    cdef double[:] threshold_v = threshold
    cdef double delay_max = 5*np.max(tau_j)
    cdef double delay
    cdef Py_ssize_t i, j
    cdef int c
    if n_spikes > 0:
        for i in range(n_spikes-1, -1, -1):
            delay = t - t_spikes[i]
            c = chan_spikes[i]
            if delay > delay_max:
                break
            else:
                for j in range(n_tau):
                    threshold_v[c] += alpha_j[j] * exp(- delay / tau_j[j])
    return threshold_v

@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_threshold_multichan(int n_chan, double[:] t_spikes, int[:] chan_spikes, double t,
                                double[:] h_t_v, int fs, double delay_max, double[:] omega):
    cdef int n_spikes = t_spikes.shape[0]
    cdef threshold = omega * np.ones(n_chan)
    cdef double[:] threshold_v = threshold
    cdef double delay
    cdef Py_ssize_t i
    cdef int c, delay_samp
    if n_spikes > 0:
        for i in range(n_spikes-1, -1, -1):
            delay = t - t_spikes[i]
            delay_samp = int(delay*fs)
            c = chan_spikes[i]
            if delay > delay_max:
                break
            else:
                threshold_v[c] += h_t_v[delay_samp]
    return threshold_v

@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_h_t_coeffs(double delay_max_s, double fs, double[:] tau_j, alpha_j):
    cdef int i_delay_max = int(delay_max_s * fs)
    cdef h_t = np.zeros(i_delay_max, dtype=np.float64)
    cdef double[:] h_t_v = h_t
    cdef double t
    cdef int i, j
    cdef int n_tau = tau_j.shape[0]
    for i in range(i_delay_max):
        t = np.float64(i) / fs
        for j in range(n_tau):
            h_t_v[i] = h_t_v[i] + alpha_j[j] * exp(- t / tau_j[j])
    return h_t_v

#############################################
# LIF functions
# Inhibition models adapted from
#   Gershon G. Furman and Lawrence S. Frishkopf. Model of Neural Inhibition in the Mammalian Cochlea.
#   The Journal of the Acoustical Society of America 1964 36:11, 2194-2201
#############################################

# LIF model for 1 channel
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lif_filter_1d_signal_cy(int fs, double[:] isyn_v, int refract_period, double t_refract, double tau, double v_thresh,
                              double v_spike, double v_reset, double v_init, int adaptive_threshold, double[:] tau_j,
                              double[:] alpha_j, double omega, double t_start=0, t_last_spike_p=[]):
    cdef Py_ssize_t i, n_pnts
    cdef double t
    cdef double dt = 1.0 / fs
    n_pnts = np.array(isyn_v).size
    tvect = np.linspace(t_start, t_start + n_pnts * dt, n_pnts)
    cdef double[:] tvect_v = tvect
    v_mult = (tau / dt) / (tau / dt + 1.0)
    i_mult = 1 / (1 + tau / dt)
    cdef double t_last_spike
    t_last_spike = t_last_spike_p if t_last_spike_p else t_start - 2.0 * t_refract
    v_out = np.zeros(n_pnts)
    cdef double[:] v_out_v = v_out
    t_spikes = np.zeros(n_pnts)
    cdef double[:] t_spikes_v = t_spikes
    cdef Py_ssize_t spike_inc = 0
    #-- ADAPTIVE THRESHOLD
    cdef double[:] threshold_v
    cdef double delay_max_s
    if adaptive_threshold:
        delay_max_s = 5 * max(tau_j)
    else:
        delay_max_s = 0
    cdef h_t = np.zeros(int(fs * delay_max_s))
    cdef double[:] h_t_v = h_t
    cdef double threshold = 0.0
    if adaptive_threshold:
        h_t_v = get_h_t_coeffs(delay_max_s, np.float64(fs), tau_j, alpha_j)
    else:
        threshold = v_thresh

    for i in range(n_pnts):
        t = tvect_v[i]
        if adaptive_threshold:
            threshold_v = get_threshold(t_spikes_v[:spike_inc], t, delay_max_s, tau_j, alpha_j, omega)

        if refract_period and t < (t_last_spike + t_refract):  # Refractory period
            v_out_v[i] = v_reset
        elif not refract_period and i > 0 and t_last_spike == tvect_v[i - 1]:  # Spiking activity just occured
            v_out_v[i] = v_reset
        else:
            if i == 0:
                v_out_v[i] = v_init * v_mult + isyn_v[i - 1] * i_mult
            else:
                v_out_v[i] = v_out_v[i - 1] * v_mult + isyn_v[i - 1] * i_mult
            if v_out_v[i] > threshold:  # Spike
                v_out_v[i] = v_spike
                t_last_spike = t
                t_spikes_v[spike_inc] = t
                spike_inc += 1

    t_spikes_v = t_spikes_v[:spike_inc]
    return np.array(v_out_v), np.array(t_spikes_v)


# LIF model for multiples channels
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lif_filter_cy(int fs, double[:, :] isyn_v, int refract_period, double[:] t_refract, double[:] tau,
                    double[:] v_thresh, double[:] v_spike, double[:] v_reset, double[:] v_init,
                    int adaptive_threshold, double[:] tau_j, double[:] alpha_j, double[:] omega,
                    double t_start=0, t_last_spike_p=[]):
    cdef Py_ssize_t i, c
    cdef double t
    cdef double dt = 1.0 / fs
    cdef Py_ssize_t n_chan = isyn_v.shape[0]
    cdef Py_ssize_t n_pnts = isyn_v.shape[1]
    tvect = np.linspace(t_start, t_start + n_pnts * dt, n_pnts)
    cdef double[:] tvect_v = tvect
    v_mult = (np.array(tau) / float(dt)) / (np.array(tau) / float(dt) + 1.0)
    i_mult = 1 / (1 + np.array(tau) / float(dt))
    cdef double[:] v_mult_v = v_mult
    cdef double[:] i_mult_v = i_mult
    cdef double[:] t_last_spike
    t_last_spike_p = np.array(t_last_spike_p)
    t_last_spike = t_last_spike_p if t_last_spike_p.size > 0 else t_start - 2.0 * np.array(t_refract)
    v_out = np.zeros((n_chan, n_pnts))
    cdef double[:, :] v_out_v = v_out
    cdef int out_list_size = int(0.1*n_pnts*n_chan)
    t_spikes = np.zeros(out_list_size)
    chan_spikes = np.zeros(out_list_size, dtype=np.int32)
    cdef double[:] t_spikes_v = t_spikes
    cdef int[:] chan_spikes_v = chan_spikes
    cdef Py_ssize_t spike_inc = 0
    #-- ADAPTIVE THRESHOLD
    cdef double[:] threshold_v
    cdef double delay_max_s
    if adaptive_threshold:
        delay_max_s = 5 * max(tau_j)
    else:
        delay_max_s = 0
    cdef h_t = np.zeros(int(fs * delay_max_s))
    cdef double[:] h_t_v = h_t
    if adaptive_threshold:
        threshold = np.zeros(n_chan)
        threshold_v = threshold
        h_t_v = get_h_t_coeffs(delay_max_s, np.float64(fs), tau_j, alpha_j)
    else:
        threshold_v = v_thresh


    for i in range(n_pnts):
        t = tvect_v[i]
        if adaptive_threshold:
            threshold_v = get_threshold_multichan(n_chan, t_spikes_v[:spike_inc], chan_spikes_v[:spike_inc],  t,
                                                  h_t_v, fs, delay_max_s, omega)

        for c in range(n_chan):
            if refract_period and t < (t_last_spike[c] + t_refract[c]):  # Refractory period
                v_out_v[c, i] = v_reset[c]
            elif not refract_period and i > 0 and t_last_spike[c] == tvect_v[i - 1]:  # Spiking activity just occured
                v_out_v[c, i] = v_reset[c]
            else:
                if i == 0:
                    v_out_v[c, i] = v_init[c] * v_mult_v[c] + isyn_v[c, i - 1] * i_mult_v[c]
                else:
                    v_out_v[c, i] = v_out_v[c, i-1] * v_mult_v[c] + isyn_v[c, i - 1] * i_mult_v[c]

                if v_out_v[c, i] > threshold_v[c]:  # Spike
                    v_out_v[c, i] = v_spike[c]
                    t_last_spike[c] = t
                    t_spikes_v[spike_inc] = t
                    chan_spikes_v[spike_inc] = c
                    spike_inc += 1
            if spike_inc>=(out_list_size-1):
                raise ValueError('Increase size of t_spikes, chan_spikes')

    t_spikes_v = t_spikes_v[:spike_inc]
    chan_spikes_v = chan_spikes_v[:spike_inc]
    return np.array(v_out_v), np.array(t_spikes_v), np.array(chan_spikes_v)


# LIF model with backward-shunting inhibition
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lif_filter_inhib_shuntfor_current_cy(int fs, double[:, :] isyn_v, int refract_period, double[:] t_refract, double[:] tau,
                    double[:] v_thresh, double[:] v_spike, double[:] v_reset, double[:] v_init, double[:] inhib_vect,
                    int adaptive_threshold, double[:] tau_j, double[:] alpha_j, double[:] omega,
                    double t_start=0, t_last_spike_p=[]):
    print('Inhibition Shunting Forward Current')
    cdef Py_ssize_t i, j, c, c_j
    cdef double t
    cdef double dt = 1.0 / fs
    cdef Py_ssize_t n_chan = isyn_v.shape[0]
    cdef Py_ssize_t n_pnts = isyn_v.shape[1]
    #-- INHIBITION
    cdef Py_ssize_t n_inhib = int(np.array(inhib_vect).size)
    cdef int n_inhib_half = int(np.floor(n_inhib/2))
    cdef double inhib_vect_sum = np.array(inhib_vect).sum()
    inhib_vect_norm = np.zeros(n_inhib)
    inhib_vect_norm_mult = np.zeros(n_chan)
    cdef double[:] inhib_vect_norm_v = inhib_vect_norm
    cdef double[:] inhib_vect_norm_mult_v = inhib_vect_norm_mult
    #--------------
    tvect = np.linspace(t_start, t_start + n_pnts * dt, n_pnts)
    cdef double[:] tvect_v = tvect
    v_mult = (np.array(tau) / float(dt)) / (np.array(tau) / float(dt) + 1.0)
    i_mult = 1 / (1 + np.array(tau) / float(dt))
    cdef double[:] v_mult_v = v_mult
    cdef double[:] i_mult_v = i_mult
    cdef double[:] t_last_spike
    t_last_spike_p = np.array(t_last_spike_p)
    t_last_spike = t_last_spike_p if t_last_spike_p.size > 0 else t_start - 2.0 * np.array(t_refract)
    v_temp = np.zeros(n_chan)
    v_out = np.zeros((n_chan, n_pnts))
    cdef double[:] v_temp_v = v_temp
    cdef double[:, :] v_out_v = v_out
    cdef double v_divide
    cdef int out_list_size = int(0.1*n_pnts*n_chan)
    t_spikes = np.zeros(out_list_size)
    chan_spikes = np.zeros(out_list_size, dtype=np.int32)
    cdef double[:] t_spikes_v = t_spikes
    cdef int[:] chan_spikes_v = chan_spikes
    cdef Py_ssize_t spike_inc = 0
    cdef double inhib_shunt, i_c
    #-- ADAPTIVE THRESHOLD
    cdef double[:] threshold_v
    cdef double delay_max_s
    if adaptive_threshold:
        delay_max_s = 5 * max(tau_j)
    else:
        delay_max_s = 0
    cdef h_t = np.zeros(int(fs * delay_max_s))
    cdef double[:] h_t_v = h_t
    if adaptive_threshold:
        threshold = np.zeros(n_chan)
        threshold_v = threshold
        h_t_v = get_h_t_coeffs(delay_max_s, np.float64(fs), tau_j, alpha_j)
    else:
        threshold_v = v_thresh


    for c in range(n_chan):
        if 0 < c < n_inhib_half:
            inhib_vect_norm_mult[c] = inhib_vect_sum /  np.array(inhib_vect[n_inhib_half-c:n_inhib_half+c+1]).sum()
        elif (n_chan - n_inhib_half) <= c < (n_chan - 1):
            inhib_vect_norm_mult[c] = inhib_vect_sum / np.array(inhib_vect[c+1-n_chan+n_inhib_half:n_chan-1-c+n_inhib_half]).sum()

    for i in range(n_pnts):
        t = tvect_v[i]

        # If adaptive threshold, compute the current threshold for each channel
        if adaptive_threshold:
            threshold_v = get_threshold_multichan(n_chan, t_spikes_v[:spike_inc], chan_spikes_v[:spike_inc],  t,
                                                  h_t_v, fs, delay_max_s, omega)

        for c in range(n_chan):
            if refract_period and t < (t_last_spike[c] + t_refract[c]):  # Refractory period
                v_out_v[c, i] = v_reset[c]
            elif not refract_period and i > 0 and t_last_spike[c] == tvect_v[i - 1]:  # Spiking activity just occured
                v_out_v[c, i] = v_reset[c]
            else:
                # A part of the input current is shunted away through conductance instead of reaching the exictable part
                # of the neuron
                inhib_shunt = 0
                if n_inhib_half <= c < (n_chan - n_inhib_half):
                    for j in range(n_inhib):
                        inhib_shunt += inhib_vect[j] * isyn_v[c - n_inhib_half + j, i-1]
                elif 0 < c < n_inhib_half:
                    inhib_vect_norm_mult_c = inhib_vect_norm_mult_v[c]
                    for c_j in range(0, 2*c+1):
                        inhib_shunt += isyn_v[c_j, i - 1] * inhib_vect[n_inhib_half - c + c_j] * inhib_vect_norm_mult_c
                elif (n_chan - n_inhib_half) <= c < (n_chan-1):
                    inhib_vect_norm_mult_c = inhib_vect_norm_mult_v[c]
                    for c_j in range(2*c+1-n_chan, n_chan):
                        inhib_shunt += isyn_v[c_j, i - 1] * inhib_vect[c_j - c + n_inhib_half] * inhib_vect_norm_mult_c
                elif c==0 or c==(n_chan-1):
                    inhib_shunt = isyn_v[c, i-1]*inhib_vect_sum
                i_c = isyn_v[c, i-1] / (1 + inhib_shunt)
                if i == 0:
                    v_out_v[c, i] = v_init[c] * v_mult_v[c] + i_c * i_mult_v[c]
                else:
                    v_out_v[c, i] = v_out_v[c, i-1] * v_mult_v[c] + i_c * i_mult_v[c]
                if v_out_v[c, i] > threshold_v[c]:  # Spike
                    v_out_v[c, i] = v_spike[c]
                    t_last_spike[c] = t
                    t_spikes_v[spike_inc] = t
                    chan_spikes_v[spike_inc] = c
                    spike_inc += 1
            if spike_inc>=(out_list_size-1):
                raise ValueError('Increase size of t_spikes, chan_spikes')

    t_spikes_v = t_spikes_v[:spike_inc]
    chan_spikes_v = chan_spikes_v[:spike_inc]
    return np.array(v_out_v), np.array(t_spikes_v), np.array(chan_spikes_v)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef lif_filter_inhib_shuntfor_current_mpver_cy(int fs, double[:, :] isyn_v, int refract_period, double[:] t_refract, double[:] tau,
#                     double[:] v_thresh, double[:] v_spike, double[:] v_reset, double[:] v_init, double[:] inhib_vect,
#                     double t_start=0, t_last_spike_p=[]):
#     print('Inhibition Sunting Forward Current - Parralel version')
#     cdef Py_ssize_t i, j, c, c_j
#     cdef double t
#     cdef double dt = 1.0 / fs
#     cdef int n_chan = isyn_v.shape[0]
#     cdef int n_pnts = isyn_v.shape[1]
#     cdef int n_inhib = int(np.array(inhib_vect).size)
#     cdef int n_inhib_half = int(np.floor(n_inhib/2))
#     cdef double inhib_vect_sum = np.array(inhib_vect).sum()
#     inhib_vect_norm = np.zeros(n_inhib)
#     inhib_vect_norm_mult = np.zeros(n_chan)
#     # cdef double[:] inhib_vect_norm_v = inhib_vect_norm
#     cdef double[:] inhib_vect_norm_mult_v = inhib_vect_norm_mult
#     cdef double inhib_vect_norm_mult_c = 0
#     tvect = np.linspace(t_start, t_start + n_pnts * dt, n_pnts)
#     cdef double[:] tvect_v = tvect
#     v_mult = (np.array(tau) / float(dt)) / (np.array(tau) / float(dt) + 1.0)
#     i_mult = 1 / (1 + np.array(tau) / float(dt))
#     cdef double[:] v_mult_v = v_mult
#     cdef double[:] i_mult_v = i_mult
#     cdef double[:] t_last_spike
#     t_last_spike_p = np.array(t_last_spike_p)
#     t_last_spike = t_last_spike_p if t_last_spike_p.size > 0 else t_start - 2.0 * np.array(t_refract)
#     v_temp = np.zeros(n_chan)
#     v_out = np.zeros((n_chan, n_pnts))
#     cdef double[:] v_temp_v = v_temp
#     cdef double[:, :] v_out_v = v_out
#     cdef double v_divide
#     t_spikes = np.zeros(int(0.1*n_pnts*n_chan))
#     chan_spikes = np.zeros(int(0.1*n_pnts*n_chan), dtype=int)
#     cdef double[:] t_spikes_v = t_spikes
#     cdef int[:] chan_spikes_v = chan_spikes
#     cdef Py_ssize_t spike_inc = 0
#     cdef double inhib_shunt, i_c
#     # NEW variables (for prange - 11/06/2018)
#     chan_has_spiked = np.zeros(n_chan, dtype=int)
#     cdef int[:] chan_has_spiked_v = chan_has_spiked
#
#     # For the first and the last channels, the inhibition vector must be adapted
#     for c in range(n_chan):
#         if 0 < c < n_inhib_half:
#             inhib_vect_norm_mult[c] = inhib_vect_sum /  np.array(inhib_vect[n_inhib_half-c:n_inhib_half+c+1]).sum()
#         elif (n_chan - n_inhib_half) <= c < (n_chan - 1):
#             inhib_vect_norm_mult[c] = inhib_vect_sum / np.array(inhib_vect[c+1-n_chan+n_inhib_half:n_chan-1-c+n_inhib_half]).sum()
#
#     for i in range(n_pnts):
#         t = tvect_v[i]
#         with nogil:
#             for c in prange(n_chan):
#                 if refract_period and t < (t_last_spike[c] + t_refract[c]):  # Refractory period
#                     v_out_v[c, i] = v_reset[c]
#                     chan_has_spiked_v[c] = 0
#                 elif not refract_period and i > 0 and t_last_spike[c] == tvect_v[i - 1]:  # Spiking activity just occured
#                     v_out_v[c, i] = v_reset[c]
#                     chan_has_spiked_v[c] = 0
#                 else:
#                     # A part of the input current is shunted away through conductance instead of reaching the exictable part
#                     # of the neuron
#                     inhib_shunt = 0
#                     # "Center" channels
#                     if n_inhib_half <= c < (n_chan - n_inhib_half):
#                         for j in range(n_inhib):
#                             inhib_shunt += inhib_vect[j] * isyn_v[c - n_inhib_half + j, i-1]
#                     # First channels
#                     elif 0 < c < n_inhib_half:
#                         inhib_vect_norm_mult_c = inhib_vect_norm_mult_v[c]
#                         for c_j in range(0, 2*c+1):
#                             inhib_shunt += isyn_v[c_j, i-1] * inhib_vect[n_inhib_half-c+c_j] * inhib_vect_norm_mult_c
#                     # Last channels
#                     elif (n_chan - n_inhib_half) <= c < (n_chan-1):
#                         inhib_vect_norm_mult_c = inhib_vect_norm_mult_v[c]
#                         for c_j in range(2*c+1-n_chan, n_chan):
#                             inhib_shunt += isyn_v[c_j, i-1] * inhib_vect[c_j-c+n_inhib_half] * inhib_vect_norm_mult_c
#                     elif c==0 or c==(n_chan-1):
#                         inhib_shunt = isyn_v[c, i-1]*inhib_vect_sum
#                     i_c = isyn_v[c, i-1] / (1 + inhib_shunt)
#                     if i == 0:
#                         v_out_v[c, i] = v_init[c] * v_mult_v[c] + i_c * i_mult_v[c]
#                     else:
#                         v_out_v[c, i] = v_out_v[c, i-1] * v_mult_v[c] + i_c * i_mult_v[c]
#                     if v_out_v[c, i] > v_thresh[c]:  # Spike
#                         v_out_v[c, i] = v_spike[c]
#                         t_last_spike[c] = t
#                         chan_has_spiked_v[c] = 1
#                     else:
#                         chan_has_spiked_v[c] = 0
#
#         for c in range(n_chan):
#             if chan_has_spiked_v[c]:
#                 t_spikes_v[spike_inc] = t
#                 chan_spikes_v[spike_inc] = c
#                 spike_inc += 1
# if spike_inc >= (out_list_size - 1):
#     raise ValueError('Increase size of t_spikes, chan_spikes')
#
#     t_spikes_v = t_spikes_v[:spike_inc]
#     chan_spikes_v = chan_spikes_v[:spike_inc]
#     return np.array(v_out_v), np.array(t_spikes_v), np.array(chan_spikes_v)
