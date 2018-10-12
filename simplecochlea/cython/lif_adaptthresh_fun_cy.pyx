import numpy as np
from cython.parallel import prange
cimport numpy as np
cimport cython

# cdef extern from "math.h":
#     double max(double x, double y)
#
# cdef extern from "math.h":
#     int min(int x, int y)

cdef extern from "math.h":
    double exp(double x)


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
cdef get_threshold_multichan(int n_chan, double[:] t_spikes, int[:] chan_spikes, double t, double delay_max,
                             double[:] tau_j, double[:] alpha_j, double[:] omega):
    cdef int n_spikes = t_spikes.shape[0]
    cdef int n_tau = tau_j.shape[0]
    cdef threshold = omega * np.ones(n_chan)
    cdef double[:] threshold_v = threshold
    cdef double delay_i
    cdef int c
    cdef Py_ssize_t i, j
    if n_spikes > 0:
        for i in range(n_spikes-1, -1, -1):
            delay_i = t - t_spikes[i]
            c = chan_spikes[i]
            if delay_i > delay_max:
                break
            for j in range(n_tau):
                threshold_v[c] = threshold_v[c] + alpha_j[j] * exp(- delay_i / tau_j[j])
    return threshold_v


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lif_filter_adaptive_threshold_cy(int fs, double[:, :] isyn_v, int refract_period, double[:] t_refract, double[:] tau,
                    double[:] v_spike, double[:] v_reset, double[:] v_init, double[:] tau_j,
                    double[:] alpha_j, double[:] omega, double t_start=0, t_last_spike_p=[]):
    cdef Py_ssize_t i, c
    cdef double t
    cdef double dt = 1.0 / fs
    cdef int n_chan = isyn_v.shape[0]
    cdef int n_pnts = isyn_v.shape[1]
    tvect = np.linspace(t_start, t_start + n_pnts * dt, n_pnts)
    cdef double[:] tvect_v = tvect
    v_mult = (np.array(tau) / float(dt)) / (np.array(tau) / float(dt) + 1.0)
    i_mult = 1 / (1 + np.array(tau) / float(dt))
    cdef double[:] v_mult_v = v_mult
    cdef double[:] i_mult_v = i_mult
    cdef double[:] t_last_spike
    t_last_spike_p = np.array(t_last_spike_p)
    t_last_spike = t_last_spike_p if t_last_spike_p.size > 0 else t_start - 1.0 - 2.0 * np.array(t_refract)
    v_out = np.zeros((n_chan, n_pnts))
    cdef double[:, :] v_out_v = v_out
    t_spikes = np.zeros(int(0.1*n_pnts*n_chan))
    chan_spikes = np.zeros(int(0.1*n_pnts*n_chan), dtype=int)
    cdef double[:] t_spikes_v = t_spikes
    cdef int[:] chan_spikes_v = chan_spikes
    cdef int spike_inc = 0
    # Adaptive threshold
    cdef double delay_max = np.max(tau_j)
    threshold = np.zeros(n_chan)
    cdef double[:] threshold_v = threshold

    for i in range(n_pnts):
        t = tvect_v[i]
        threshold_v = get_threshold_multichan(n_chan, t_spikes_v[:spike_inc], chan_spikes_v[:spike_inc], t, delay_max,
                                                    tau_j, alpha_j, omega)
        for c in range(n_chan):
            if i == 0:
                v_out_v[c, i] = v_init[c] * v_mult_v[c] + isyn_v[c, i - 1] * i_mult_v[c]
            else:
                v_out_v[c, i] = v_out_v[c, i-1] * v_mult_v[c] + isyn_v[c, i - 1] * i_mult_v[c]
            if v_out_v[c, i] > threshold_v[c]:  # Spike
                if refract_period and (t-t_last_spike[c]) < t_refract[c]:
                    continue
                else:
                    t_last_spike[c] = t
                    t_spikes_v[spike_inc] = t
                    chan_spikes_v[spike_inc] = c
                    spike_inc += 1
            if spike_inc>0.1*n_pnts*n_chan:
                raise ValueError('Increase size of t_spikes, chan_spikes')

    t_spikes_v = t_spikes_v[:spike_inc]
    chan_spikes_v = chan_spikes_v[:spike_inc]
    return np.array(v_out_v), np.array(t_spikes_v), np.array(chan_spikes_v), np.array(threshold_v)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lif_filter_one_channel_adaptive_threshold_cy(
    int fs, double[:] isyn_v, int refract_period, double t_refract, double tau,
    double v_spike, double v_reset, double v_init, double[:] tau_j,
    double[:] alpha_j, double omega, double t_start=0, t_last_spike_p=[]):

    cdef Py_ssize_t i
    cdef int n_pnts
    cdef double t
    cdef double dt = 1.0 / fs
    n_pnts = np.array(isyn_v).size
    tvect = np.linspace(t_start, t_start + n_pnts * dt, n_pnts)
    cdef double[:] tvect_v = tvect
    v_mult = (tau / dt) / (tau / dt + 1.0)
    i_mult = 1 / (1 + tau / dt)
    cdef double t_last_spike
    t_last_spike = t_last_spike_p if t_last_spike_p else t_start - 1.0 - 2.0 * t_refract
    v_out = np.zeros(n_pnts)
    cdef double[:] v_out_v = v_out
    t_spikes = np.zeros(n_pnts)
    cdef double[:] t_spikes_v = t_spikes
    cdef int spike_inc = 0
    # Adaptive threshold
    cdef double delay_max = np.max(tau_j)
    threshold = np.zeros(n_pnts)
    cdef double[:] threshold_v = threshold

    for i in range(n_pnts):
        t = tvect_v[i]
        threshold_v[i] = get_threshold(t_spikes_v[:spike_inc], t, delay_max, tau_j, alpha_j, omega)
        if i == 0:
            v_out_v[i] = v_init * v_mult + isyn_v[i - 1] * i_mult
        else:
            v_out_v[i] = v_out_v[i - 1] * v_mult + isyn_v[i - 1] * i_mult
        if v_out_v[i] > threshold_v[i]:  # Spike
            if refract_period and (t - t_last_spike) < t_refract:
                continue
            else:
                t_last_spike = t
                t_spikes_v[spike_inc] = t
                spike_inc += 1

    t_spikes_v = t_spikes_v[:spike_inc]
    return np.array(v_out_v), np.array(t_spikes_v), np.array(threshold_v)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_threshold_multichan_samplever(int n_chan, int[:] i_spikes, int[:] chan_spikes, int i_t, int i_delay_max,
                                       double[:] h_t, double[:] omega):
    cdef int n_spikes = i_spikes.shape[0]
    cdef threshold = omega * np.ones(n_chan)
    cdef double[:] threshold_v = threshold
    cdef int delay_i
    cdef Py_ssize_t i
    cdef int c
    if n_spikes > 0:
        for i in range(n_spikes-1, -1, -1):
            delay_i = i_t - i_spikes[i]
            c = chan_spikes[i]
            if delay_i > i_delay_max:
                break
            else:
                threshold_v[c] = threshold_v[c] + h_t[delay_i]
    return threshold_v


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lif_filter_adaptive_threshold_samplever_cy(int fs, double[:, :] isyn_v, int refract_period, double[:] t_refract, double[:] tau,
                    double[:] v_spike, double[:] v_reset, double[:] v_init, double[:] tau_j,
                    double[:] alpha_j, double[:] omega, double t_start=0, t_last_spike_p=[]):
    cdef Py_ssize_t i, j, c
    cdef double t
    cdef double dt = 1.0 / fs
    cdef int n_chan = isyn_v.shape[0]
    cdef int n_pnts = isyn_v.shape[1]
    tvect = np.linspace(t_start, t_start + n_pnts * dt, n_pnts)
    cdef double[:] tvect_v = tvect
    v_mult = (np.array(tau) / float(dt)) / (np.array(tau) / float(dt) + 1.0)
    i_mult = 1 / (1 + np.array(tau) / float(dt))
    cdef double[:] v_mult_v = v_mult
    cdef double[:] i_mult_v = i_mult
    cdef double[:] t_last_spike
    t_last_spike_p = np.array(t_last_spike_p)
    t_last_spike = t_last_spike_p if t_last_spike_p.size > 0 else t_start - 1.0 - 2.0 * np.array(t_refract)
    v_out = np.zeros((n_chan, n_pnts))
    cdef double[:, :] v_out_v = v_out
    t_spikes = np.zeros(int(0.1*n_pnts*n_chan))
    chan_spikes = np.zeros(int(0.1*n_pnts*n_chan), dtype=int)
    cdef double[:] t_spikes_v = t_spikes
    cdef int[:] chan_spikes_v = chan_spikes
    cdef int spike_inc = 0
    # Adaptive threshold
    cdef int n_tau_j = tau_j.size
    cdef double delay_max = 5 * np.max(tau_j)
    threshold = np.zeros(n_chan)
    cdef double[:] threshold_v = threshold
    cdef i_spikes = np.zeros(int(0.1*n_pnts*n_chan), dtype=int)
    cdef int[:] i_spikes_v = i_spikes
    cdef int i_delay_max = int(delay_max*fs)
    cdef h_t = np.zeros(i_delay_max)
    cdef double[:] h_t_v = h_t
    for i in range(min(i_delay_max, n_pnts)):
        t = tvect_v[i]
        for j in range(n_tau_j):
            h_t_v[i] = h_t_v[i] + alpha_j[j] * exp(- t / tau_j[j])

    for i in range(n_pnts):
        t = tvect_v[i]
        threshold_v = get_threshold_multichan_samplever(n_chan, i_spikes_v[:spike_inc], chan_spikes_v[:spike_inc], i, i_delay_max,
                                                        h_t_v, omega)
        for c in range(n_chan):
            if i == 0:
                v_out_v[c, i] = v_init[c] * v_mult_v[c] + isyn_v[c, i - 1] * i_mult_v[c]
            else:
                v_out_v[c, i] = v_out_v[c, i-1] * v_mult_v[c] + isyn_v[c, i - 1] * i_mult_v[c]
            if v_out_v[c, i] > threshold_v[c]:  # Spike
                if refract_period and (t-t_last_spike[c]) < t_refract[c]:
                    continue
                else:
                    t_last_spike[c] = t
                    t_spikes_v[spike_inc] = t
                    i_spikes_v[spike_inc] = i
                    chan_spikes_v[spike_inc] = c
                    spike_inc += 1
            if spike_inc>0.1*n_pnts*n_chan:
                raise ValueError('Increase size of t_spikes, chan_spikes')

    t_spikes_v = t_spikes_v[:spike_inc]
    chan_spikes_v = chan_spikes_v[:spike_inc]
    return np.array(v_out_v), np.array(t_spikes_v), np.array(chan_spikes_v), np.array(threshold_v)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lif_filter_adaptive_threshold_samplever_inhib_shuntfor_current_cy(
    int fs, double[:, :] isyn_v, int refract_period, double[:] t_refract, double[:] tau, double[:] v_spike,
    double[:] v_reset, double[:] v_init, double[:] tau_j, double[:] alpha_j, double[:] omega, double[:] inhib_vect,
    double t_start=0, t_last_spike_p=[]):

    print('Inhibition Shunting Forward Current with Adaptive threshold')
    cdef Py_ssize_t i, c
    cdef double t
    cdef double dt = 1.0 / fs
    cdef int n_chan = isyn_v.shape[0]
    cdef int n_pnts = isyn_v.shape[1]
    tvect = np.linspace(t_start, t_start + n_pnts * dt, n_pnts)
    cdef double[:] tvect_v = tvect
    v_mult = (np.array(tau) / float(dt)) / (np.array(tau) / float(dt) + 1.0)
    i_mult = 1 / (1 + np.array(tau) / float(dt))
    cdef double[:] v_mult_v = v_mult
    cdef double[:] i_mult_v = i_mult
    cdef double[:] t_last_spike
    t_last_spike_p = np.array(t_last_spike_p)
    t_last_spike = t_last_spike_p if t_last_spike_p.size > 0 else t_start - 1.0 - 2.0 * np.array(t_refract)
    v_out = np.zeros((n_chan, n_pnts))
    cdef double[:, :] v_out_v = v_out
    t_spikes = np.zeros(int(0.1*n_pnts*n_chan))
    chan_spikes = np.zeros(int(0.1*n_pnts*n_chan), dtype=int)
    cdef double[:] t_spikes_v = t_spikes
    cdef int[:] chan_spikes_v = chan_spikes
    cdef int spike_inc = 0

    # Inhibition
    cdef int n_inhib = int(np.array(inhib_vect).size)
    cdef int n_inhib_half = int(np.floor(n_inhib/2))
    cdef double inhib_vect_sum = np.array(inhib_vect).sum()
    inhib_vect_norm = np.zeros(n_inhib)
    inhib_vect_norm_mult = np.zeros(n_chan)
    cdef double[:] inhib_vect_norm_v = inhib_vect_norm
    cdef double[:] inhib_vect_norm_mult_v = inhib_vect_norm_mult
    cdef double inhib_shunt, i_c
    cdef int c_j

    # Adaptive threshold
    cdef int n_tau_j = tau_j.size
    cdef Py_ssize_t j
    cdef double delay_max = 5 * np.max(tau_j)
    threshold = np.zeros(n_chan)
    cdef double[:] threshold_v = threshold
    cdef i_spikes = np.zeros(int(0.1*n_pnts*n_chan), dtype=int)
    cdef int[:] i_spikes_v = i_spikes
    cdef int i_delay_max = int(delay_max*fs)
    cdef h_t = np.zeros(i_delay_max)
    cdef double[:] h_t_v = h_t

    for c in range(n_chan):
        if 0 < c < n_inhib_half:
            inhib_vect_norm_mult[c] = inhib_vect_sum / np.array(inhib_vect[n_inhib_half - c:n_inhib_half + c + 1]).sum()
        elif (n_chan - n_inhib_half) <= c < (n_chan - 1):
            inhib_vect_norm_mult[c] = inhib_vect_sum / np.array(
                inhib_vect[c + 1 - n_chan + n_inhib_half:n_chan - 1 - c + n_inhib_half]).sum()

    for i in range(min(n_pnts, i_delay_max)):
        t = tvect_v[i]
        for j in range(n_tau_j):
            h_t_v[i] = h_t_v[i] + alpha_j[j] * exp(- t / tau_j[j])

    for i in range(n_pnts):
        t = tvect_v[i]
        threshold_v = get_threshold_multichan_samplever(n_chan, i_spikes_v[:spike_inc], chan_spikes_v[:spike_inc], i, i_delay_max,
                                                        h_t_v, omega)
        for c in range(n_chan):
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
                if refract_period and (t-t_last_spike[c]) < t_refract[c]:
                    continue
                else:
                    t_last_spike[c] = t
                    t_spikes_v[spike_inc] = t
                    i_spikes_v[spike_inc] = i
                    chan_spikes_v[spike_inc] = c
                    spike_inc += 1
            if spike_inc>0.1*n_pnts*n_chan:
                raise ValueError('Increase size of t_spikes, chan_spikes')

    t_spikes_v = t_spikes_v[:spike_inc]
    chan_spikes_v = chan_spikes_v[:spike_inc]
    return np.array(v_out_v), np.array(t_spikes_v), np.array(chan_spikes_v), np.array(threshold_v)
