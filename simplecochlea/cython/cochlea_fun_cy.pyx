import numpy as np
from cython.parallel import prange
cimport numpy as np
cimport cython


# cdef extern from "math.h":
#     double max(double x, double y)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lif_filter_1d_signal_cy(int fs, double[:] isyn_v, int refract_period, double t_refract, double tau, double v_thresh,
                              double v_spike, double v_reset, double v_init=0, double t_start=0, t_last_spike_p=[]):
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

    for i in range(n_pnts):
        t = tvect_v[i]
        if refract_period and t < (t_last_spike + t_refract):  # Refractory period
            v_out_v[i] = v_reset
        elif not refract_period and i > 0 and t_last_spike == tvect_v[i - 1]:  # Spiking activity just occured
            v_out_v[i] = v_reset
        else:
            if i == 0:
                v_out_v[i] = v_init * v_mult + isyn_v[i - 1] * i_mult
            else:
                v_out_v[i] = v_out_v[i - 1] * v_mult + isyn_v[i - 1] * i_mult
            if v_out_v[i] > v_thresh:  # Spike
                v_out_v[i] = v_spike
                t_last_spike = t
                t_spikes_v[spike_inc] = t
                spike_inc += 1

    t_spikes_v = t_spikes_v[:spike_inc]
    return np.array(v_out_v), np.array(t_spikes_v)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lif_filter_cy(int fs, double[:, :] isyn_v, int refract_period, double[:] t_refract, double[:] tau,
                    double[:] v_thresh, double[:] v_spike, double[:] v_reset, double[:] v_init,
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
    t_spikes = np.zeros(int(0.1*n_pnts*n_chan))
    chan_spikes = np.zeros(int(0.1*n_pnts*n_chan), dtype=int)
    cdef double[:] t_spikes_v = t_spikes
    cdef int[:] chan_spikes_v = chan_spikes
    cdef Py_ssize_t spike_inc = 0

    for i in range(n_pnts):
        t = tvect_v[i]
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
                if v_out_v[c, i] > v_thresh[c]:  # Spike
                    v_out_v[c, i] = v_spike[c]
                    t_last_spike[c] = t
                    t_spikes_v[spike_inc] = t
                    chan_spikes_v[spike_inc] = c
                    spike_inc += 1
            if spike_inc>0.1*n_pnts*n_chan:
                raise ValueError('Increase size of t_spikes, chan_spikes')

    t_spikes_v = t_spikes_v[:spike_inc]
    chan_spikes_v = chan_spikes_v[:spike_inc]
    return np.array(v_out_v), np.array(t_spikes_v), np.array(chan_spikes_v)


cpdef lif_filter_inhib_subfor_cy(int fs, double[:, :] isyn_v, int refract_period, double[:] t_refract, double[:] tau,
                    double[:] v_thresh, double[:] v_spike, double[:] v_reset, double[:] v_init, double[:] inhib_vect,
                    double t_start=0, t_last_spike_p=[]):
    cdef Py_ssize_t i, j, c
    cdef double t
    cdef double dt = 1.0 / fs
    cdef Py_ssize_t n_chan = isyn_v.shape[0]
    cdef Py_ssize_t n_pnts = isyn_v.shape[1]
    cdef int n_inhib = int(np.array(inhib_vect).size)
    cdef int n_inhib_half = int(np.floor(n_inhib/2))
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
    cdef double v_inhib
    t_spikes = np.zeros(int(0.1*n_pnts*n_chan))
    chan_spikes = np.zeros(int(0.1*n_pnts*n_chan), dtype=int)
    cdef double[:] t_spikes_v = t_spikes
    cdef int[:] chan_spikes_v = chan_spikes
    cdef Py_ssize_t spike_inc = 0

    for i in range(n_pnts):
        t = tvect_v[i]
        for c in range(n_chan):
            if refract_period and t < (t_last_spike[c] + t_refract[c]):  # Refractory period
                v_out_v[c, i] = v_reset[c]
            elif not refract_period and i > 0 and t_last_spike[c] == tvect_v[i - 1]:  # Spiking activity just occured
                v_out_v[c, i] = v_reset[c]
            else:
                if i == 0:
                    v_temp_v[c] = v_init[c] * v_mult_v[c] + isyn_v[c, i - 1] * i_mult_v[c]
                else:
                    v_temp_v[c] = v_out_v[c, i-1] * v_mult_v[c] + isyn_v[c, i - 1] * i_mult_v[c]
        # Do inhibition
        for c in range(n_chan):
            if (refract_period and t > (t_last_spike[c] + t_refract[c])) or \
                (not refract_period and i>0 and t_last_spike[c] != tvect_v[i - 1]):
                # Inhibition
                # if c >= n_inhib_half and c < (n_chan - n_inhib_half):
                if n_inhib_half <= c < (n_chan - n_inhib_half):
                    v_inhib = 0
                    for j in range(n_inhib):
                        v_inhib += abs(v_temp[c-n_inhib_half+j]) * inhib_vect[j]
                    # if v_inhib > 0:
                    #     print('v_inhib > 0 : {}'.format(v_inhib))
                    v_out[c, i] = v_temp_v[c] + v_inhib
                else:
                    v_out[c, i] = v_temp_v[c]
            if v_out_v[c, i] > v_thresh[c]:  # Spike
                v_out_v[c, i] = v_spike[c]
                t_last_spike[c] = t
                t_spikes_v[spike_inc] = t
                chan_spikes_v[spike_inc] = c
                spike_inc += 1
            if spike_inc>=0.1*n_pnts*n_chan:
                raise ValueError('Increase size of t_spikes, chan_spikes')

    t_spikes_v = t_spikes_v[:spike_inc]
    chan_spikes_v = chan_spikes_v[:spike_inc]
    return np.array(v_out_v), np.array(t_spikes_v), np.array(chan_spikes_v)


cpdef lif_filter_inhib_shuntfor_cy(int fs, double[:, :] isyn_v, int refract_period, double[:] t_refract, double[:] tau,
                    double[:] v_thresh, double[:] v_spike, double[:] v_reset, double[:] v_init, double[:] inhib_vect,
                    double t_start=0, t_last_spike_p=[]):
    cdef Py_ssize_t i, j, c, c_j
    cdef double t
    cdef double dt = 1.0 / fs
    cdef Py_ssize_t n_chan = isyn_v.shape[0]
    cdef Py_ssize_t n_pnts = isyn_v.shape[1]
    cdef Py_ssize_t n_inhib = int(np.array(inhib_vect).size)
    cdef int n_inhib_half = int(np.floor(n_inhib/2))
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
    t_spikes = np.zeros(int(0.1*n_pnts*n_chan))
    chan_spikes = np.zeros(int(0.1*n_pnts*n_chan), dtype=int)
    cdef double[:] t_spikes_v = t_spikes
    cdef int[:] chan_spikes_v = chan_spikes
    cdef Py_ssize_t spike_inc = 0

    for i in range(n_pnts):
        t = tvect_v[i]
        for c in range(n_chan):
            if refract_period and t < (t_last_spike[c] + t_refract[c]):  # Refractory period
                v_out_v[c, i] = v_reset[c]
            elif not refract_period and i > 0 and t_last_spike[c] == tvect_v[i - 1]:  # Spiking activity just occured
                v_out_v[c, i] = v_reset[c]
            else:
                if i == 0:
                    v_temp_v[c] = v_init[c] * v_mult_v[c] + isyn_v[c, i - 1] * i_mult_v[c]
                else:
                    v_temp_v[c] = v_out_v[c, i-1] * v_mult_v[c] + isyn_v[c, i - 1] * i_mult_v[c]
        # Do inhibition
        for c in range(n_chan):
            if (refract_period and t > (t_last_spike[c] + t_refract[c])) or \
                (not refract_period and i>0 and t_last_spike[c] != tvect_v[i - 1]):
                # Inhibition
                v_divide = 1
                if n_inhib_half <= c < (n_chan - n_inhib_half):
                    for j in range(n_inhib):
                        v_divide += inhib_vect[j] * v_temp[c - n_inhib_half + j]
                elif 0 < c < n_inhib_half:
                    for c_j in range(0, 2*c+1):
                        v_divide += v_temp[c_j] * inhib_vect[n_inhib_half-c+c_j]
                elif (n_chan - n_inhib_half) <= c < (n_inhib_half-1):
                    for c_j in range(2*c+1-n_chan, n_chan):
                        v_divide += v_temp[c_j] * inhib_vect[n_chan-c_j]
                else:
                    v_divide = 1
                v_out[c, i] = v_temp_v[c] / v_divide
            if v_out_v[c, i] > v_thresh[c]:  # Spike
                v_out_v[c, i] = v_spike[c]
                t_last_spike[c] = t
                t_spikes_v[spike_inc] = t
                chan_spikes_v[spike_inc] = c
                spike_inc += 1
            if spike_inc>=0.1*n_pnts*n_chan:
                raise ValueError('Increase size of t_spikes, chan_spikes')

    t_spikes_v = t_spikes_v[:spike_inc]
    chan_spikes_v = chan_spikes_v[:spike_inc]
    return np.array(v_out_v), np.array(t_spikes_v), np.array(chan_spikes_v)


cpdef lif_filter_spike_inhib_cy(int fs, double[:, :] isyn_v, int refract_period, double[:] t_refract, double[:] tau,
                    double[:] v_thresh, double[:] v_spike, double[:] v_reset, double[:] v_init, double[:] inhib_vect,
                    double t_start=0, t_last_spike_p=[]):
    cdef Py_ssize_t i, j, c
    cdef double t
    cdef double dt = 1.0 / fs
    cdef Py_ssize_t n_chan = isyn_v.shape[0]
    cdef Py_ssize_t n_pnts = isyn_v.shape[1]
    cdef int n_inhib = int(np.array(inhib_vect).size)
    cdef int n_inhib_half = int(np.floor(n_inhib/2))
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
    t_spikes = np.zeros(int(0.1*n_pnts*n_chan))
    chan_spikes = np.zeros(int(0.1*n_pnts*n_chan), dtype=int)
    cdef double[:] t_spikes_v = t_spikes
    cdef int[:] chan_spikes_v = chan_spikes
    cdef Py_ssize_t spike_inc = 0
    cdef int spike_inc_last = 0

    for i in range(n_pnts):
        t = tvect_v[i]
        for c in range(n_chan):
            if refract_period and t < (t_last_spike[c] + t_refract[c]):  # Refractory period
                v_out_v[c, i] = v_reset[c]
            elif not refract_period and i > 0 and t_last_spike[c] == tvect_v[i - 1]:  # Spiking activity just occured
                v_out_v[c, i] = v_reset[c]
            else:
                if i == 0:
                    v_out_v[c, i] = v_init[c] * v_mult_v[c] + isyn_v[c, i] * i_mult_v[c]
                else:
                    v_out_v[c, i] = v_out_v[c, i-1] * v_mult_v[c] + isyn_v[c, i - 1] * i_mult_v[c]
                if v_out_v[c, i] > v_thresh[c]:  # Spike
                    v_out_v[c, i] = v_spike[c]
                    t_last_spike[c] = t
                    t_spikes_v[spike_inc] = t
                    chan_spikes_v[spike_inc] = c
                    spike_inc += 1
            if spike_inc>0.1*n_pnts*n_chan:
                raise ValueError('Increase size of t_spikes, chan_spikes')
        # Go through each channel that has just spiked and do the inhibition :
        for c in chan_spikes_v[spike_inc_last:spike_inc]:
            if n_inhib_half <= c < (n_chan - n_inhib_half):
                for j in range(n_inhib):
                    v_out_v[c-n_inhib_half+j, i] += inhib_vect[j]
        spike_inc_last = spike_inc

    t_spikes_v = t_spikes_v[:spike_inc]
    chan_spikes_v = chan_spikes_v[:spike_inc]
    return np.array(v_out_v), np.array(t_spikes_v), np.array(chan_spikes_v)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lif_filter_inhib_shuntfor_current_cy(int fs, double[:, :] isyn_v, int refract_period, double[:] t_refract, double[:] tau,
                    double[:] v_thresh, double[:] v_spike, double[:] v_reset, double[:] v_init, double[:] inhib_vect,
                    double t_start=0, t_last_spike_p=[]):
    print('Inhibition Sunting Forward Current')
    cdef Py_ssize_t i, j, c, c_j
    cdef double t
    cdef double dt = 1.0 / fs
    cdef Py_ssize_t n_chan = isyn_v.shape[0]
    cdef Py_ssize_t n_pnts = isyn_v.shape[1]
    cdef Py_ssize_t n_inhib = int(np.array(inhib_vect).size)
    cdef int n_inhib_half = int(np.floor(n_inhib/2))
    cdef double inhib_vect_sum = np.array(inhib_vect).sum()
    inhib_vect_norm = np.zeros(n_inhib)
    inhib_vect_norm_mult = np.zeros(n_chan)
    cdef double[:] inhib_vect_norm_v = inhib_vect_norm
    cdef double[:] inhib_vect_norm_mult_v = inhib_vect_norm_mult
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
    t_spikes = np.zeros(int(0.1*n_pnts*n_chan))
    chan_spikes = np.zeros(int(0.1*n_pnts*n_chan), dtype=int)
    cdef double[:] t_spikes_v = t_spikes
    cdef int[:] chan_spikes_v = chan_spikes
    cdef Py_ssize_t spike_inc = 0
    cdef double inhib_shunt, i_c

    for c in range(n_chan):
        if 0 < c < n_inhib_half:
            inhib_vect_norm_mult[c] = inhib_vect_sum /  np.array(inhib_vect[n_inhib_half-c:n_inhib_half+c+1]).sum()
        elif (n_chan - n_inhib_half) <= c < (n_chan - 1):
            inhib_vect_norm_mult[c] = inhib_vect_sum / np.array(inhib_vect[c+1-n_chan+n_inhib_half:n_chan-1-c+n_inhib_half]).sum()

    for i in range(n_pnts):
        t = tvect_v[i]
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
                    # inhib_vect_norm_v = inhib_vect * inhib_vect_norm_mult[c]
                    for c_j in range(0, 2*c+1):
                        inhib_shunt += isyn_v[c_j, i - 1] * inhib_vect[n_inhib_half - c + c_j] * inhib_vect_norm_mult_c
                        # inhib_shunt += isyn_v[c_j, i-1] * inhib_vect_norm_v[n_inhib_half-c+c_j]
                elif (n_chan - n_inhib_half) <= c < (n_chan-1):
                    inhib_vect_norm_mult_c = inhib_vect_norm_mult_v[c]
                    # inhib_vect_norm_v = inhib_vect * inhib_vect_norm_mult[c]
                    for c_j in range(2*c+1-n_chan, n_chan):
                        inhib_shunt += isyn_v[c_j, i - 1] * inhib_vect[c_j - c + n_inhib_half] * inhib_vect_norm_mult_c
                        # inhib_shunt += isyn_v[c_j, i-1] * inhib_vect_norm_v[c_j-c+n_inhib_half]
                elif c==0 or c==(n_chan-1):
                    inhib_shunt = isyn_v[c, i-1]*inhib_vect_sum
                i_c = isyn_v[c, i-1] / (1 + inhib_shunt)
                if i == 0:
                    v_out_v[c, i] = v_init[c] * v_mult_v[c] + i_c * i_mult_v[c]
                else:
                    v_out_v[c, i] = v_out_v[c, i-1] * v_mult_v[c] + i_c * i_mult_v[c]
                if v_out_v[c, i] > v_thresh[c]:  # Spike
                    v_out_v[c, i] = v_spike[c]
                    t_last_spike[c] = t
                    t_spikes_v[spike_inc] = t
                    chan_spikes_v[spike_inc] = c
                    spike_inc += 1
            if spike_inc>int(0.1*n_pnts*n_chan):
                raise ValueError('Increase size of t_spikes, chan_spikes')

    t_spikes_v = t_spikes_v[:spike_inc]
    chan_spikes_v = chan_spikes_v[:spike_inc]
    return np.array(v_out_v), np.array(t_spikes_v), np.array(chan_spikes_v)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lif_filter_inhib_shuntfor_current_mpver_cy(int fs, double[:, :] isyn_v, int refract_period, double[:] t_refract, double[:] tau,
                    double[:] v_thresh, double[:] v_spike, double[:] v_reset, double[:] v_init, double[:] inhib_vect,
                    double t_start=0, t_last_spike_p=[]):
    print('Inhibition Sunting Forward Current - Parralel version')
    cdef Py_ssize_t i, j, c, c_j
    cdef double t
    cdef double dt = 1.0 / fs
    cdef int n_chan = isyn_v.shape[0]
    cdef int n_pnts = isyn_v.shape[1]
    cdef int n_inhib = int(np.array(inhib_vect).size)
    cdef int n_inhib_half = int(np.floor(n_inhib/2))
    cdef double inhib_vect_sum = np.array(inhib_vect).sum()
    inhib_vect_norm = np.zeros(n_inhib)
    inhib_vect_norm_mult = np.zeros(n_chan)
    # cdef double[:] inhib_vect_norm_v = inhib_vect_norm
    cdef double[:] inhib_vect_norm_mult_v = inhib_vect_norm_mult
    cdef double inhib_vect_norm_mult_c = 0
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
    t_spikes = np.zeros(int(0.1*n_pnts*n_chan))
    chan_spikes = np.zeros(int(0.1*n_pnts*n_chan), dtype=int)
    cdef double[:] t_spikes_v = t_spikes
    cdef int[:] chan_spikes_v = chan_spikes
    cdef Py_ssize_t spike_inc = 0
    cdef double inhib_shunt, i_c
    # NEW variables (for prange - 11/06/2018)
    chan_has_spiked = np.zeros(n_chan, dtype=int)
    cdef int[:] chan_has_spiked_v = chan_has_spiked

    # For the first and the last channels, the inhibition vector must be adapted
    for c in range(n_chan):
        if 0 < c < n_inhib_half:
            inhib_vect_norm_mult[c] = inhib_vect_sum /  np.array(inhib_vect[n_inhib_half-c:n_inhib_half+c+1]).sum()
        elif (n_chan - n_inhib_half) <= c < (n_chan - 1):
            inhib_vect_norm_mult[c] = inhib_vect_sum / np.array(inhib_vect[c+1-n_chan+n_inhib_half:n_chan-1-c+n_inhib_half]).sum()

    for i in range(n_pnts):
        t = tvect_v[i]
        with nogil:
            for c in prange(n_chan):
                if refract_period and t < (t_last_spike[c] + t_refract[c]):  # Refractory period
                    v_out_v[c, i] = v_reset[c]
                    chan_has_spiked_v[c] = 0
                elif not refract_period and i > 0 and t_last_spike[c] == tvect_v[i - 1]:  # Spiking activity just occured
                    v_out_v[c, i] = v_reset[c]
                    chan_has_spiked_v[c] = 0
                else:
                    # A part of the input current is shunted away through conductance instead of reaching the exictable part
                    # of the neuron
                    inhib_shunt = 0
                    # "Center" channels
                    if n_inhib_half <= c < (n_chan - n_inhib_half):
                        for j in range(n_inhib):
                            inhib_shunt += inhib_vect[j] * isyn_v[c - n_inhib_half + j, i-1]
                    # First channels
                    elif 0 < c < n_inhib_half:
                        inhib_vect_norm_mult_c = inhib_vect_norm_mult_v[c]
                        # inhib_vect_norm_v = inhib_vect * inhib_vect_norm_mult_v[c]
                        for c_j in range(0, 2*c+1):
                            inhib_shunt += isyn_v[c_j, i-1] * inhib_vect[n_inhib_half-c+c_j] * inhib_vect_norm_mult_c
                    # Last channels
                    elif (n_chan - n_inhib_half) <= c < (n_chan-1):
                        inhib_vect_norm_mult_c = inhib_vect_norm_mult_v[c]
                        # inhib_vect_norm_v = inhib_vect * inhib_vect_norm_mult_v[c]
                        for c_j in range(2*c+1-n_chan, n_chan):
                            inhib_shunt += isyn_v[c_j, i-1] * inhib_vect[c_j-c+n_inhib_half] * inhib_vect_norm_mult_c
                    elif c==0 or c==(n_chan-1):
                        inhib_shunt = isyn_v[c, i-1]*inhib_vect_sum
                    i_c = isyn_v[c, i-1] / (1 + inhib_shunt)
                    if i == 0:
                        v_out_v[c, i] = v_init[c] * v_mult_v[c] + i_c * i_mult_v[c]
                    else:
                        v_out_v[c, i] = v_out_v[c, i-1] * v_mult_v[c] + i_c * i_mult_v[c]
                    if v_out_v[c, i] > v_thresh[c]:  # Spike
                        v_out_v[c, i] = v_spike[c]
                        t_last_spike[c] = t
                        chan_has_spiked_v[c] = 1
                    else:
                        chan_has_spiked_v[c] = 0

        for c in range(n_chan):
            if chan_has_spiked_v[c]:
                t_spikes_v[spike_inc] = t
                chan_spikes_v[spike_inc] = c
                spike_inc += 1
            # if spike_inc>0.1*n_pnts*n_chan:
            #     raise ValueError('Increase size of t_spikes, chan_spikes')

    t_spikes_v = t_spikes_v[:spike_inc]
    chan_spikes_v = chan_spikes_v[:spike_inc]
    return np.array(v_out_v), np.array(t_spikes_v), np.array(chan_spikes_v)


