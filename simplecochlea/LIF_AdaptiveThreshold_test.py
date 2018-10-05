import numpy as np
import matplotlib.pyplot as plt
import generate_signals
import seaborn as sns
sns.set()


def lif_standard(fs, refract_period, isyn, tau, v_thresh, v_spike, t_refract, v_reset, v_init=0, t_start=0, t_last_spike=[]):
    dt = 1 / fs
    tvect = np.linspace(t_start, t_start + len(isyn) * dt, len(isyn))
    v_mult = (tau / dt) / (tau / dt + 1)
    i_mult = 1 / (1 + tau / dt)
    if not t_last_spike:
        t_last_spike = -2 * (0.2 + t_start + t_refract)
    t_spikes = []
    v_out = np.zeros(len(isyn))
    for i, t in enumerate(tvect):
        if refract_period and t < (t_last_spike + t_refract):  # Refractory period
            v_out[i] = 0
        elif not refract_period and i > 0 and t_last_spike == tvect[i - 1]:  # Spiking activity just occured
            v_out[i] = v_reset
        else:
            if i == 0:
                v_out[i] = v_init * v_mult + isyn[i - 1] * i_mult
            else:
                v_out[i] = v_out[i - 1] * v_mult + isyn[i - 1] * i_mult
            if v_out[i] > v_thresh:  # Spike
                v_out[i] = v_spike
                t_last_spike = t
                t_spikes.append(t)
    return v_out, t_spikes


def lif_mat_threshold(t, t_spikes, tau_j, alpha, w):
    tau_j, alpha, t_spikes = np.atleast_1d(tau_j), np.atleast_1d(alpha), np.atleast_1d(t_spikes)
    tau_j_max = 3*tau_j.max()
    t_spikes_sel = t_spikes[t_spikes > (t-tau_j_max)]
    if not tau_j.size == alpha.size:
        raise ValueError('tau_j and alpha arguments must have the same size')
    if t_spikes_sel.size == 0:
        v_thresh = w
    else:
        v_thresh = w
        for delta_t in t-t_spikes_sel:
            v_thresh += np.sum(alpha * np.exp(-delta_t / tau_j))
    return v_thresh


def lif_mat(fs, isyn, refract_period, t_refract, tau_m, tau_j, alpha, w, v_init=0, t_start=0, v_spike=0.2, v_reset=0):
    dt = 1 / fs
    tvect = np.linspace(t_start, t_start + len(isyn) * dt, len(isyn))
    v_mult = (tau_m / dt) / (tau_m / dt + 1)
    i_mult = 1 / (1 + tau_m / dt)
    t_spikes, i_spikes, v_out, v_thresh = [], [], np.zeros(len(isyn)), np.zeros(len(isyn))
    t_last_spike = -1-2*t_start

    for i, t in enumerate(tvect):
        v_thresh[i] = lif_mat_threshold(t, t_spikes, tau_j, alpha, w)
        if i == 0:
            v_out[i] = v_init * v_mult + isyn[i - 1] * i_mult
        else:
            v_out[i] = v_out[i - 1] * v_mult + isyn[i - 1] * i_mult
        if v_out[i] > v_thresh[i]:  # Spike
            if refract_period and (t-t_last_spike) < t_refract:
                continue
            t_last_spike = t
            t_spikes.append(t)
            i_spikes.append(i)
    i_spikes = np.array(i_spikes)
    # Modify the potential for each spike
    if i_spikes.any():
        v_out[i_spikes] = v_spike
    return v_out, t_spikes, v_thresh


def get_H_t(tau_j, alpha, tmin=0.001, tmax=0.1, ax=[]):
    dt = 0.001
    t_vect = np.linspace(tmin, tmax, int((tmax-tmin)/dt))
    h_t = np.zeros(t_vect.size)
    for i, t in enumerate(t_vect):
        h_t[i] = np.sum(np.array(alpha) * np.exp(-t / np.array(tau_j)))
    if not ax:
        f, ax = plt.subplots()
    ax.plot(1E3*t_vect, h_t)
    ax.set(xlabel='time (ms)', ylabel='H(t)')
    return ax


if __name__ == '__main__':
    sns.set_context('paper')
    fs, tmax = 10000, 0.3
    i_step = generate_signals.generate_step(fs, 0.05, tmax)
    i_random = np.cumsum(-0.5 + np.random.rand(int(tmax*fs)))
    i_random = (i_random-np.mean(i_random))/np.max(np.abs(i_random))
    i_random = i_random+0.5*max(i_random)
    i_sin = generate_signals.generate_sinus(fs, 200, t_max=tmax)
    # i_random = 1 + np.random.rand(int(tmax * fs))
    # LIF parameter
    refract_period, tau, v_thresh, v_spike, t_refract, v_reset = 0, 0.006, 0.02, 0.2, 0, 0

    i_syn = i_step / 10
    # i_syn = i_sin
    # i_syn = i_random
    # i_syn = 0.1 + i_syn / 5
    # i_syn = np.abs(i_syn)


    v_out, t_spikes = lif_standard(fs, refract_period, i_syn, tau, v_thresh, v_spike, t_refract, v_reset)
    refract_period_mat, t_refract_mat, tau_m, tau_j = 0, 0.002, 0.006, [0.010, 0.200]
    alpha_rs, alpha_ib, alpha_fs = np.array([0.037, 0.002]), np.array([0.0017, 0.002]), np.array([0.010, 0.000002])
    # t_refract, tau_m, tau_j = 0.001, 0.030, 0.005
    # alpha_rs, alpha_ib, alpha_fs = 0.1, 0.01, 0.001
    # w_rs, w_ib, w_fs = 0.019, 0.026, 0.011
    w_rs, w_ib, w_fs = 0.02, 0.02, 0.02
    v_out_mat_rs, t_spikes_mat_rs, v_thresh_mat_rs = lif_mat(fs, i_syn, refract_period_mat, t_refract_mat, tau_m, tau_j, alpha_rs, w_rs)
    v_out_mat_ib, t_spikes_mat_ib, v_thresh_mat_ib = lif_mat(fs, i_syn, refract_period_mat, t_refract_mat, tau_m, tau_j, alpha_ib, w_ib)
    v_out_mat_fs, t_spikes_mat_fs, v_thresh_mat_fs = lif_mat(fs, i_syn, refract_period_mat, t_refract_mat, tau_m, tau_j, alpha_fs, w_fs)

    t_vect = np.linspace(0, tmax, int(tmax*fs))
    # f = plt.figure()
    # ax = f.add_subplot(211)
    # ax.plot(t_vect, i_syn)
    # ax2 = f.add_subplot(212, sharex=ax)
    # ax2.plot(t_vect, v_out)
    # plt.autoscale(axis='x', tight=True)
    # ax2.hlines(v_thresh, 0, tmax)
    # LIF MAT model
    f = plt.figure()
    ax = f.add_subplot(411)
    ax.plot(t_vect, v_out)
    ax.plot([t_vect[0], t_vect[-1]], [v_thresh, v_thresh])
    ax.legend(['V(t)', 'Threshold'])
    ax.plot(t_vect, i_syn, c='k')
    ax.set(title='Input signal / Standard LIF')
    ax2 = f.add_subplot(412, sharex=ax)
    ax2.plot(t_vect, v_out_mat_rs)
    ax2.plot(t_vect, v_thresh_mat_rs)
    ax2.hlines(w_rs, xmin=t_vect[0], xmax=t_vect[-1], lw=1, zorder=0)
    ax2.legend(['V(t)', 'Threshold'])
    ax2.set(title='LIF Mat - RS type')
    ax3 = f.add_subplot(413, sharex=ax, sharey=ax2)
    ax3.plot(t_vect, v_out_mat_ib)
    ax3.plot(t_vect, v_thresh_mat_ib)
    ax3.legend(['V(t)', 'Threshold'])
    ax3.hlines(w_ib, xmin=t_vect[0], xmax=t_vect[-1], lw=1, zorder=0)
    ax3.set(title='LIF Mat - IB type')
    ax4 = f.add_subplot(414, sharex=ax, sharey=ax2)
    ax4.plot(t_vect, v_out_mat_fs)
    ax4.plot(t_vect, v_thresh_mat_fs)
    ax4.hlines(w_fs, xmin=t_vect[0], xmax=t_vect[-1], lw=1, zorder=0)
    ax4.legend(['V(t)', 'Threshold'])
    ax4.set(title='LIF Mat - FS type')
    plt.autoscale(axis='x', tight=True)

    # tmin = 0
    # ax = get_H_t(tau_j, 1E3*alpha_rs, tmin=tmin)
    # ax = get_H_t(tau_j, 1E3*alpha_ib, tmin=tmin, ax=ax)
    # ax = get_H_t(tau_j, 1E3*alpha_fs, tmin=tmin, ax=ax)
    # ax.legend(['RS', 'IB', 'FS'])
