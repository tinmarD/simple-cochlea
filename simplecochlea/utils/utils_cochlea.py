import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import gaussian


def linearscale(fs, fmin_hz, fmax_hz, n_filters):
    """ Design n_filters band-pass filters ranging from fmin_hz to fmax_hz with a central frequencies increases
     linearly and with a fixed bandwidth defined such that the overlap of the frequency responses of adjacent filters
     is equal to half the bandwidth.

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz)
    fmin_hz : float
        Minimum frequency  (Hz)
    fmax_hz : float
        Maximum frequency (Hz)
    n_filters : int
        Number of filters in the range [fmin_hz, fmax_hz]

    Returns
    -------
    wn : array [n_filters*2]
        Normalized cutoff frequencies for all filters to be used for filter design (half-cycle / sample)
    cf : array
        Center frequency (Hz)
    """
    filter_bw = 2 * (fmax_hz - fmin_hz) / n_filters
    f_start = np.linspace(fmin_hz, fmax_hz - filter_bw, n_filters)
    f_end = np.linspace(fmin_hz + filter_bw, fmax_hz, n_filters)
    # cf = np.linspace(fmin_hz + filter_bw / 2, fmax_hz - filter_bw / 2, n_filters)
    f_start[f_start < 0] = 0.01
    f_end[f_end > fs / 2] = fs / 2.01
    wn = np.vstack([f_start, f_end]).T * 2 / fs
    cf = (f_start + f_end) / 2
    return wn, cf


def erbspace(fmin_hz, fmax_hz, N, q_ear=9.26449, bw_min=24.7):
    f_vect = np.arange(N-1, -1, -1)
    cf = -(q_ear*bw_min) + np.exp(f_vect * (-np.log(fmax_hz + q_ear * bw_min) +
                                            np.log(fmin_hz + q_ear * bw_min))/(N-1)) * (fmax_hz + q_ear * bw_min)
    return cf


def erbscale(fs, fmin_hz, fmax_hz, n_filters, q_ear=9.26449, bw_min=24.7, bw_mult=1):
    # Calcul of the central frequencies of band-pass filters
    cf = erbspace(fmin_hz, fmax_hz, n_filters, q_ear, bw_min)
    # ERB bandwith
    erb_bw = q_ear + cf / bw_min
    # Butterworth bandpass filters with pass-band centered on filterFcHz
    f_start = cf - bw_mult * erb_bw / 2
    f_end = cf + bw_mult * erb_bw / 2
    f_start[f_start < 0] = 0.01
    f_end[f_end > fs / 2] = fs / 2.01
    wn = np.vstack([f_start, f_end]).T * 2 / fs
    cf = (f_start + f_end) / 2
    return wn, cf


def musicscale(fs, fmin_hz, fmax_hz, n_filters, poly_coeff=[], bw_mult=10):
    """ Experimental - Try to compute equal-power band in a mean spectrum of music segments
        Do not use
    """
    poly_coeff = np.array(poly_coeff)
    if poly_coeff.size == 0:
        poly_coeff = np.array([7.84391019e-31, -1.01365138e-25, 3.78476589e-21,
                              -6.67590704e-17, 6.44986899e-13, -3.59218049e-09,
                              1.16813176e-05, -2.37661853e-02, 4.40987147e+01])
    cf, f_start, f_end = find_equal_areas_limits(poly_coeff, n_filters, fmin_hz, fmax_hz)
    # music bandwith
    music_bw = bw_mult * (f_end - f_start)
    f_start = cf - music_bw / 2
    f_end = cf + music_bw / 2
    f_start[f_start < 0] = 0.01
    f_end[f_end > fs / 2] = fs / 2.01
    wn = np.vstack([f_start, f_end]).T * 2 / fs
    cf = (f_start + f_end) / 2
    return wn, cf


def find_equal_areas_limits(poly_coeff, n_areas, fmin, fmax):
    poly_int_coeff = np.hstack([poly_coeff / np.arange(len(poly_coeff), 0, -1), 0])
    area_total = np.polyval(poly_int_coeff, fmax) - np.polyval(poly_int_coeff, fmin)
    f_limits = np.zeros(n_areas + 1)
    f_limits[0], f_limits[-1] = fmin, fmax
    gint_fmin = np.polyval(poly_int_coeff, fmin)
    for i in range(1, n_areas):
        poly_int_coeff_i = poly_int_coeff
        poly_int_coeff_i[-1] = -gint_fmin - i * area_total / n_areas
        roots = np.roots(poly_int_coeff_i)
        roots_real = roots[np.isreal(roots)]
        if roots_real.size == 0:
            raise ValueError('No real roots')
        elif roots_real.size > 1:
            roots_real = roots_real[(roots_real > fmin) & (roots_real < fmax)]
            if roots_real.size > 1:
                raise ValueError('Multiple real roots')
        f_limits[i] = np.real(roots_real)
    f_start, f_end = f_limits[:-1], f_limits[1:]
    cf = (f_start + f_end) / 2
    return cf, f_start, f_end


def plot_input_output(input_sig, output_sig, fs, input_sig_label, output_sig_label, same_colobar=0):
    f = plt.figure()
    n_pnts = input_sig.shape[1]
    n_channels = input_sig.shape[0]
    t_vect = np.linspace(0, n_pnts / fs, n_pnts)
    min_val = np.min([np.min(input_sig), np.min(output_sig)])
    max_val = np.max([np.max(input_sig), np.max(output_sig)])
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=1)
    if input_sig.shape[0] == 1 or len(input_sig.shape) == 1:
        ax1.plot(t_vect, input_sig)
        ax1.set(ylabel='Amplitude', title=input_sig_label)
    else:
        if same_colobar:
            im_in = plt.imshow(input_sig, aspect='auto', origin='lower', extent=(0, n_pnts / fs, 0, n_channels),
                               vmin=min_val, vmax=max_val)
        else:
            im_in = plt.imshow(input_sig, aspect='auto', origin='lower', extent=(0, n_pnts / fs, 0, n_channels))
        ax1.set(ylabel='Channel', title=input_sig_label)
    ax1.autoscale(axis='x', tight=True)
    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2, sharex=ax1)
    if same_colobar:
        im_out = plt.imshow(output_sig, aspect='auto', origin='lower', extent=(0, n_pnts / fs, 0, n_channels),
                            vmin=min_val, vmax=max_val)
    else:
        im_out = plt.imshow(output_sig, aspect='auto', origin='lower', extent=(0, n_pnts / fs, 0, n_channels))
    ax2.set(xlabel='time (s)', ylabel='Channel', title=output_sig_label)
    f.subplots_adjust(right=0.85)
    if same_colobar:
        cbar_ax = f.add_axes([0.90, 0.2, 0.025, 0.6])
        plt.colorbar(cax=cbar_ax)
    else:
        cbar_ax_in = f.add_axes([0.90, 0.68, 0.025, 0.18])
        cbar_ax_out = f.add_axes([0.90, 0.15, 0.025, 0.4])
        plt.colorbar(im_in, cax=cbar_ax_in)
        plt.colorbar(im_out, cax=cbar_ax_out)


def normalize_vector(x):
    x = np.array(x).squeeze()
    if len(x.shape) > 1:
        raise ValueError('Input must be a vector')
    a = (1 / (x.max() - 0))
    x_norm = a * x + 1 - a * x.max()
    return x_norm


def t_spikes_to_spikerate(t_spikes, fs, n_pnts, kernel_duration=0.015):
    t_vect_spk = np.zeros(n_pnts, dtype=int)
    t_vect_spk[np.round(t_spikes * fs).astype(int)] = 1
    kernel = gaussian(int(fs * kernel_duration), std=0.2 * int(fs * kernel_duration))
    kernel = kernel / np.sum(kernel_duration)
    t_vect_spk_smooth = np.convolve(t_vect_spk, kernel, 'same')
    return t_vect_spk_smooth

