import numpy as np
from scipy.signal import welch, periodogram
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from librosa import feature
    HAS_LIBROSA = True
except:
    HAS_LIBROSA = False
try:
    import peakutils
    HAS_PEAKUTILS = True
except:
    HAS_PEAKUTILS = False


def get_spectral_features(x, fs, fmin=[], fmax=[], nfft=2048, do_plot=False, logscale=True):
    """ Compute some spectral features using the `librosa <https://librosa.github.io/librosa/index.html>`_ library :
     * Spectrum centroid
     * Spectrum rolloff
     * Peaks in the power spectral density

    Parameters
    ----------
    x : array
        Input array. Must be 1D.
    fs : float
        Sampling frequency (Hz)
    fmin : float
        Minimum frequency (Hz)
    fmax : float
        Maximum frequency (Hz)
    nfft : int
        Number of points for the FFT - Default: 2048
    do_plot : bool
        If true, plot the spectral features - Default: False
    logscale : bool
        If True, use a log-scale for the x-axis - Default : True

    Returns
    -------
    spect_centroid : float
        Spetrum centroid. See :func:`librosa.feature.spectral_centroid`
    spect_rolloff : float
        Spectrum rolloff. See :func:`librosa.feature.spectral_centroid`
    peaks_freq : array
        Peak in the spectrum
    pxx_db : array
        Power Spectral Density (PSD), in dB
    freqs : array
        Frequency associated with the PSD

    """
    x = np.array(x)
    if x.ndim > 1:
        raise ValueError('Input x must be 1D')
    if not HAS_LIBROSA:
        raise ImportError('Librosa is not installed/available')
    if fmin and fmax:
        spect_centroid = np.mean(feature.spectral_centroid(x, fs, n_fft=nfft, freq=np.linspace(fmin, fmax, 1 + int(nfft/2))))
        spect_rolloff = np.mean(feature.spectral_rolloff(x, fs, n_fft=nfft, freq=np.linspace(fmin, fmax, 1 + int(nfft/2))))
    else:
        spect_centroid = np.mean(feature.spectral_centroid(x, fs, n_fft=nfft))
        spect_rolloff = np.mean(feature.spectral_rolloff(x, fs, n_fft=nfft))
    peaks_freq, peak_amps, pxx_db, freqs = find_spectrum_peaks(x, fs, fmin, fmax, nfft)
    # n_peaks = peaks_freq.size
    if do_plot:
        colors = sns.color_palette(n_colors=3)
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(freqs, pxx_db, color=colors[0])
        ax.axvline(spect_centroid, color=colors[2])
        ax.scatter(peaks_freq, peak_amps, color=colors[1])
        # ax.axvline(spect_rolloff)
        ax.autoscale(axis="x", tight=True)
        ax.set(xlabel='Frequency (Hz)', ylabel='Gain (dB)', title='Spectral Features')
        if logscale:
            ax.set_xscale('log')
            ax.grid(True, which="both", ls="-")
        plt.legend(['Pxx (dB)', 'Spectral Centroid', 'Spectral Peaks'])
    return spect_centroid, spect_rolloff, peaks_freq, pxx_db, freqs


def find_spectrum_peaks(x, fs, fmin=[], fmax=[], nfft=4092, thresh_db_from_baseline=6, do_plot=False):
    """ Find the peaks in the Power Spectral Density of signal `x` between `fmin` and `fmax`.
    A peak is detected if its amplitude is over the threshold defined by `thresh_db_from_baseline`.

    Parameters
    ----------
    x : array
        Input signal
    fs : float
        Sampling frequency (Hz)
    fmin : float
        Lower range frequency (Hz)
    fmax : float
        Upper range frequency (Hz)
    nfft : int
        Number of points for the FFT - Default : 4092
    thresh_db_from_baseline : float
        Threshold for detecting peaks from the baseline, in dB - Default: 6
    do_plot : bool
        If True, plot the PSD and the peaks - Default : False

    Returns
    -------
    peak_freqs : array
        Peaks frequency (Hz)
    peak_amps_db : array
        Peaks amplitude (dB)
    pxx_sel_db : array
        Power Spectral Density (dB)
    freqs_sel : array
        frequency associated with the PSD
    """
    if not fmin:
        fmin = 0
    if not fmax:
        fmax = fs/2
    freqs, pxx = welch(x, fs, nfft=nfft)
    # freqs, pxx = periodogram(x, fs, nfft=nfft, window='hamming')
    fsel_ind = (freqs >= fmin) & (freqs <= fmax)
    freqs_sel, pxx_sel = freqs[fsel_ind], pxx[fsel_ind]
    pxx_sel_db = 10*np.log10(pxx_sel)
    peak_ind, peak_amps_db = find_peaks(pxx_sel_db, thresh_from_baseline=thresh_db_from_baseline)
    peak_freqs = freqs_sel[peak_ind]
    if do_plot:
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(freqs_sel, pxx_sel_db)
        ax.scatter(peak_freqs, peak_amps_db)
    return peak_freqs, peak_amps_db, pxx_sel_db, freqs_sel


def find_peaks(x, thresh_from_baseline, min_dist=1):
    """ Algorithm for detecting peaks above the baseline.
    A peak should be `thresh_from_baseline` above the baseline to be detected.

    Parameters
    ----------
    x : array
        Input array
    thresh_from_baseline : float
        Threshold for detecting peaks from the baseline, in dB
    min_dist : int
        Minimum distance between peak indices - Default : 1

    Returns
    -------
    peak_indexes_sel : array
        Peak indices
    peak_amp : array
        Peak amplitudes

    """
    if not HAS_PEAKUTILS:
        raise ImportError('peakutils is not installed/available')
    x_scaled, old_range = peakutils.prepare.scale(x, (0, 1))
    x_baseline = peakutils.baseline(x_scaled)
    thresh_norm = thresh_from_baseline / np.diff(old_range)
    x_corrected = (x_scaled - x_baseline)
    # thresh_norm_scaled = thresh_norm * (x_corrected.max() - x_corrected.min())
    peak_indexes = peakutils.indexes(x_corrected, min_dist=min_dist)
    peak_indexes_sel = peak_indexes[x_corrected[peak_indexes] > thresh_norm]
    peak_amp = x[peak_indexes_sel]
    return peak_indexes_sel, peak_amp

