import numpy as np
import peakutils
from scipy.signal import welch, periodogram
import matplotlib.pyplot as plt
import seaborn as sns
from librosa import feature


def get_spectral_features(x, fs, fmin=[], fmax=[], nfft=2048, do_plot=False, logscale=1):
    if fmin and fmax:
        spect_centroid = np.mean(feature.spectral_centroid(x, fs, n_fft=nfft, freq=np.linspace(fmin, fmax, 1 + int(nfft/2))))
        spect_rolloff = np.mean(feature.spectral_rolloff(x, fs, n_fft=nfft, freq=np.linspace(fmin, fmax, 1 + int(nfft/2))))
    else:
        spect_centroid = np.mean(feature.spectral_centroid(x, fs, n_fft=nfft))
        spect_rolloff = np.mean(feature.spectral_rolloff(x, fs, n_fft=nfft))
    peaks_freq, peak_amps, pxx_db, freqs = find_spectrum_peaks(x, fs, fmin, fmax, nfft)
    n_peaks = peaks_freq.size
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
    if not fmin:
        fmin = 0
    if not fmax:
        fmax = fs/2
    freqs, pxx = welch(x, fs, nfft=nfft)
    # freqs, pxx = periodogram(x, fs, nfft=nfft, window='hamming')
    fsel_ind = (freqs >= fmin) & (freqs <= fmax)
    freqs_sel, pxx_sel = freqs[fsel_ind], pxx[fsel_ind]
    pxx_sel_db = 10*np.log10(pxx_sel)
    peak_ind, peak_amps = find_peaks(pxx_sel_db, thresh_from_baseline=thresh_db_from_baseline)
    peak_freqs = freqs_sel[peak_ind]
    if do_plot:
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(freqs_sel, pxx_sel_db)
        ax.scatter(peak_freqs, peak_amps)
    return peak_freqs, peak_amps, pxx_sel_db, freqs_sel


def find_peaks(x, thresh_from_baseline, min_dist=1):
    x_scaled, old_range = peakutils.prepare.scale(x, (0, 1))
    x_baseline = peakutils.baseline(x_scaled)
    thresh_norm = thresh_from_baseline / np.diff(old_range)
    x_corrected = (x_scaled - x_baseline)
    # thresh_norm_scaled = thresh_norm * (x_corrected.max() - x_corrected.min())
    peak_indexes = peakutils.indexes(x_corrected, min_dist=min_dist)
    peak_indexes_sel = peak_indexes[x_corrected[peak_indexes] > thresh_norm]
    peak_amp = x[peak_indexes_sel]
    return peak_indexes_sel, peak_amp


