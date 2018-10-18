import numpy as np
from os import path, mkdir
import pandas as pd
from scipy import signal
import _pickle
from datetime import datetime
import tqdm
import time
import multiprocessing as mp
from sklearn.model_selection import ParameterGrid
from functools import wraps
import matplotlib
import seaborn as sns

from .generate_signals import *
from .spikes.spikelist import *
from .utils.utils_cochlea import *
from .cython.cochlea_fun_cy import *
from .utils.utils_freqanalysis import *

matplotlib.use('agg')
sns.set()
sns.set_context('paper')


def timethis(func):
    @wraps(func)
    def add_timer(*args, **kwargs):
        t_start = time.time()
        res = func(*args, **kwargs)
        print('Function : {} - Time elapsed : {}'.format(func.__name__, time.time() - t_start))
        return res
    return add_timer


class BandPassFilterbank:
    """ Band-pass Filterbank Class
    Defines a bank of band-pass filters. The Cochlea class uses it to model the inner hair cells.

    Attributes
    ----------
    fs : float
        Sampling frequency (Hz). Each BandPassFilterbank instance must be applied to signals with the same sampling
        frequency.
    order : int
        Filter order
    ftype : str
        Filter type. Must be in :
            * 'butter' : Butterworth filters
            * 'bessel' : Bessel filters
            * 'fir' : FIR filters
    n_filters : int
        Number of band-pass filters
    a : array
        Filter coefficients - equal to 1 if FIR filters
    b : array
        Filter coefficients

    """
    def __init__(self, order, wn, fs, ftype='butter'):
        wn = np.array(wn)
        if wn.ndim != 2 and wn.shape[1] != 2:
            raise ValueError('wn argument must be a matrix of size [n_filters, 2]')
        self.n_filters = wn.shape[0]
        self.b = np.zeros([self.n_filters, 2 * order + 1])
        if ftype.lower() == 'fir':
            self.a = np.ones((self.n_filters, 1))
        else:
            self.a = np.zeros([self.n_filters, 2 * order + 1])
        self.fs = fs
        for i in range(0, self.n_filters):
            if ftype.lower() == 'butter':
                self.b[i, :], self.a[i, :] = signal.butter(order, wn[i, :], btype='bandpass')
            elif ftype.lower() == 'bessel':
                self.b[i, :], self.a[i, :] = signal.bessel(order, wn[i, :], btype='bandpass')
            elif ftype.lower() == 'fir':
                self.b[i, :] = signal.firwin(2*order+1, wn[i, :], pass_zero=False)
            else:
                raise ValueError('Wrong ftype argument {}'.format(ftype))
        sns.set()
        sns.set_context('paper')
        plt.rcParams['image.cmap'] = 'viridis'

    def plot_frequency_response(self, n_freqs=1024):
        """ Plot the frequency response of the differents band-pass filters

        Parameters
        ----------
        n_freqs : int
            Number of frequencies used to compute the frequency response - Default : 1024

        Returns
        -------
        ax : handle
            Pyplot axis handle

        """
        f = plt.figure()
        ax = f.add_subplot(111)
        for i in range(0, self.n_filters):
            w, h = signal.freqz(self.b[i, :], self.a[i, :], worN=n_freqs)
            plt.plot((self.fs * 0.5 / np.pi) * w, 20 * np.log10(np.absolute(h)+1e-8))
        ax.set(xlabel='f (Hz)', ylabel='H (dB)',
               title='Filters frequency responses - {} filters'.format(self.n_filters))
        ax.autoscale(axis='x', tight=True)
        f.show()
        return ax

    def filter_one_channel(self, input_sig, channel_pos, zi_in=[]):
        """ Filter input signal `input_sig` with channel specified by `channel_pos`

        Parameters
        ----------
        input_sig : array
            Input signal
        channel_pos : int
            Channel position
        zi_in : array
            Initial conditions for the filter delays - default : None

        Returns
        -------
        output : array (1D)
            Output filtered signal
        zi_out : array
            Final filter delay values

        """
        if len(np.array(input_sig).shape) > 1 and not np.min(np.array(input_sig).shape) == 1:
            raise ValueError('input_sig must be a 1D array')
        if not np.isscalar(channel_pos) or channel_pos < 0 or channel_pos > self.n_filters-1:
            raise ValueError('argument channel_pos must be a scalar ranging between 0 and the number of channels-1')
        if not np.array(zi_in).any():
            zi_in = np.zeros(self.b.shape[1] - 1)
        output, zi_out = signal.lfilter(self.b[channel_pos, :], self.a[channel_pos, :], input_sig, zi=zi_in)
        return output, zi_out

    def filter_signal(self, input_sig, do_plot=0, zi_in=[]):
        """ Filter input signal `input_sig`.
        Input signal is passed through all the band-pass filters.

        Parameters
        ----------
        input_sig : array
            Input signal
        do_plot : bool
            If True, plot the output filtered signals as an image.
        zi_in : array
            Initial conditions for the filter delays - default : None

        Returns
        -------
        output : array (2D)
            Output filtered signals [n_filters * n_pnts]
        zi_out : array
            Final filter delay values


        """
        if len(np.array(input_sig).shape) > 1 and not np.min(np.array(input_sig).shape) == 1:
            raise ValueError('input_sig must be a 1D array')
        n_pnts = len(input_sig)
        output = np.zeros([self.n_filters, n_pnts])
        if not np.array(zi_in).any():
            zi_in = np.zeros((self.n_filters, self.b.shape[1] - 1))
        zi_out = np.zeros((self.n_filters, self.b.shape[1] - 1))
        for i in range(0, self.n_filters):
            output[i, :], zi_out[i, :] = signal.lfilter(self.b[i, :], self.a[i, :], input_sig, zi=zi_in[i, :])
        if do_plot:
            f = plt.figure()
            t_vect = np.linspace(0, n_pnts/self.fs, n_pnts)
            ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=1, colspan=1)
            ax1.plot(t_vect, input_sig)
            ax1.set(ylabel='Amplitude', title='input_sig signal')
            ax1.autoscale(axis='x', tight=True)
            ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=3, colspan=1, sharex=ax1)
            plt.imshow(output, aspect='auto', origin='lower', extent=(0, n_pnts/self.fs, 0, self.n_filters))
            ax2.set(xlabel='time (s)', ylabel='channel', title='Filterbank output')
            f.subplots_adjust(right=0.85)
            f.add_axes([0.90, 0.15, 0.025, 0.5])
            f.show()
        return output, zi_out


class RectifierBank:
    """ RectifierBank Class
    Rectifier Bank, possesses multilples channels. Half and full rectification are possible :
        * half-rectification : :math:`f(x) = x if x > 0, 0 otherwise`
        * full-rectification : :math:`f(x) = abs(x)`
    Rectification can include a low-pass filtering using a Butterworth filters. Each channel can have its own filtering
    cutoff frequency.
    The cochlea uses a rectifier bank just after the band-pass filtering step.

    Attributes
    ----------
    fs : float
        Sampling frequency (Hz). Each BandPassFilterbank instance must be applied to signals with the same sampling
        frequency.
    n_channels : int
        Number of channels
    rtype : str
        Rectification type - to choose in : (Default : 'half')
            * 'half' : :math:`f(x) = x if x > 0, 0 otherwise`
            * 'full' : :math:`f(x) = abs(x)`
    lowpass_freq : flaot | array | None
        Low pass filter cutoff frequency. If scalar all channels are filtered in the same way. If it contains as much
        frequencies as there are channels, each channel is filtered with its own cutoff frequency. If None, no filtering.
        Default : None
    filtorder : int
        Order of the low-pass filter
    filttype : str
        Indicates the filtering scheme :
            * 'global' : all channels are filtered with the same filter
            * 'channel' : each channel has its own filtering parameters
            * 'none' : no low-pass filtering
    a : array
        Low-pass filter coefficients
    b : array
        Low-pass filter coefficients
    """
    def __init__(self, n_channels, fs, rtype='half', lowpass_freq=[], filtorder=1):
        if not rtype.lower() == 'half' and not rtype.lower() == 'full':
            raise ValueError('rtype argument can take the values : half or full')
        if lowpass_freq and np.isscalar(lowpass_freq):
            self.filttype = 'global'
            self.b, self.a = signal.butter(filtorder, lowpass_freq * 2 / fs, btype='lowpass')
        elif lowpass_freq and len(lowpass_freq) == self.n_channels:
            self.filttype = 'channel'
            self.b, self.a = np.zeros((self.n_channels, filtorder+1)), np.zeros((self.n_channels, filtorder+1))
            for i in range(0, self.n_channels):
                self.b[i, :], self.a[i, :] = signal.butter(filtorder, lowpass_freq[i] * 2 / fs, btype='lowpass')
        elif not lowpass_freq:
            self.filttype = 'none'
            self.b, self.a = [], []
        else:
            raise ValueError('Wrong argument lowpass_freq')
        self.fs = fs
        self.n_channels = n_channels
        self.rtype = rtype.lower()
        self.lowpass_freq = lowpass_freq
        self.filtorder = filtorder

    def __str__(self):
        if self.filttype == 'none':
            return 'Rectifier Bank - {} rectification - No low-pass filtering\n'.format(self.rtype)
        else:
            return 'Rectifier Bank - {} rectification - Low-pass frequency : {} Hz' \
                   ' ({} order Butterworth filter)\n'.format(self.rtype, self.lowpass_freq, self.filtorder)

    def rectify_one_channel(self, input_sig, channel_pos):
        """ Rectify input signal `input_sig` with channel specified by `channel_pos`

        Parameters
        ----------
        input_sig : array (1D)
            Input signal
        channel_pos : int
            Channel position

        Returns
        -------
        output_sig : array (1D)
            Output rectified signal

        """
        input_sig = np.array([input_sig]).squeeze()
        if len(np.array(input_sig).shape) > 1 and not np.min(np.array(input_sig).shape) == 1:
            raise ValueError('input_sig must be a 1D array')
        if not np.isscalar(channel_pos) or channel_pos < 0 or channel_pos > self.n_channels-1:
            raise ValueError('argument channel_pos must be a scalar ranging between 0 and the number of channels-1')
        output_sig = input_sig
        # Rectification
        if self.rtype == 'half':
            output_sig[output_sig < 0] = 0
        elif self.rtype == 'full':
            output_sig = np.abs(output_sig)
        else:
            raise ValueError('Wrong rtype argument : {}'.format(self.rtype))
        # Low pass Filtering
        if self.filttype == 'global':
            output_sig = signal.lfilter(self.b, self.a, output_sig)
        elif self.filttype == 'channel':
            for i in range(0, self.n_channels):
                output_sig = signal.lfilter(self.b[channel_pos, :], self.a[channel_pos, :], output_sig)
        return output_sig

    def rectify_signal(self, input_sig, do_plot=0):
        """ Rectify input signal `input_sig`.
        Input signal is passed through all the band-pass filters.

        Parameters
        ----------
        input_sig : array (1D)
            Input signal
        do_plot : bool
            If True, plot the output rectified signals as an image.

        Returns
        -------
        output_sig : array (2D)
            Output rectified signal [n_filters * n_pnts]

        """
        input_sig = np.array([input_sig]).squeeze()
        if len(input_sig.shape) == 1:
            input_sig = np.tile(input_sig, (self.n_channels, 1))
        if not input_sig.shape[0] == self.n_channels:
            raise ValueError('input_sig must be a [n_channels, n_pnts] matrice')
        output_sig = input_sig
        # Rectification
        if self.rtype == 'half':
            output_sig[output_sig < 0] = 0
        elif self.rtype == 'full':
            output_sig = np.abs(output_sig)
        else:
            raise ValueError('Wrong rtype argument : {}'.format(self.rtype))
        # Low pass Filtering
        if self.filttype == 'global':
            for i in range(0, self.n_channels):
                output_sig[i, :] = signal.lfilter(self.b, self.a, output_sig[i, :])
        elif self.filttype == 'channel':
            for i in range(0, self.n_channels):
                output_sig[i, :] = signal.lfilter(self.b[i, :], self.a[i, :], output_sig[i, :])
        # Plotting
        if do_plot:
            plot_input_output(input_sig, output_sig, self.fs, 'Input signal', 'Rectifier Output', 0)
        return output_sig


class CompressionBank:
    """ CompressionBank Class
    Apply an amplitude compression.

    """
    def __init__(self, n_channels, fs, compression_factor, compression_gain):
        self.n_channels = n_channels
        if not np.isscalar(compression_factor) and not len(compression_factor) == n_channels:
            raise ValueError('Argument compression_factor must be either a scalar or a vector of length n_channels')
        if not np.isscalar(compression_gain) and not len(compression_gain) == n_channels:
            raise ValueError('Argument compression_gain must be either a scalar or a vector of length n_channels')
        if np.isscalar(compression_factor):
            compression_factor = np.repeat(compression_factor, self.n_channels)
        if np.isscalar(compression_gain):
            compression_gain = np.repeat(compression_gain, self.n_channels)
        self.comp_factor = np.array(compression_factor)
        self.comp_gain = np.array(compression_gain)
        self.fs = fs

    def __str__(self):
        unique_comp_gain, unique_comp_factor = np.unique(self.comp_gain), np.unique(self.comp_factor)
        desc_str = r'Compression Bank : y = '
        if unique_comp_gain.size == 1:
            desc_str += '{} * x ^'.format(float(unique_comp_gain))
        else:
            desc_str += '[{}, {}] * x ^'.format(self.comp_gain[0], self.comp_gain[-1])
        if unique_comp_factor.size == 1:
            desc_str += ' {}\n'.format(float(unique_comp_factor))
        else:
            desc_str += ' [{}, {}]\n'.format(self.comp_factor[0], self.comp_factor-1)
        return desc_str

    def compress_one_channel(self, input_sig, channel_pos):
        input_sig = np.array(input_sig).squeeze()
        if len(np.array(input_sig).shape) > 1 and not np.min(np.array(input_sig).shape) == 1:
            raise ValueError('input_sig must be a 1D array')
        if not np.isscalar(channel_pos) or channel_pos < 0 or channel_pos > self.n_channels-1:
            raise ValueError('argument channel_pos must be a scalar ranging between 0 and the number of channels-1')
        if not np.isscalar(self.comp_factor):
            comp_factor = self.comp_factor[channel_pos]
        else:
            comp_factor = self.comp_factor
        if not np.isscalar(self.comp_gain):
            comp_gain = self.comp_gain[channel_pos]
        else:
            comp_gain = self.comp_gain
        output_sig = np.power(input_sig, comp_factor) * comp_gain
        return output_sig

    def compress_signal(self, input_sig, do_plot=0):
        input_sig = np.array(input_sig).squeeze()
        if len(input_sig.shape) == 1:
            input_sig = np.tile(input_sig, (self.n_channels, 1))
        if not input_sig.shape[0] == self.n_channels:
            raise ValueError('input_sig must be a [n_channels, n_pnts] matrice')
        if np.isscalar(self.comp_factor):
            output_sig = np.power(input_sig, self.comp_factor)
        elif len(self.comp_factor) == self.n_channels:
            output_sig = np.power(input_sig, np.array([self.comp_factor]).T)
        else:
            raise ValueError('Wrong compression factor parameter : {}'.format(self.comp_factor))
        if np.isscalar(self.comp_gain):
            output_sig = output_sig * self.comp_gain
        elif len(self.comp_gain) == self.n_channels:
            output_sig = output_sig * np.array([self.comp_gain]).T
        else:
            raise ValueError('Wrong compression gain parameter : {}'.format(self.comp_gain))
        # Plotting
        if do_plot:
            plot_input_output(input_sig, output_sig, self.fs, 'Input signal', 'Compression output', 0)
        return output_sig


class LIFBank:
    def __init__(self, n_channels, fs, tau, v_thresh=0.020, v_spike=0.035, t_refract=[], v_reset=[], inhib_type=[],
                 inhib_vect=[], tau_j=[], alpha_j=[], omega=[]):
        t_refract, v_reset, inhib_vect = np.array(t_refract), np.array(v_reset), np.array(inhib_vect)
        if t_refract.size == 0 and v_reset.size == 0:
            raise ValueError('Either t_refract or v_reset must be specified')
        if not np.isscalar(tau) and not len(tau) == n_channels:
            raise ValueError('Argument tau must be either a scalar or a vector of length n_channels')
        if not np.isscalar(v_thresh) and not len(v_thresh) == n_channels:
            raise ValueError('Argument v_thresh must be either a scalar or a vector of length n_channels')
        if not t_refract.size == 0 and not t_refract.size == 1 and not t_refract.size == n_channels:
            raise ValueError('Argument t_refract must be either a scalar or a vector of length n_channels')
        if v_reset.size > 0 and not v_reset.size == 1 and not v_reset.size == n_channels:
            raise ValueError('Argument v_reset must be either a scalar or a vector of length n_channels')
        self.n_channels = n_channels
        if np.isscalar(tau):
            tau = np.repeat(tau, self.n_channels)
        if np.isscalar(v_thresh):
            v_thresh = np.repeat(v_thresh, self.n_channels)
        if np.isscalar(v_spike):
            v_spike = np.repeat(v_spike, self.n_channels)
        if t_refract.size == 1:
            t_refract = np.repeat(t_refract, self.n_channels)
        if v_reset.size == 0:
            v_reset = np.zeros(self.n_channels)
        if v_reset.size == 1:
            v_reset = v_reset * np.ones(self.n_channels)
        if t_refract.size == 0:
            self.refract_period = False
            self.v_reset = v_reset
            self.t_refract = np.repeat(np.nan, self.n_channels)
        else:
            self.refract_period = True
            self.v_reset = v_reset
            self.t_refract = t_refract
        self.tau = tau
        self.v_thresh = v_thresh
        self.v_spike = v_spike
        self.fs = fs
        if inhib_type:
            if inhib_type not in ['sub_for', 'shunt_for', 'shunt_for_current', 'spike']:
                raise ValueError('Wrong inhib_type arguement : {}. Possible values are {}'
                                 .format(inhib_type, ['sub_for', 'shunt_for', 'shunt_for_current' 'spike']))
            if inhib_vect.size % 2 == 0:
                raise ValueError('Length of argument inhib_vect should be odd')
        self.inhib_type = inhib_type
        self.inhib_vect = inhib_vect if inhib_type else []
        self.tau_j, self.alpha_j, self.omega = np.array(tau_j), np.array(alpha_j), np.array(omega)
        if self.omega.size == 1:
            self.omega = np.repeat(self.omega, self.n_channels)
        if not self.tau_j.size == self.alpha_j.size:
            raise ValueError('Arguments tau_j and alpha_j must have the same size')

    def __str__(self):
        desc_str = 'LIF bank - '
        mat_model = hasattr(self, 'alpha_j') and self.alpha_j.size > 0
        unique_tau, unique_v_thresh = np.unique(self.tau), np.unique(self.v_thresh)
        unique_v_reset, unique_t_refract = np.unique(self.v_reset), np.unique(self.t_refract)
        if unique_tau.size == 1:
            desc_str += 'Tau = {} ms - '.format(float(1000*unique_tau))
        else:
            desc_str += 'Tau = [{:.2f}, {:.2f}] ms - '.format(1000*self.tau[0], 1000*self.tau[-1])
        if not mat_model:
            if unique_v_thresh.size == 1:
                desc_str += 'V_thresh = {} - '.format(float(unique_v_thresh))
            else:
                desc_str += 'V_thresh = [{}, {}] - '.format(self.v_thresh[0], self.v_thresh[-1])
            if unique_v_reset.size == 1:
                desc_str += 'V_reset = {}\n'.format(float(unique_v_reset))
            else:
                desc_str += 'V_reset = [{}, {}] - '.format(self.v_reset[0], self.v_reset[-1])
        else:
            unique_omega = np.unique(self.omega)
            desc_str += '\nAdaptive Threshold model - tau_j = {}, alpha_j = {}'.format(self.tau_j, self.alpha_j)
            if unique_omega.size == 1:
                desc_str += ', omega = {:.2f}\n'.format(unique_omega[0])
            else:
                desc_str += ', omega = [{:.2f}, {:.2f}]\n'.format(unique_omega[0], unique_omega[-1])
        if unique_t_refract.size == 1:
            desc_str += 'Refractory period : {}s\n'.format(float(unique_t_refract))
        else:
            desc_str += 'Refractory period : [{}, {}s]\n'.format(self.t_refract[0], self.t_refract[-1])
        if hasattr(self, 'inhib_type') and self.inhib_type:
            desc_str += 'Inhibition mode : {}. Length of inhibition vector : {}'.format(self.inhib_type,
                                                                                        self.inhib_vect.size)
        else:
            desc_str += 'No inhibition'
        return desc_str

    def filter_signal_with_inhib_cy(self, input_sig, do_plot=0, v_init=[], t_start=0, t_last_spike=[], mp_ver=0):
        input_sig = np.array(input_sig).squeeze()
        if len(input_sig.shape) == 1:
            input_sig = np.tile(input_sig, (self.n_channels, 1))
        if not input_sig.shape[0] == self.n_channels:
            raise ValueError('input_sig must be a [n_channels, n_pnts] matrice')
        if not np.array(v_init).any():
            v_init = np.zeros(self.n_channels)
        elif not np.array(v_init).shape[0] == self.n_channels:
            raise ValueError('v_init must be a [n_channels, 1] array')
        if not np.array(t_last_spike).any():
            t_last_spike = -2 * self.t_refract * np.ones(self.n_channels)
        elif not t_last_spike.shape[0] == self.n_channels:
            raise ValueError('t_last_spike must be a [n_channels, 1] array')
        if self.inhib_type == 'sub_for':
            output_sig, t_spikes, chan_spikes = lif_filter_inhib_subfor_cy\
                (int(self.fs), np.float64(input_sig), int(self.refract_period), np.float64(self.t_refract), np.float64(self.tau),
                 np.float64(self.v_thresh), np.float64(self.v_spike), np.float64(self.v_reset), np.float64(v_init),
                 np.float64(self.inhib_vect), t_start=np.float64(t_start), t_last_spike_p=np.float64(t_last_spike))
        elif self.inhib_type == 'shunt_for':
            output_sig, t_spikes, chan_spikes = lif_filter_inhib_shuntfor_cy\
                (int(self.fs), np.float64(input_sig), int(self.refract_period), np.float64(self.t_refract), np.float64(self.tau),
                 np.float64(self.v_thresh), np.float64(self.v_spike), np.float64(self.v_reset), np.float64(v_init),
                 np.float64(self.inhib_vect), t_start=np.float64(t_start), t_last_spike_p=np.float64(t_last_spike))
        elif self.inhib_type == 'shunt_for_current':
            if mp_ver:
                output_sig, t_spikes, chan_spikes = lif_filter_inhib_shuntfor_current_mpver_cy\
                    (int(self.fs), np.float64(input_sig), int(self.refract_period), np.float64(self.t_refract), np.float64(self.tau),
                     np.float64(self.v_thresh), np.float64(self.v_spike), np.float64(self.v_reset), np.float64(v_init),
                     np.float64(self.inhib_vect), t_start=np.float64(t_start), t_last_spike_p=np.float64(t_last_spike))
            else:
                output_sig, t_spikes, chan_spikes = lif_filter_inhib_shuntfor_current_cy\
                    (int(self.fs), np.float64(input_sig), int(self.refract_period), np.float64(self.t_refract), np.float64(self.tau),
                     np.float64(self.v_thresh), np.float64(self.v_spike), np.float64(self.v_reset), np.float64(v_init),
                     np.float64(self.inhib_vect), t_start=np.float64(t_start), t_last_spike_p=np.float64(t_last_spike))
        elif self.inhib_type == 'spike':
            output_sig, t_spikes, chan_spikes = lif_filter_spike_inhib_cy\
                (int(self.fs), np.float64(input_sig), int(self.refract_period), np.float64(self.t_refract), np.float64(self.tau),
                 np.float64(self.v_thresh), np.float64(self.v_spike), np.float64(self.v_reset), np.float64(v_init),
                 np.float64(self.inhib_vect), t_start=np.float64(t_start), t_last_spike_p=np.float64(t_last_spike))
        else:
            raise ValueError('Wrong argument inhib_type : {}'.format(inhib_type))
        spikes = np.vstack([t_spikes, chan_spikes]).T
        return output_sig, spikes, []

    def filter_signal_adaptive_threshold(self, input_sig, v_init=[], t_start=0, t_last_spike=[], samplever=1,
                                         inhib_shuntfor=0):
        input_sig = np.array(input_sig).squeeze()
        if len(input_sig.shape) == 1:
            input_sig = np.tile(input_sig, (self.n_channels, 1))
        if not input_sig.shape[0] == self.n_channels:
            raise ValueError('input_sig must be a [n_channels, n_pnts] matrice')
        if not np.array(v_init).any():
            v_init = np.zeros(self.n_channels)
        elif not np.array(v_init).shape[0] == self.n_channels:
            raise ValueError('v_init must be a [n_channels, 1] array')
        if not np.array(t_last_spike).any():
            t_last_spike = -2 * self.t_refract * np.ones(self.n_channels)
        elif not t_last_spike.shape[0] == self.n_channels:
            raise ValueError('t_last_spike must be a [n_channels, 1] array')

        if inhib_shuntfor:
            output_sig, t_spikes, chan_spikes, threshold = lif_filter_adaptive_threshold_samplever_inhib_shuntfor_current_cy \
                (int(self.fs), np.float64(input_sig), int(self.refract_period), np.float64(self.t_refract),
                 np.float64(self.tau), np.float64(self.v_spike), np.float64(self.v_reset), np.float64(v_init),
                 np.float64(self.tau_j), np.float64(self.alpha_j), np.float64(self.omega), np.float64(self.inhib_vect),
                 t_start=np.float64(t_start), t_last_spike_p=np.float64(t_last_spike))

        elif samplever:
            output_sig, t_spikes, chan_spikes, threshold = lif_filter_adaptive_threshold_samplever_cy \
                (int(self.fs), np.float64(input_sig), int(self.refract_period), np.float64(self.t_refract),
                 np.float64(self.tau), np.float64(self.v_spike), np.float64(self.v_reset), np.float64(v_init),
                 np.float64(self.tau_j), np.float64(self.alpha_j), np.float64(self.omega),
                 t_start=np.float64(t_start), t_last_spike_p=np.float64(t_last_spike))
        else:
            output_sig, t_spikes, chan_spikes, threshold = lif_filter_adaptive_threshold_cy \
                (int(self.fs), np.float64(input_sig), int(self.refract_period), np.float64(self.t_refract),
                 np.float64(self.tau), np.float64(self.v_spike), np.float64(self.v_reset), np.float64(v_init),
                 np.float64(self.tau_j), np.float64(self.alpha_j), np.float64(self.omega),
                 t_start=np.float64(t_start), t_last_spike_p=np.float64(t_last_spike))
        spikes = np.vstack([t_spikes, chan_spikes]).T
        return output_sig, spikes, threshold

    def filter_signal(self, input_sig, do_plot=0, v_init=[], t_start=0, t_last_spike=[], n_processes=[]):
        input_sig = np.array(input_sig).squeeze()
        if len(input_sig.shape) == 1:
            input_sig = np.tile(input_sig, (self.n_channels, 1))
        if not input_sig.shape[0] == self.n_channels:
            raise ValueError('input_sig must be a [n_channels, n_pnts] matrice')
        if not np.array(v_init).any():
            v_init = np.zeros(self.n_channels)
        elif not np.array(v_init).shape[0] == self.n_channels:
            raise ValueError('v_init must be a [n_channels, 1] array')
        if not np.array(t_last_spike).any():
            t_last_spike = -2 * self.t_refract * np.ones(self.n_channels)
        elif not t_last_spike.shape[0] == self.n_channels:
            raise ValueError('t_last_spike must be a [n_channels, 1] array')
        n_pnts = input_sig.shape[1]
        output_sig = np.zeros((self.n_channels, n_pnts))
        if not n_processes:
            t_last_spike_out = np.zeros(self.n_channels)
            spikes = []
            for i in range(0, self.n_channels):
                output_sig[i, :], t_spikes_i = lif_filter_1d_signal_cy(self.fs, input_sig[i, :], self.refract_period, self.t_refract[i],
                                                                       self.tau[i], self.v_thresh[i], self.v_spike[i],
                                                                       self.v_reset[i], v_init[i], t_start=np.float64(t_start),
                                                                       t_last_spike_p=np.float64(t_last_spike[i]))
                spikes_i = np.vstack([t_spikes_i, i*np.ones(len(t_spikes_i))]).T
                t_last_spike_out[i] = t_spikes_i[-1] if spikes_i.any() else -2*self.t_refract[i]
                spikes.append(spikes_i)
            spikes = np.vstack(np.array(spikes))
        else:
            pool = mp.Pool(processes=n_processes)
            async_out = [pool.apply_async(lif_filter_1d_signal_cy,
                                          args=(self.fs, input_sig[i, :], self.refract_period, self.t_refract[i],
                                                self.tau[i], self.v_thresh[i], self.v_spike[i], self.v_reset[i], v_init[i],
                                                np.float64(t_start), np.float64(t_last_spike[i])))
                         for i in range(0, self.n_channels)]
            lif_output = [p.get() for p in async_out]
            output_sig = [lif_output[i][0] for i in range(self.n_channels)]
            t_spikes = [lif_output[i][1] for i in range(self.n_channels)]
            spikes = np.vstack([np.array([t_spikes[i], i*np.ones(t_spikes[i].size, dtype=int)]).T for i in range(self.n_channels)])
            t_last_spike_out = np.array([t_spikes[i][-1] if t_spikes[i].any() else -2 * self.t_refract[i]] for i in range(self.n_channels))

        if do_plot:
            self.plot_spikes(spikes[:, 0], spikes[:, 1], tmin=0, tmax=n_pnts/self.fs, potential=output_sig)
        return output_sig, spikes, t_last_spike_out

    def filter_one_channel(self, input_sig, i, v_init=[], t_start=0, t_last_spike=[]):
        if len(np.array(input_sig).shape) > 1 and not np.min(np.array(input_sig).shape) == 1:
            raise ValueError('input_sig must be a 1D array')
        if not np.isscalar(i) or i < 0 or i > self.n_channels-1:
            raise ValueError('argument channel_pos must be a scalar ranging between 0 and the number of channels-1')
        if not np.array(v_init).any():
            v_init = np.zeros(self.n_channels)
        elif not np.array(v_init).shape[0] == self.n_channels:
            raise ValueError('v_init must be a [n_channels, 1] array')
        if not np.array(t_last_spike).any():
            t_last_spike = -2 * self.t_refract * np.ones(self.n_channels)
        elif not t_last_spike.shape[0] == self.n_channels:
            raise ValueError('t_last_spike must be a [n_channels, 1] array')
        v_out, t_spikes = lif_filter_1d_signal_cy(self.fs, input_sig, self.refract_period, self.t_refract[i],
                                                  self.tau[i], self.v_thresh[i], self.v_spike[i], self.v_reset[i],
                                                  v_init[i], t_start=np.float64(t_start),
                                                  t_last_spike_p=np.float64(t_last_spike[i]))
        return v_out, t_spikes

    def filter_one_channel_adaptive_threshold(self, input_sig, i, v_init=[], t_start=0, t_last_spike=[]):
        if len(np.array(input_sig).shape) > 1 and not np.min(np.array(input_sig).shape) == 1:
            raise ValueError('input_sig must be a 1D array')
        if not np.isscalar(i) or i < 0 or i > self.n_channels-1:
            raise ValueError('argument channel_pos must be a scalar ranging between 0 and the number of channels-1')
        if not np.array(v_init).any():
            v_init = np.zeros(self.n_channels)
        elif not np.array(v_init).shape[0] == self.n_channels:
            raise ValueError('v_init must be a [n_channels, 1] array')
        if not np.array(t_last_spike).any():
            t_last_spike = -2 * self.t_refract * np.ones(self.n_channels)
        elif not t_last_spike.shape[0] == self.n_channels:
            raise ValueError('t_last_spike must be a [n_channels, 1] array')
        v_out, t_spikes, threshold = lif_filter_one_channel_adaptive_threshold_cy(
            self.fs, input_sig, self.refract_period, self.t_refract[i], self.tau[i], self.v_spike[i], self.v_reset[i],
            v_init[i], self.tau_j, self.alpha_j, self.omega[i],  t_start=np.float64(t_start),
            t_last_spike_p=np.float64(t_last_spike[i]))

        return v_out, t_spikes, threshold

    # def filter_one_channel_nocheck(self, input_sig, i, v_init=0, t_start=0, t_last_spike=[]):
    #     if not t_last_spike:
    #         t_last_spike = -2 * self.t_refract[i]
    #     _, t_spikes = lif_filter_1d_signal_cy(self.fs, input_sig, self.refract_period, self.t_refract[i],
    #                                               self.tau[i], self.v_thresh[i], self.v_spike[i], self.v_reset[i],
    #                                               v_init, t_start=np.float64(t_start),
    #                                               t_last_spike_p=np.float64(t_last_spike))
    #     return t_spikes

    # def filter_signal_2(self, input_sig, do_plot=0, v_init=[], t_start=0, t_last_spike=[]):
    #     """ Filter all the channels at the same time and not 1 by 1 """
    #     input_sig = np.array(input_sig).squeeze()
    #     if len(input_sig.shape) == 1:
    #         input_sig = np.tile(input_sig, (self.n_channels, 1))
    #     if not input_sig.shape[0] == self.n_channels:
    #         raise ValueError('input_sig must be a [n_channels, n_pnts] matrice')
    #     if not np.array(v_init).any():
    #         v_init = np.zeros(self.n_channels)
    #     elif not np.array(v_init).shape[0] == self.n_channels:
    #         raise ValueError('v_init must be a [n_channels, 1] array')
    #     if not np.array(t_last_spike).any():
    #         t_last_spike = -2 * self.t_refract * np.ones(self.n_channels)
    #     elif not t_last_spike.shape[0] == self.n_channels:
    #         raise ValueError('t_last_spike must be a [n_channels, 1] array')
    #
    #     n_pnts, n_chan = input_sig.shape[1], self.n_channels
    #     output_sig = np.zeros((self.n_channels, n_pnts))
    #     t_last_spike_out = np.zeros(self.n_channels)
    #     spike_time, spike_chan = [], []
    #     dt = 1/self.fs
    #     t_last_spike = t_start - 2 * self.t_refract * np.ones(n_chan)
    #     v_init = 0
    #     tvect = np.linspace(t_start, t_start + n_pnts * dt, n_pnts)
    #     v_out = np.zeros((n_chan, n_pnts))
    #
    #     v_mult = (self.tau / dt) / (self.tau / dt + 1)
    #     i_mult = 1 / (1 + self.tau / dt)
    #     i_syn = input_sig
    #
    #     for i, t in enumerate(tvect):
    #         for c in range(n_chan):
    #             if self.refract_period and t < (t_last_spike[c] + self.t_refract[c]):  # in refractory period
    #                 v_out[c, i] = 0
    #             elif i > 0 and t_last_spike[c] == tvect[i-1]:  # Reset potential mode Spiking activity just occured
    #                 v_out[c, i] = self.v_reset[c]
    #             else:
    #                 if i == 0:
    #                     v_out[c, i] = v_init * v_mult[c] + i_syn[c, i - 1] * i_mult[c]
    #                 else:
    #                     v_out[c, i] = v_out[c, i - 1] * v_mult[c] + i_syn[c, i - 1] * i_mult[c]
    #                 if v_out[c, i] > self.v_thresh[c]:  # Spike
    #                     v_out[c, i] = self.v_spike[c]
    #                     t_last_spike[c] = t
    #                     spike_time.append(t)
    #                     spike_chan.append(c)
    #     spikes = np.vstack([np.array(spike_time), np.array(spike_chan)]).T
    #     return v_out, spikes, t_last_spike

    # def filter_signal_with_inhib(self, input_sig, do_plot=0, v_init=[], t_start=0, t_last_spike=[], inhib_vect=[]):
    #     """ Filter all the channels at the same time and not 1 by 1 """
    #     input_sig = np.array(input_sig).squeeze()
    #     if len(input_sig.shape) == 1:
    #         input_sig = np.tile(input_sig, (self.n_channels, 1))
    #     if not input_sig.shape[0] == self.n_channels:
    #         raise ValueError('input_sig must be a [n_channels, n_pnts] matrice')
    #     if not np.array(v_init).any():
    #         v_init = np.zeros(self.n_channels)
    #     elif not np.array(v_init).shape[0] == self.n_channels:
    #         raise ValueError('v_init must be a [n_channels, 1] array')
    #     if not np.array(t_last_spike).any():
    #         t_last_spike = -2 * self.t_refract * np.ones(self.n_channels)
    #     elif not t_last_spike.shape[0] == self.n_channels:
    #         raise ValueError('t_last_spike must be a [n_channels, 1] array')
    #
    #     n_pnts, n_chan = input_sig.shape[1], self.n_channels
    #     spike_time, spike_chan = [], []
    #     dt = 1/self.fs
    #     t_last_spike = t_start - 2 * self.t_refract * np.ones(n_chan)
    #     v_init = 0
    #     tvect = np.linspace(t_start, t_start + n_pnts * dt, n_pnts)
    #     v_temp, v_out = np.zeros((2, n_chan, n_pnts))
    #
    #     v_mult = (self.tau / dt) / (self.tau / dt + 1)
    #     i_mult = 1 / (1 + self.tau / dt)
    #     i_syn = input_sig
    #     n_inhib = int(np.round(len(inhib_vect)/2))
    #     for i, t in enumerate(tvect):
    #         for c in range(n_chan):
    #             if self.refract_period and t < (t_last_spike[c] + self.t_refract[c]):  # in refractory period
    #                 v_temp[c, i] = 0
    #             elif i > 0 and t_last_spike[c] == tvect[i-1]:  # Reset potential mode Spiking activity just occured
    #                 v_temp[c, i] = self.v_reset[c]
    #             else:
    #                 if i == 0:
    #                     v_temp[c, i] = v_init * v_mult[c] + i_syn[c, i - 1] * i_mult[c]
    #                 else:
    #                     v_temp[c, i] = v_out[c, i - 1] * v_mult[c] + i_syn[c, i - 1] * i_mult[c]
    #         # Do inhibition :
    #         for c in range(n_chan):
    #             if self.refract_period and t < (t_last_spike[c] + self.t_refract[c]):  # in refractory period
    #                 v_out[c, i] = v_temp[c, i]
    #             elif i > 0 and t_last_spike[c] == tvect[i-1]:  # Reset potential mode Spiking activity just occured
    #                 v_out[c, i] = v_temp[c, i]
    #             else:
    #                 if i == 0:
    #                     v_out[c, i] = v_temp[c, i]
    #                 else:  # Inhibition
    #                     if n_inhib <= c < (n_chan-n_inhib):
    #                         v_out[c, i] = v_temp[c, i] + np.sum(v_temp[c-n_inhib:c+1+n_inhib, i] * inhib_vect)
    #                 if v_out[c, i] > self.v_thresh[c]:  # Spike
    #                     v_temp[c, i] = self.v_spike[c]
    #                     t_last_spike[c] = t
    #                     spike_time.append(t)
    #                     spike_chan.append(c)
    #
    #     spikes = np.vstack([np.array(spike_time), np.array(spike_chan)]).T
    #     return v_out, spikes, t_last_spike

    # def filter_1d_signal(self, isyn, tau, v_thresh, v_spike, t_refract, v_reset, v_init=0, t_start=0, t_last_spike=[]):
    #     dt = 1/self.fs
    #     tvect = np.linspace(t_start, t_start + len(isyn) * dt, len(isyn))
    #     v_mult = (tau/dt) / (tau/dt + 1)
    #     i_mult = 1 / (1 + tau/dt)
    #     if not t_last_spike:
    #         t_last_spike = t_start - 2 * t_refract
    #     t_spikes = []
    #     v_out = np.zeros(len(isyn))
    #     for i, t in enumerate(tvect):
    #         if self.refract_period and t < (t_last_spike + t_refract):  # Refractory period
    #             v_out[i] = 0
    #         elif not self.refract_period and i > 0 and t_last_spike == tvect[i-1]:  # Spiking activity just occured
    #             v_out[i] = v_reset
    #         else:
    #             if i == 0:
    #                 v_out[i] = v_init * v_mult + isyn[i-1] * i_mult
    #             else:
    #                 v_out[i] = v_out[i-1] * v_mult + isyn[i-1] * i_mult
    #             if v_out[i] > v_thresh:  # Spike
    #                 v_out[i] = v_spike
    #                 t_last_spike = t
    #                 t_spikes.append(t)
    #     return v_out, t_spikes

    def plot_spikes(self, spike_time, spike_channel, tmin, tmax, bin_duration=0.002, potential=[]):
        fig = plt.figure()
        ax0 = plt.subplot2grid((4, 7), (0, 1), rowspan=3, colspan=5)
        if np.array(potential).any():
            plt.plot(spike_time, spike_channel, 'w.', markersize=4)
        else:
            plt.plot(spike_time, spike_channel, '.', markersize=4)
        ax0.set_ylim(0, self.n_channels)
        ax0.set(title='Raster plot')
        if np.array(potential).any():
            im_pot = ax0.imshow(potential, origin='lower', aspect='auto', extent=(tmin, tmax, 0, self.n_channels), alpha=1)
            ax0.grid(False)
            cb_ax = fig.add_axes((0.845, 0.13, 0.02, 0.12))
            plt.colorbar(im_pot, cax=cb_ax)
        ax1 = plt.subplot2grid((4, 7), (3, 1), rowspan=1, colspan=5, sharex=ax0)
        ax1.hist(spike_time, bins=int((tmax - tmin) / bin_duration))
        ax1.set_xlim(tmin, tmax)
        ax1.set(xlabel='Time (s)', ylabel='count')
        ax3 = plt.subplot2grid((4, 7), (0, 6), rowspan=3, sharey=ax0)
        spikes_per_channel = [np.sum(spike_channel == i) for i in range(0, self.n_channels)]
        ax3.barh(range(0, self.n_channels), spikes_per_channel, height=1)
        ax3.set_ylim((0, self.n_channels))
        ax3.set(xlabel='count', ylabel='Channel')
        ax3.invert_xaxis()
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.tick_right()
        ax4 = plt.subplot2grid((4, 7), (0, 0), rowspan=3, colspan=1, sharey=ax0)
        ax4.set(ylabel='Channel', xlabel='Median ISI (ms)')
        ax4.barh(range(0, self.n_channels), 1000 * self.get_median_isi(spike_time, spike_channel), 1)
        ax4.autoscale(axis='y', tight=True)
        fig.show()

    def get_median_isi(self, spike_time, spike_channel):
        median_isi = np.zeros(self.n_channels)
        for i in range(0, self.n_channels):
            spike_times_i = spike_time[spike_channel == i]
            if len(spike_times_i) > 1:
                median_isi[i] = np.median(spike_times_i[1:] - spike_times_i[:-1])
        return median_isi

    def plot_caracteristics(self):
        f = plt.figure()
        ax1 = f.add_subplot(211)
        ax1.plot(1000 * self.tau)
        ax1.plot(1000 * self.t_refract)
        ax1.autoscale(axis='x', tight=True)
        ax1.set(xlabel='Channel', title='Time constant for each LIF neuron', ylabel='(ms)')
        plt.legend(['Time constant', 'Refractory Period'])
        ax2 = f.add_subplot(212)
        ax2.plot(self.v_thresh)
        ax2.autoscale(axis='x', tight=True)
        ax2.legend(['Spiking Threshold'])
        ax2.set(xlabel='Channel', title='Spiking Threshold for each LIF neuron')
        f.show(block=False)


class Cochlea:
    __slots__ = ['n_channels', 'freq_scale', 'fs', 'fmin', 'fmax', 'forder', 'cf', 'filterbank', 'rectifierbank',
                 'compressionbank', 'lifbank']

    def __init__(self, n_channels, fs, fmin, fmax, freq_scale, order=2, fbank_type='butter', rect_type='full',
                 rect_lowpass_freq=[], comp_factor=1.0/3, comp_gain=1, lif_tau=0.010, lif_v_thresh=0.020,
                 lif_v_spike=0.035, lif_t_refract=0.001, lif_v_reset=[], inhib_type=[], inhib_vect=[],
                 tau_j=[], alpha_j=[], omega=[]):
        if freq_scale.lower() not in ['erbscale', 'linearscale', 'musicscale']:
            raise ValueError('freq_scale argument can take the values : erbscale or linearscale or musicscale')
        self.n_channels = n_channels
        self.freq_scale = freq_scale.lower()
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.forder = order
        if self.freq_scale == 'erbscale':
            wn, self.cf = erbscale(fs, fmin, fmax, n_channels)
        elif self.freq_scale == 'linearscale':
            wn, self.cf = linearscale(fs, fmin, fmax, n_channels)
        elif self.freq_scale == 'musicscale':
            wn, self.cf = musicscale(fs, fmin, fmax, n_channels)
        self.filterbank = BandPassFilterbank(order, wn, fs, ftype=fbank_type)
        self.rectifierbank = RectifierBank(n_channels, fs, rect_type, rect_lowpass_freq)
        self.compressionbank = CompressionBank(n_channels, fs, comp_factor, comp_gain)
        self.lifbank = LIFBank(n_channels, fs, lif_tau, lif_v_thresh, lif_v_spike, lif_t_refract, lif_v_reset,
                               inhib_type, inhib_vect, alpha_j=alpha_j, tau_j=tau_j, omega=omega)

    def __str__(self):
        desc_str = 'Cochlea model - {} channels [{} - {} Hz] - {} - {} order Butterworth filters\n'.format(
            self.n_channels, self.fmin, self.fmax, self.freq_scale, self.forder)
        desc_str += self.rectifierbank.__str__()
        desc_str += self.compressionbank.__str__()
        desc_str += self.lifbank.__str__()
        return desc_str

    def save(self, dirpath, filename=[]):
        if not path.isdir(dirpath):
            print('Creating save directory : {}'.format(dirpath))
            mkdir(dirpath)
        if not filename:
            filename = 'cochlea_model_{}.p'.format(datetime.strftime(datetime.now(), '%d%m%y_%H%M'))
        with open(path.join(dirpath, filename), 'wb') as f:
            _pickle.dump(self, f)

    def plot_filterbank_frequency_response(self):
        ax = self.filterbank.plot_frequency_response()
        ax.set(title='Filters frequency responses - {} - {} filters in [{}, {} Hz] - order {}'.format(
            self.freq_scale, self.n_channels, self.fmin, self.fmax, self.forder))

    def plot_channel_evolution(self, input_sig, channel_pos=[]):
        """ Plot the processing of input_sig signal through the cochlea, for channels specified by channel_pos.
        If channel_pos is not provided, 4 channels are selected ranging from low-frequency channel to high-frequency
        channel.

        Parameters
        ----------
        input_sig : array [1D]
            Input signal
        channel_pos : scalar | array | list
            Position of the channel(s) to plot. If multiples channels are selected, plot one figure per channel.
            If none,  4 channels are selected ranging from low-frequency channel to high-frequency channel.

        """
        if len(np.array(input_sig).shape) > 1 and not np.min(np.array(input_sig).shape) == 1:
            raise ValueError('input_sig must be a 1D array')
        if not channel_pos:
            channel_pos = np.round(self.n_channels*np.array([0.1, 0.4, 0.6, 0.9])).astype(int)
        if not np.isscalar(channel_pos):
            for chan_pos in channel_pos:
                self.plot_channel_evolution(input_sig, chan_pos)
        else:
            y_filt, y_rect, y_comp, v_out, threshold, t_spikes = self.process_one_channel(input_sig, channel_pos)
            tvect = np.linspace(0, len(input_sig) / self.fs, len(input_sig))
            # Plot
            f = plt.figure()
            ax = plt.subplot2grid((3, 1), (0, 0), rowspan=1)
            ax.plot(tvect, input_sig)
            ax.vlines(t_spikes, ymin=0, ymax=1.3*np.max(input_sig), zorder=3)
            ax.autoscale(axis='x', tight=True)
            ax.set(title='Input signal - channel {} [f = {} Hz]'.format(channel_pos, np.round(self.cf[channel_pos])),
                   ylabel='Amplitude')
            axx = ax.twinx()
            spkrate = t_spikes_to_spikerate(t_spikes, self.fs, input_sig.size)
            axx.plot(tvect, spkrate, color='r', alpha=0.5)
            axx.grid(False)
            ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2, sharex=ax)
            ax2.plot(tvect, y_filt, alpha=0.8)
            ax2.plot(tvect, y_rect, 'y', alpha=0.8)
            ax2.plot(tvect, y_comp, alpha=0.8)
            ax2.plot(tvect, v_out, 'k')
            if hasattr(self.lifbank, 'tau_j') and self.lifbank.tau_j.size > 0:
                ax2.plot(tvect, threshold)
            ax2.autoscale(axis='x', tight=True)
            lowpassfreq = self.rectifierbank.lowpass_freq[channel_pos] if self.rectifierbank.lowpass_freq else 'None'
            if self.compressionbank.comp_factor[channel_pos] < 1:
                comp_transform = '$y ={} * x ^{{ 1 / {:.0f}}}$'.format(self.compressionbank.comp_gain[channel_pos],
                                                                       1/self.compressionbank.comp_factor[channel_pos])
            else:
                comp_transform = '$y ={}* x ^{{ {:.2f} }} $'.format(self.compressionbank.comp_gain[channel_pos],
                                                                    self.compressionbank.comp_factor[channel_pos])
            ax2.legend(['Filtered - cf = {:.0f}Hz'.format(self.cf[channel_pos]),
                        'Rectified - {} - lpf = {}'.format(self.rectifierbank.rtype, lowpassfreq),
                        'Compressed - {}'.format(comp_transform),
                        'LIF output - $ \\tau = {:.1f} ms $'.format(1000 * self.lifbank.tau[channel_pos])])
            ax2.set(title='Cochlea signals', ylabel='Amplitude', xlabel='Time (s)')
            f.show()

    def process_one_channel(self, input_sig, channel_pos, do_lif=1):
        v_out, spikes, threshold = [], [], []
        input_sig = np.array(input_sig)
        if not input_sig.ndim == 1:
            raise ValueError('Argument input_sig should be 1D')
        if input_sig.max() > 10 or input_sig.min() < -10:
            print('Be sure to normalize the signal before applying the cochlea')
        sig_filtered, _ = self.filterbank.filter_one_channel(input_sig, channel_pos)
        sig_rectify = self.rectifierbank.rectify_one_channel(sig_filtered, channel_pos)
        sig_compressed = self.compressionbank.compress_one_channel(sig_rectify, channel_pos)
        if do_lif:
            if hasattr(self.lifbank, 'tau_j') and self.lifbank.tau_j.size > 0:
                v_out, t_spikes, threshold = self.lifbank.filter_one_channel_adaptive_threshold(sig_compressed, channel_pos)
            else:
                v_out, t_spikes = self.lifbank.filter_one_channel(sig_compressed, channel_pos)
        return sig_filtered, sig_rectify, sig_compressed, v_out, threshold, t_spikes
        # return spikes, sig_compressed

    # def process_one_channel_adaptive_threshold(self, input_sig, channel_pos, do_lif=1):
    #     input_sig = np.array(input_sig)
    #     if not input_sig.ndim == 1:
    #         raise ValueError('Argument input_sig should be 1D')
    #     if input_sig.max() > 10 or input_sig.min() < -10:
    #         print('Be sure to normalize the signal before applying the cochlea')
    #     sig_filtered, _ = self.filterbank.filter_one_channel(input_sig, channel_pos)
    #     sig_rectify = self.rectifierbank.rectify_one_channel(sig_filtered, channel_pos)
    #     sig_compressed = self.compressionbank.compress_one_channel(sig_rectify, channel_pos)
    #     if do_lif:
    #         v_out, t_spikes, threshold = self.lifbank.filter_one_channel_adaptive_threshold(sig_compressed, channel_pos)

        # return sig_filtered, sig_rectify, sig_compressed, v_out, spikes
        # return t_spikes, v_out, sig_compressed, threshold


    # @timethis
    # def process_input_with_inhib(self, input_sig, do_plot=0, inhib_type='sub_for', inhib_vect=[]):
    #     if not input_sig.ndim == 1:
    #         raise ValueError('Argument input_sig should be 1D')
    #     inhib_vect = np.array(inhib_vect)
    #     if inhib_vect.size == 0:
    #         raise ValueError('Argument inhib_vect must be defined')
    #     sig_filtered, _ = self.filterbank.filter_signal(input_sig, do_plot=do_plot)
    #     sig_rectified = self.rectifierbank.rectify_signal(sig_filtered, do_plot=do_plot)
    #     sig_comp = self.compressionbank.compress_signal(sig_rectified, do_plot=do_plot)
    #     lif_out_sig, spikes, _ = self.lifbank.filter_signal_with_inhib_cy(sig_comp, do_plot=do_plot,
    #                                                                       inhib_type=inhib_type, inhib_vect=inhib_vect)
    #     tmin, tmax = 0, input_sig.size / self.fs
    #     spike_list = SpikeList(time=spikes[:, 0], channel=spikes[:, 1], n_channels=self.n_channels, name=self.__str__(),
    #                            tmin=tmin, tmax=tmax)
    #     return spike_list, lif_out_sig

    @timethis
    def process_input(self, input_sig, do_plot=0, samplever=1):
        input_sig = np.array(input_sig)
        if not input_sig.ndim == 1:
            raise ValueError('Argument input_sig should be 1D')
        sig_filtered, _ = self.filterbank.filter_signal(input_sig, do_plot=do_plot)
        sig_rectified = self.rectifierbank.rectify_signal(sig_filtered, do_plot=do_plot)
        sig_comp = self.compressionbank.compress_signal(sig_rectified, do_plot=do_plot)
        if hasattr(self.lifbank, 'inhib_type') and self.lifbank.inhib_type and hasattr(self.lifbank, 'tau_j') and self.lifbank.tau_j.size > 0:
            lif_out_sig, spikes, threshold = self.lifbank.filter_signal_adaptive_threshold(sig_comp, samplever=samplever, inhib_shuntfor=1)
        elif hasattr(self.lifbank, 'inhib_type') and self.lifbank.inhib_type:
            lif_out_sig, spikes, _ = self.lifbank.filter_signal_with_inhib_cy(sig_comp, do_plot=do_plot, mp_ver=1)
        elif hasattr(self.lifbank, 'tau_j') and self.lifbank.tau_j.size > 0:
            lif_out_sig, spikes, threshold = self.lifbank.filter_signal_adaptive_threshold(sig_comp, samplever=samplever)
        else:
            lif_out_sig, spikes, _ = self.lifbank.filter_signal(sig_comp, do_plot=do_plot)
        tmin, tmax = 0, input_sig.size / self.fs
        spike_list = SpikeList(time=spikes[:, 0], channel=spikes[:, 1], n_channels=self.n_channels, name=self.__str__(),
                               tmin=tmin, tmax=tmax)
        return spike_list


    @timethis
    def process_input_mpver(self, input_sig, n_processes=6):
        """ Process an input signal channel by channel using the multiprocessing module"""
        input_sig = np.array(input_sig)
        if not input_sig.ndim == 1:
            raise ValueError('Argument input_sig should be 1D')
        tmin, tmax = 0, input_sig.size / self.fs
        pool = mp.Pool(processes=n_processes)
        if hasattr(self.lifbank, 'inhib_type') and self.lifbank.inhib_type:
            async_out = [pool.apply_async(self.process_one_channel, args=(input_sig, i, 0)) for i in
                         range(self.n_channels)]
            sig_comp = np.vstack([p.get()[2] for p in async_out])
            _, spikes, _ = self.lifbank.filter_signal_with_inhib_cy(sig_comp, mp_ver=1)
            spike_list = SpikeList(time=spikes[:, 0], channel=spikes[:, 1], n_channels=self.n_channels,
                                   name=self.__str__(), tmin=tmin, tmax=tmax)
        else:
            async_out = [pool.apply_async(self.process_one_channel, args=(input_sig, i)) for i in range(self.n_channels)]
            t_spikes = [p.get()[-1] for p in async_out]
            t_spikes_arr = np.hstack(t_spikes)
            spike_channel = np.hstack([i * np.ones(t_spikes[i].size) for i in range(self.n_channels)])
            spike_list = SpikeList(time=t_spikes_arr, channel=spike_channel, n_channels=self.n_channels,
                                   name=self.__str__(), tmin=tmin, tmax=tmax)
        return spike_list

    def process_test_signal(self, signal_type, channel_pos=[], do_plot=True, **kwargs):
        """ Run a test signal through the cochlea. Test signals can be a sinusoidal signal, a step or an impulse signal.
        Argument signal_type selects the wanted signal.
        If channel_pos is provided and do_plot is True, channel evolution is plot.

        Parameters
        ----------
        signal_type : str
            Test signal type. Can be
        channel_pos :
        do_plot :
        kwargs :

        Returns
        -------

        """
        channel_pos = np.atleast_1d(channel_pos)
        kwargs_keys = list(kwargs.keys())
        t_offset = kwargs['t_offset'] if 't_offset' in kwargs_keys else 0.2
        t_max = kwargs['t_max'] if 't_max' in kwargs_keys else 1
        amplitude = kwargs['amplitude'] if 'amplitude' in kwargs_keys else 1
        # Get test signal
        if signal_type.lower() in ['sin', 'sinus']:
            f_sin = kwargs['f_sin'] if 'f_sin' in kwargs_keys else self.fs / 4
            x_test = generate_sinus(self.fs, f_sin, t_offset, t_max, amplitude)
        elif signal_type.lower() in ['dirac', 'impulse']:
            x_test = generate_dirac(self.fs, t_offset, t_max, amplitude)
        elif signal_type.lower() == 'step':
            x_test = generate_step(self.fs, t_offset, t_max, amplitude)
        # Run the test signal through the cochlea
        if channel_pos.size == 0:
            spike_list = self.process_input(x_test, do_plot=do_plot)
        else:
            spike_list = self.process_input(x_test, do_plot=False)
            if do_plot:
                self.plot_channel_evolution(x_test, channel_pos)
        return spike_list

    def process_input_block_ver(self, input_sig, block_len, plot_spikes=1):
        n_blocks = int(np.ceil(len(input_sig) / block_len))
        block_duration = block_len / self.fs
        zi, t_last_spikes = [], []
        v_init = np.zeros(self.n_channels)
        spikes = []
        for i in range(0, n_blocks):
            block_i = input_sig[i*block_len: np.min([len(input_sig), (i+1)*block_len])]
            block_i_filtered, zi = self.filterbank.filter_signal(block_i, do_plot=0, zi_in=zi)
            block_i_rectified = self.rectifierbank.rectify_signal(block_i_filtered, do_plot=0)
            block_i_compressed = self.compressionbank.compress_signal(block_i_rectified, do_plot=0)
            v_out_i, spikes_i, t_last_spikes = self.lifbank.filter_signal(block_i_compressed, do_plot=0, v_init=v_init,
                                                                          t_start=i*block_duration,
                                                                          t_last_spike=t_last_spikes)
            v_init = v_out_i[:, -1]
            spikes.append(spikes_i)
        spikes = np.vstack(np.array(spikes))
        spike_list = SpikeList(time=spikes[:, 0], channel=spikes[:, 1], n_channels=self.n_channels, name=self.__str__())
        if plot_spikes:
            spike_list.plot()
        return spike_list, _


def load_cochlea(dirpath, filename):
    with open(path.join(dirpath, filename), 'rb') as f:
        return _pickle.load(f)


class CochleaEstimator:
    def __init__(self, fs, n_filters, fmin, fmax, freq_scale=['linearscale', 'erbscale'], forder=[2],
                 fbank_type=['butter', 'bessel'], comp_gain=np.arange(0.8, 1, 1.2), comp_factor=1.0 / np.array([3, 4, 5]),
                 lif_tau_coeff=[0.5, 1, 3], lif_v_thresh_start=[0.3, 0.4, 0.5], lif_v_thresh_stop=[0.1, 0.15],
                 lif_v_thresh_mid=[]):
        self.fs = fs
        self.n_channels = np.atleast_1d(n_filters)
        self.fmin, self.fmax = np.atleast_1d(fmin), np.atleast_1d(fmax)
        self.freq_scale, self.forder = np.atleast_1d(freq_scale), np.atleast_1d(forder)
        self.fbank_type = np.atleast_1d(fbank_type)
        self.comp_gain, self.comp_factor = np.atleast_1d(comp_gain), np.atleast_1d(comp_factor)
        self.lif_tau_coeff = np.atleast_1d(lif_tau_coeff)
        self.lif_v_tresh_start, self.lif_v_thresh_stop = np.atleast_1d(lif_v_thresh_start), np.atleast_1d(lif_v_thresh_stop)
        self.lif_v_thresh_mid = np.atleast_1d(lif_v_thresh_mid)
        self.param_grid_dict = {'n_channels': self.n_channels, 'freq_scale': self.freq_scale, 'fmin': self.fmin,
                                'fmax': self.fmax, 'forder': self.forder, 'fbank_type': self.fbank_type,
                                'comp_gain': self.comp_gain, 'comp_factor': self.comp_factor,
                                'lif_tau_coeff': self.lif_tau_coeff, 'lif_v_thresh_start': self.lif_v_tresh_start,
                                'lif_v_thresh_stop': self.lif_v_thresh_stop, 'lif_v_thresh_mid': self.lif_v_thresh_mid}
        param_grid = ParameterGrid(self.param_grid_dict)
        self.n_combinations = len(param_grid)

    def __str__(self):
        np.set_printoptions(precision=2)
        desc_str = 'Cochlea Estimator for finding the best parameters\n'
        desc_str += ' - n_channels : {}, scale : {}\n'.format(self.n_channels, self.freq_scale)
        desc_str += ' - fmin = {} Hz, fmax = {} Hz. type : {}, order : {}\n'.format(self.fmin, self.fmax,
                                                                                    self.fbank_type, self.forder)
        desc_str += ' - Compression gain : {}, factor : {}\n'.format(self.comp_gain, self.comp_factor)
        desc_str += ' - LIF tau : 1/cf * {}\n'.format(self.lif_tau_coeff)
        desc_str += ' - LIF threshold start : {}, end : {}\n'.format(self.lif_v_tresh_start, self.lif_v_thresh_stop)
        if self.lif_v_thresh_mid.size > 0:
            desc_str += ' - LIF threshold middle : {}\n'.format(self.lif_v_thresh_mid)
        desc_str += '{} sets of parameters'.format(self.n_combinations)
        return desc_str

    def process_input(self, cochlea, input_sig, t_chunk_start, t_chunk_end, pattern_pos, block_len_max=50000,
                      w_vr=4, w_spkchanvar=1):
        if len(input_sig) > block_len_max:
            print('WARNING not fully implemented')
            spike_list, _ = cochlea.process_input_block_ver(input_sig, block_len_max, plot_spikes=0)
        else:
            spike_list, _ = cochlea.process_input(input_sig, do_plot=0)
        if spike_list.n_spikes == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        epochs = spike_list.epoch(t_chunk_start, t_chunk_end)
        epochs_pattern = epochs[pattern_pos]
        channel_with_spikes_ratio = epochs_pattern.compute_channel_with_spikes()
        _, _, _,  vr_score_pattern = epochs_pattern.compute_vanrossum_distance()
        _, spkchanvar_score_pattern = epochs_pattern.compute_spike_per_channel_variation()
        global_score = w_vr * vr_score_pattern + w_spkchanvar * spkchanvar_score_pattern
        mean_n_spike_pattern = np.mean([spklist.n_spikes for spklist in epochs_pattern])
        return global_score, vr_score_pattern, spkchanvar_score_pattern, channel_with_spikes_ratio, mean_n_spike_pattern

    def grid_search_atom(self, param_i, input_sig, t_chunk_start, t_chunk_end, pattern_pos):
        if param_i['freq_scale'] == 'linearscale':
            _, cf_i = linearscale(self.fs, param_i['fmin'], param_i['fmax'], param_i['n_channels'])
        elif param_i['freq_scale'] == 'erbscale':
            _, cf_i = erbscale(self.fs, param_i['fmin'], param_i['fmax'], param_i['n_channels'])
        else:
            raise ValueError('Wrong parameter freq_scale : '.format(param_i['freq_scale']))
        lif_tau = param_i['lif_tau_coeff'] * 1 / cf_i
        if self.lif_v_thresh_mid.size == 0:
            lif_v_thresh = np.linspace(param_i['lif_v_thresh_start'], param_i['lif_v_thresh_stop'], self.n_channels)
        else:
            lif_v_thresh = np.interp(np.linspace(0, param_i['n_channels']-1, param_i['n_channels']),
                                     [0, np.round(param_i['n_channels']/2), param_i['n_channels']-1],
                                     [param_i['lif_v_thresh_start'], param_i['lif_v_thresh_mid'],
                                      param_i['lif_v_thresh_stop']])
        # Construct cochlea model
        cochlea_i = Cochlea(n_channels=param_i['n_channels'], fs=self.fs, fmin=param_i['fmin'],
                            fmax=param_i['fmax'],
                            freq_scale=param_i['freq_scale'], order=param_i['forder'],
                            fbank_type=param_i['fbank_type'],
                            comp_factor=param_i['comp_factor'], comp_gain=param_i['comp_gain'], lif_tau=lif_tau,
                            lif_v_thresh=lif_v_thresh)
        global_score, vr_score, spkchanvar_score, channel_with_spikes_ratio, n_spikes_mean = \
            self.process_input(cochlea_i, input_sig, t_chunk_start, t_chunk_end, pattern_pos)
        return global_score, vr_score, spkchanvar_score, channel_with_spikes_ratio, n_spikes_mean

    def grid_search(self, input_sig, t_chunk_start, t_chunk_end, pattern_pos, input_sig_name=''):
        param_grid = ParameterGrid(self.param_grid_dict)
        n_comb = self.n_combinations
        global_score, vr_score, spkchanvar_score = np.zeros(n_comb), np.zeros(n_comb), np.zeros(n_comb)
        channel_with_spikes_ratio, n_spikes_mean = np.zeros(n_comb), np.zeros(n_comb)
        for i in tqdm.tqdm(range(0, n_comb)):
            # Get parameters for i-th values
            param_i = param_grid[i]
            if param_i['freq_scale'] == 'linearscale':
                _, cf_i = linearscale(self.fs, param_i['fmin'], param_i['fmax'], param_i['n_channels'])
            elif param_i['freq_scale'] == 'erbscale':
                _, cf_i = erbscale(self.fs, param_i['fmin'], param_i['fmax'], param_i['n_channels'])
            else:
                raise ValueError('Wrong parameter freq_scale : '.format(param_i['freq_scale']))
            lif_tau = param_i['lif_tau_coeff'] * 1 / cf_i
            if self.lif_v_thresh_mid.size == 0:
                lif_v_thresh = np.linspace(param_i['lif_v_thresh_start'], param_i['lif_v_thresh_stop'], self.n_channels)
            else:
                lif_v_thresh = np.interp(np.linspace(0, param_i['n_channels'] - 1, param_i['n_channels']),
                                         [0, np.round(param_i['n_channels'] / 2), param_i['n_channels'] - 1],
                                         [param_i['lif_v_thresh_start'], param_i['lif_v_thresh_mid'],
                                          param_i['lif_v_thresh_stop']])
            # Construct cochlea model
            cochlea_i = Cochlea(n_channels=param_i['n_channels'], fs=self.fs, fmin=param_i['fmin'], fmax=param_i['fmax'],
                                freq_scale=param_i['freq_scale'], order=param_i['forder'], fbank_type=param_i['fbank_type'],
                                comp_factor=param_i['comp_factor'], comp_gain=param_i['comp_gain'], lif_tau=lif_tau,
                                lif_v_thresh=lif_v_thresh)
            global_score[i], vr_score[i], spkchanvar_score[i], channel_with_spikes_ratio[i], n_spikes_mean[i] = \
                self.process_input(cochlea_i, input_sig, t_chunk_start, t_chunk_end, pattern_pos)
        # Create a pandas DataFrame with all the results (one row per cochlea)
        f_ev = lambda x: [p[x] for p in param_grid]
        data = {'Filename': input_sig_name, 'fs': self.fs, 'n_channels': f_ev('n_channels'), 'freq_scale': f_ev('freq_scale'),
                'fbank_type': f_ev('fbank_type'), 'forder': f_ev('forder'), 'fmin': f_ev('fmin'), 'fmax': f_ev('fmax'),
                'comp_factor': f_ev('comp_factor'), 'comp_gain': f_ev('comp_gain'), 'lif_tau_coeff': f_ev('lif_tau_coeff'),
                'lif_v_thresh_start': f_ev('lif_v_thresh_start'), 'lif_v_thresh_stop': f_ev('lif_v_thresh_stop'),
                'global_score': global_score, 'vr_score': vr_score, 'spkchanvar_score': spkchanvar_score,
                'channel_with_spikes_ratio': channel_with_spikes_ratio, 'n_spikes_mean': n_spikes_mean}
        df = pd.DataFrame(data, columns=['Filename', 'fs', 'n_channels', 'freq_scale', 'fbank_type', 'forder', 'fmin',
                                         'fmax', 'comp_factor', 'comp_gain', 'lif_tau_coeff', 'lif_v_thresh_start',
                                         'lif_v_thresh_stop', 'global_score', 'vr_score', 'spkchanvar_score',
                                         'channel_with_spikes_ratio', 'n_spikes_mean'])
        return global_score, vr_score, spkchanvar_score, channel_with_spikes_ratio, n_spikes_mean, param_grid, df

    def plot_param_scores(self, global_score, vr_score, spkchanvar_score, channel_with_spikes_ratio,
                          n_spikes_mean, param_key=[], inner='box'):
        param_key = np.atleast_1d(param_key)
        if param_key.size == 0:
            param_key = np.array(list(self.param_grid_dict.keys()))
        if param_key.size > 1:
            for param_key_i in param_key:
                self.plot_param_scores(global_score, vr_score, spkchanvar_score, channel_with_spikes_ratio,
                                       n_spikes_mean, param_key=param_key_i, inner=inner)
        elif param_key.size == 1:
            param_key = str(param_key[0])
            param_grid = ParameterGrid(self.param_grid_dict)
            param_ev = np.array([param_grid[i][param_key] for i in range(0, len(param_grid))])
            data = {'Global Score': global_score, 'VR Score': vr_score, 'CV(spk_per_channel) Score': spkchanvar_score,
                    'Channel With Spike': channel_with_spikes_ratio, 'Number of Spikes': n_spikes_mean, param_key: param_ev}
            df = pd.DataFrame(data)
            f = plt.figure()
            ax0 = plt.subplot2grid((2, 6), (0, 0), colspan=2)
            sns.violinplot(x=param_key, y='Global Score', data=df, inner=inner, ax=ax0)
            ax1 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
            sns.violinplot(x=param_key, y='VR Score', data=df, inner=inner, ax=ax1)
            ax2 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
            sns.violinplot(x=param_key, y='CV(spk_per_channel) Score', data=df, inner=inner, ax=ax2)
            ax3 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
            sns.violinplot(x=param_key, y='Channel With Spike', data=df, inner=inner, ax=ax3)
            ax4 = plt.subplot2grid((2, 6), (1, 3), colspan=2)
            sns.violinplot(x=param_key, y='Number of Spikes', data=df, inner=inner, ax=ax4)
            ax1.set(title='Cochlea parameter : {}'.format(param_key))
            f.show()

    def plot_results(self, global_score, vr_score, spkchanvar_score, channel_with_spikes_ratio, n_spikes_mean,
                     param_grid, params_to_plot=['fmin', 'fmax', 'forder', 'freq_scale', 'fbank_type',
                                                 'comp_gain', 'comp_factor',
                                                 'lif_tau_coeff', 'lif_v_thresh_start', 'lif_v_thresh_stop',
                                                 'lif_v_thresh_mid'], order=[]):
        n_param_sets = len(vr_score)
        no_spikes_ind = n_spikes_mean == 0
        vr_score[no_spikes_ind], spkchanvar_score[no_spikes_ind], global_score[no_spikes_ind] = np.nan, np.nan, np.nan
        n_spikes_mean[no_spikes_ind], channel_with_spikes_ratio[no_spikes_ind] = np.nan, np.nan
        # Ordering
        if not order:
            sort_vect = np.arange(0, n_param_sets)
        elif order == 'global_score' or order == 1:
            sort_vect = np.argsort(global_score)
            score_sel = global_score
        elif order == 'vr_score':
            sort_vect = np.argsort(vr_score)
            score_sel = vr_score
        elif order == 'spkchanvar_score':
            sort_vect = np.argsort(spkchanvar_score)
            score_sel = spkchanvar_score
        elif order == 'n_spikes_mean':
            sort_vect = np.argsort(n_spikes_mean)
            score_sel = n_spikes_mean
        elif order == 'channel_with_spikes_ratio':
            sort_vect = np.argsort(channel_with_spikes_ratio)
            score_sel = channel_with_spikes_ratio
        elif order in param_grid.__dict__['param_grid'][0].keys():
            param_ev = [param_grid[i][order] for i in range(0, len(param_grid))]
            sort_vect = np.argsort(param_ev)
            score_sel = param_ev
        else:
            raise ValueError('Wrong order argument')
        # Plot
        n_subplots = 3
        for i_key, param_key in enumerate(['fmin', 'fmax', 'forder', 'fbank_type']):
            param_ev = np.array([param_grid[i][param_key] for i in range(0, len(param_grid))])
            if len(np.unique(param_ev)) > 1:
                n_subplots = 4
        f = plt.figure()
        ax0 = f.add_subplot(n_subplots, 1, 1)
        l0 = ax0.plot(vr_score[sort_vect], lw=1, marker='.')
        l02 = ax0.plot(channel_with_spikes_ratio[sort_vect], lw=1, marker='.')
        ax1 = ax0.twinx()
        ax1.grid(False)
        l1 = ax1.plot(spkchanvar_score[sort_vect], 'r', lw=1, marker='.')
        ax1.legend(l0 + l02 + l1, ['Van Rossum Distance norm.', 'Channel with spikes', 'CV(spike_per_channel)'])
        ax0.set(ylabel='Van Rossum Distance norm.')
        ax1.set(ylabel='CV(spike_per_channel)')
        ax2 = f.add_subplot(n_subplots, 1, 2, sharex=ax0)
        l1 = ax2.plot(global_score[sort_vect], marker='.')
        ax3 = ax2.twinx()
        l2 = ax3.plot(n_spikes_mean[sort_vect], 'y', marker='.')
        ax3.autoscale(axis='x', tight=True)
        ax3.plot([0, len(global_score)], [0, 0], 'k', alpha=0.2, zorder=0)
        ax3.grid(False)
        ax2.set(ylabel='Global Score')
        ax3.set(ylabel='Mean number of spikes')
        ax3.legend(l1+l2, ['Global Score', 'Mean number of spikes'])
        ax4 = f.add_subplot(n_subplots, 1, 3, sharex=ax0)
        legend_str, title_str = [], 'Cochlea parameters : '
        # Plot Compression and LIF parameters
        for param_key in params_to_plot:
            param_ev = [param_grid[i][param_key] for i in range(0, len(param_grid))]
            if len(np.unique(param_ev)) == 1:
                title_str += '{} = {} - '.format(param_key, param_ev[0])
            else:
                if param_key in ['fmin', 'fmax', 'forder', 'fbank_type']:
                    continue
                ax4.plot(np.array(param_ev)[sort_vect], marker='.')
                legend_str += [param_key]
        ax4.autoscale(axis='x', tight=True)
        ax4.legend(legend_str, loc='upper right')
        if n_subplots == 4:
            ax5 = f.add_subplot(n_subplots, 1, 4, sharex=ax0)
            ax6 = ax5.twinx()
            legend_str_5, legend_str_6, i_tick, ticks, tick_names = [], [], 0, [], []
            colors = sns.color_palette()
            for i_key, param_key in enumerate(['fmin', 'fmax', 'forder', 'fbank_type']):
                param_ev = np.array([param_grid[i][param_key] for i in range(0, len(param_grid))])
                if len(np.unique(param_ev)) > 1:
                    if param_key in ['fbank_type', 'freq_scale']:
                        param_num_ev = np.zeros(param_ev.shape[0])
                        for name in np.unique(param_ev):
                            param_num_ev[param_ev == name] = i_tick
                            ticks.append(i_tick)
                            tick_names.append(name)
                            i_tick += 1
                        ax6.plot(param_num_ev[sort_vect], c=colors[i_key], marker='.')
                        legend_str_6 += [param_key]
                    else:
                        ax5.plot(param_ev[sort_vect], c=colors[i_key], marker='.')
                        legend_str_5 += [param_key]
            ax5.legend(legend_str_5, loc='upper left')
            ax6.legend(legend_str_6, loc='upper right')
            ax6.yaxis.set_ticks(ticks)
            ax6.yaxis.set_ticklabels(tick_names)
            ax6.grid(False)
        order_str = 'None' if not order else order
        title_str += 'Order : {}'.format(order_str)
        ax0.set(title=title_str)
        ax0.autoscale(axis='x', tight=True)
        f.show()

        # Print best configurations
        for i in range(0, min(20, len(score_sel))):
            sort_vect_score = np.argsort(score_sel)
            if np.isnan(score_sel[sort_vect_score[i]]):
                continue
            line_str = 'N {} - Global Score {:.2f} - '.format(sort_vect_score[i], global_score[sort_vect_score[i]])
            line_str += 'VR score {:.2f} - '.format(vr_score[sort_vect_score[i]])
            line_str += 'CV score {:.2f} - '.format(spkchanvar_score[sort_vect_score[i]])
            for param_key in params_to_plot:
                param_val = param_grid[sort_vect_score[i]][param_key]
                if type(param_val) == str or type(param_val) == np.str_:
                    line_str += '{} = {}, '.format(param_key, param_val)
                else:
                    line_str += '{} = {:.2f}, '.format(param_key, param_val)
            line_str += ' Mean spike = {:.0f}'.format(n_spikes_mean[sort_vect_score[i]])
            print(line_str)

    def print_cochlea_parameters(self, param_grid, grid_pos):
        for key in param_grid.__dict__['param_grid'][0].keys():
            print('{} value : {}'.format(key, param_grid[grid_pos][key]))

