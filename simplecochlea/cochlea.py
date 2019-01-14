"""
This file is part of simplecochlea.

simplecochlea is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

simplecochlea is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with simplecochlea.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from os import path, mkdir
import pandas as pd
from scipy import signal
import pickle
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

# matplotlib.use('TkAgg')
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

    def plot_caracteristics(self, n_freqs=1024):
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
    Parralel association of rectifier units.
    Half and full rectification are possible :
        * half-rectification : :math:`f(x) = x if x > 0, 0 otherwise`
        * full-rectification : :math:`f(x) = abs(x)`
    Rectification can include a low-pass filtering using a Butterworth filters. Each channel can have its own filtering
    cutoff frequency.
    The cochlea uses a rectifier bank just after the band-pass filtering step.

    Attributes
    ----------
    fs : float
        Sampling frequency (Hz). Each RectifierBank instance must be applied to signals with the same sampling
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
        Each channel of the input signal is rectified by its correponding rectifier.

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

    # def plot_caracteristics(self, f_sin=1000):
    #     """ Plot the RectifierBank response to a sinusoidal signal
    #
    #     Parameters
    #     ----------
    #     f_sin : float
    #         Sinusoid frequency (Hz). Default 1000 Hz.
    #     """
    #     # Create the test signal and process it
    #     x_sin = generate_sinus(self.fs, f_sin=f_sin)
    #     # x_sin_rect =
    #     f = plt.figure()
    #     ax = f.add_subplot()




class CompressionBank:
    """ CompressionBank Class
    Parralel association of compression units.
    Apply an amplitude compression using a logarithmic transform :
     :math:`f(x) = comp_{gain} * x^{comp_{factor}}`
    Each channel can have a separate compression gain and compression factor.
    In the cochlea, the compression bank follows the rectifier bank and precede the LIF bank.

    Attributes
    ----------
    fs : float
        Sampling frequency (Hz). Each BandPassFilterbank instance must be applied to signals with the same sampling
        frequency.
    n_channels : int
        Number of channels
    comp_factor : float | array
        Compression factor - :math:`f(x) = comp_{gain} * x^{comp_{factor}}`
    comp_gain : float | array
        Compression gain - :math:`f(x) = comp_{gain} * x^{comp_{factor}}`

    """
    def __init__(self, n_channels, fs, comp_factor, comp_gain):
        self.n_channels = n_channels
        if not np.isscalar(comp_factor) and not len(comp_factor) == n_channels:
            raise ValueError('Argument compression_factor must be either a scalar or a vector of length n_channels')
        if not np.isscalar(comp_gain) and not len(comp_gain) == n_channels:
            raise ValueError('Argument compression_gain must be either a scalar or a vector of length n_channels')
        if np.isscalar(comp_factor):
            comp_factor = np.repeat(comp_factor, self.n_channels)
        if np.isscalar(comp_gain):
            comp_gain = np.repeat(comp_gain, self.n_channels)
        self.comp_factor = np.array(comp_factor)
        self.comp_gain = np.array(comp_gain)
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
        """ Apply the compression to input signal `input_sig` with channel specified by `channel_pos`

        Parameters
        ----------
        input_sig : array (1D)
            Input signal
        channel_pos : int
            Channel position

        Returns
        -------
        output_sig : array (1D)
            Output compressed signal

        """
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
        """ Compress input signal `input_sig`.
        Each channel of the input signal is compressed by its correponding compression unit.

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
    """ Leaky Integrate and Fire (LIF) Bank
    Parralel association of LIF neuron model.
    The LIF neuron is a simple spiking neuron model where the neuron is modeled as a leaky integrator of the input
    synaptic current I(t).

    A refractiory period can be defined preventing the neuron to fire just after emitting a spike.

    By default each LIF unit (i.e. channel) is independant from the others. To model more complex comportement,
    lateral inhibition can be added between units (channels).
    This lateral inhibition is inspired from [1].

    Adaptive threshold is another option. Instead of using a fixed spiking threshold defined bt `v_thresh`, it is
    possible to model an adaptive threshold using the `tau_j`, `alpha_j` and `omega` attributes.
    These variables implements the adaptive threshold model described in [2].


    Attributes
    -----------
    fs : float
        Sampling frequency (Hz). Each LIFBank instance must be applied to signals with the same sampling frequency.
    n_channels : int
        Number of channels
    tau : float | array
        Time constant (s)
    v_thresh : float | array
        Spiking threshold - Default : 0.020
    v_spike : float | array
        Spike potential - Default : 0.035
    t_refract : float | array | None
        Refractory period (s). If none, no refractory period - Default : None
    v_reset : float | array | None
        Reset potential after a spike. Can act as a adaptable refractory period.
    inhib_vect : array | None - Default : None
        If defined, lateral inhibition is activated. If none, no lateral inhibition
    tau_j : array (1D) | None
        Time constants for the adaptive threshold. Usually no more than 3 time constants are needed. See [2].
    alpha_j : array (1D) | None
        Coefficients for each time constant defined by `tau_j`.
    omega : float
        Threshold resting value (minimum threshold value).


    References
    ----------
    .. [1] Gershon G. Furman and Lawrence S. Frishkopf. Model of Neural Inhibition in the Mammalian Cochlea.
           The Journal of the Acoustical Society of America 1964 36:11, 2194-2201

    .. [2] Kobayashi Ryota, Tsubo Yasuhiro, Shinomoto Shigeru. Made-to-order spiking neuron model equipped with a
           multi-timescale adaptive threshold. Frontiers in Computational Neuroscience. 2009

    """
    def __init__(self, n_channels, fs, tau, v_thresh=0.020, v_spike=0.035, t_refract=[], v_reset=[],
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
        self.tau = np.float64(tau)
        self.v_thresh = np.float64(v_thresh)
        self.v_spike = np.float64(v_spike)
        self.fs = np.float64(fs)
        self.inhib_vect = np.float64(inhib_vect)
        if self.inhib_vect.size > 0:
            self.inhib_on = 1
        else:
            self.inhib_on = 0
        self.tau_j, self.alpha_j, self.omega = np.float64(tau_j), np.float64(alpha_j), np.float64(omega)
        if self.tau_j.size > 0:
            self.adaptive_threshold = 1
        else:
            self.adaptive_threshold = 0
        if self.omega.size == 1:
            self.omega = np.float64(np.repeat(self.omega, self.n_channels))
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
        if self.inhib_on:
            desc_str += 'Lateral inhibition activated. Length of inhibition vector : {}'.format(self.inhib_vect.size)
        else:
            desc_str += 'No inhibition'
        return desc_str

    def filter_signal(self, input_sig, v_init=[], t_start=0, t_last_spike=[]):
        """

        Parameters
        ----------
        input_sig : array [1D]
            Input signal
        v_init : array
            Initial LIF potential
        t_start : float (default: 0)
            ???
        t_last_spike : float | none (default: none)
            Time of the last spike

        Returns
        -------
        output_sig : array [n_chan, n_pnts]
            LIF output potential
        spikes : array [n_spikes, 2]
            First column contain the spikes time; the 2nd column contains the spike channel

        """
        input_sig = np.array(input_sig).squeeze()
        if len(input_sig.shape) == 1:
            input_sig = np.tile(input_sig, (self.n_channels, 1))
        if not input_sig.shape[0] == self.n_channels:
            raise ValueError('input_sig must be a [n_channels, n_pnts] matrice')
        if not np.array(v_init).any():
            v_init = np.zeros(self.n_channels, dtype=np.float64)
        elif not np.array(v_init).shape[0] == self.n_channels:
            raise ValueError('v_init must be a [n_channels, 1] array')
        if not np.array(t_last_spike).any():
            t_last_spike = np.float64(-2 * self.t_refract * np.ones(self.n_channels))
        elif not t_last_spike.shape[0] == self.n_channels:
            raise ValueError('t_last_spike must be a [n_channels, 1] array')
        # Call cython function given the inhibition type :
        if not self.inhib_on:
            output_sig, t_spikes, chan_spikes = lif_filter_cy \
                (self.fs, input_sig, self.refract_period, self.t_refract, self.tau, self.v_thresh, self.v_spike,
                 self.v_reset, v_init, self.adaptive_threshold, self.tau_j, self.alpha_j,  self.omega,
                 t_start, t_last_spike)
        else:
            output_sig, t_spikes, chan_spikes = lif_filter_inhib_shuntfor_current_cy \
                (self.fs, input_sig, self.refract_period, self.t_refract, self.tau, self.v_thresh, self.v_spike,
                 self.v_reset, v_init, self.inhib_vect, self.adaptive_threshold, self.tau_j, self.alpha_j,  self.omega,
                 t_start, t_last_spike)
        spikes = np.vstack([t_spikes, chan_spikes]).T
        return output_sig, spikes, []

    def filter_one_channel(self, input_sig, i, v_init=[], t_start=0, t_last_spike=[]):
        """ Filter one channel

        Parameters
        ----------
        input_sig : 1D array
            Input vector to be passed through the LIF
        i : int
            Channel position
        v_init : array
            Initial LIF potential
        t_start : float (default: 0)
            ???
        t_last_spike : float | none (default: none)
            Time of the last spike

        Returns
        -------
        v_out : array [n_pnts]
            LIF output potential
        t_spikes : array [n_spikes]
            Spikes time
        threshold : array [n_pnts]
            Threshold at each time point

        """
        if len(np.array(input_sig).shape) > 1 and not np.min(np.array(input_sig).shape) == 1:
            raise ValueError('input_sig must be a 1D array')
        if not np.isscalar(i) or i < 0 or i > self.n_channels-1:
            raise ValueError('argument channel_pos must be a scalar ranging between 0 and the number of channels-1')
        if not np.array(v_init).any():
            v_init = np.zeros(self.n_channels, dtype=np.float64)
        elif not np.array(v_init).shape[0] == self.n_channels:
            raise ValueError('v_init must be a [n_channels, 1] array')
        if not np.array(t_last_spike).any():
            t_last_spike = -2 * self.t_refract * np.ones(self.n_channels)
        elif not t_last_spike.shape[0] == self.n_channels:
            raise ValueError('t_last_spike must be a [n_channels, 1] array')
        if not self.inhib_on:
            v_out, t_spikes, threshold = lif_filter_1d_signal_cy(int(self.fs), np.float64(input_sig), int(self.refract_period), np.float64(self.t_refract[i]),
                                                      self.tau[i], self.v_thresh[i], self.v_spike[i], self.v_reset[i],
                                                      v_init[i], self.adaptive_threshold, self.tau_j, self.alpha_j,
                                                      self.omega[i], t_start=np.float64(t_start), t_last_spike_p=np.float64(t_last_spike[i]))
        return v_out, t_spikes, threshold

    def plot_caracteristics(self):
        """ Plot the LIFBank characteristics """
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


class Cochlea:
    __slots__ = ['n_channels', 'freq_scale', 'fs', 'fmin', 'fmax', 'forder', 'cf', 'filterbank', 'rectifierbank',
                 'compressionbank', 'lifbank']

    def __init__(self, n_channels, fs, fmin, fmax, freq_scale, order=2, fbank_type='butter', rect_type='full',
                 rect_lowpass_freq=[], comp_factor=1.0/3, comp_gain=1, lif_tau=0.010, lif_v_thresh=0.020,
                 lif_v_spike=0.035, lif_t_refract=0.001, lif_v_reset=[], inhib_vect=[],
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
                               inhib_vect, alpha_j=alpha_j, tau_j=tau_j, omega=omega)

    def __str__(self):
        desc_str = 'Cochlea model - {} channels [{} - {} Hz] - {} - {} order Butterworth filters\n'.format(
            self.n_channels, self.fmin, self.fmax, self.freq_scale, self.forder)
        desc_str += self.rectifierbank.__str__()
        desc_str += self.compressionbank.__str__()
        desc_str += self.lifbank.__str__()
        return desc_str

    def save(self, dirpath, filename=[]):
        """ Save the cochlea.
        The cochlea model is saved as a .p (pickle) file. The filename is appended with the current date and time.

        Parameters
        ----------
        dirpath : str
            Directory path
        filename : str
            Cochlea filename.

        """
        if not path.isdir(dirpath):
            print('Creating save directory : {}'.format(dirpath))
            mkdir(dirpath)
        if not filename:
            filename = 'cochlea_model_{}.p'.format(datetime.strftime(datetime.now(), '%d%m%y_%H%M'))
        with open(path.join(dirpath, filename), 'wb') as f:
            pickle.dump(self, f)

    def plot_channel_evolution(self, input_sig, channel_pos=[]):
        """ Plot the processing of input_sig signal through the cochlea, for channels specified by channel_pos.
        If channel_pos is not provided, 4 channels are selected ranging from low-frequency channel to high-frequency
        channel.

        Parameters
        ----------
        input_sig : array [1D]
            Input signal
        channel_pos : scalar | array | list | None
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
            y_filt, y_rect, y_comp, v_out, t_spikes, threshold = self.process_one_channel(input_sig, channel_pos)
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
            axx.set_ylabel('Mean Spikerate (Hz)')
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

    def process_one_channel(self, input_sig, channel_pos):
        """ Process a signal with 1 channel of the cochlea

        Parameters
        ----------
        input_sig : array [1D]
            Input signal
        channel_pos : scalar | array | list | None
            Position of the channel(s) to plot. If multiples channels are selected, plot one figure per channel.
            If none,  4 channels are selected ranging from low-frequency channel to high-frequency channel.

        Returns
        -------
        sig_filtered : array [n_chan, n_pnts]
            Signals after the band pass filterbank
        sig_rectiry : array [n_chan, n_pnts]
            Signals after the rectifier bank
        sig_compressed : array [n_chan, n_pnts]
            Signals after the compression bank
        lif_out_sig : array [n_chan, n_pnts]
            Signals after the LIF bank (Membrane potential)
        t_spikes : array
            Spikes time (s)

        """
        input_sig = np.array(input_sig)
        if not input_sig.ndim == 1:
            raise ValueError('Argument input_sig should be 1D')
        if input_sig.max() > 10 or input_sig.min() < -10:
            print('Be sure to normalize the signal before applying the cochlea')
        sig_filtered, _ = self.filterbank.filter_one_channel(input_sig, channel_pos)
        sig_rectify = self.rectifierbank.rectify_one_channel(sig_filtered, channel_pos)
        sig_compressed = self.compressionbank.compress_one_channel(sig_rectify, channel_pos)
        lif_out_sig, t_spikes, threshold = self.lifbank.filter_one_channel(sig_compressed, channel_pos)
        return sig_filtered, sig_rectify, sig_compressed, lif_out_sig, t_spikes, threshold

    @timethis
    def process_input(self, input_sig, do_plot=False):
        """ Process input signal through the Cochlea

        Parameters
        ----------
        input_sig : array (1D)
            Input signal
        do_plot : bool
            If True, plot the differents steps of the cochlea

        Returns
        -------
        spikelist : SpikeList
            Output spikelist
        (sig_filtered, sig_rectified, sig_comp, lif_out_sig) : tuple
            Contains the intermediary signals correponding to the different steps of the cochlea

        """
        input_sig = np.array(input_sig)
        if not input_sig.ndim == 1:
            raise ValueError('Argument input_sig should be 1D')
        sig_filtered, _ = self.filterbank.filter_signal(input_sig, do_plot=do_plot)
        sig_rectified = self.rectifierbank.rectify_signal(sig_filtered, do_plot=do_plot)
        sig_comp = self.compressionbank.compress_signal(sig_rectified, do_plot=do_plot)
        lif_out_sig, spikes, _ = self.lifbank.filter_signal(sig_comp)
        tmin, tmax = 0, input_sig.size / self.fs
        spikelist = SpikeList(time=spikes[:, 0], channel=spikes[:, 1], n_channels=self.n_channels, name=self.__str__(),
                              tmin=tmin, tmax=tmax)
        return spikelist, (sig_filtered, sig_rectified, sig_comp, lif_out_sig)

    def process_test_signal(self, signal_type, channel_pos=[], do_plot=True, **kwargs):
        """ Run a test signal through the cochlea. Test signals can be a sinusoidal signal, a step or an impulse signal.
        Argument signal_type selects the wanted signal.
        If channel_pos is provided and do_plot is True, channel evolution is plot.

        Parameters
        ----------
        signal_type : str
            Test signal type. To choose between :
             * 'sin' or 'sinus' : Sinusoidal signal
             * 'dirac' or 'impulse': Impulse signal
             * 'step' : Step signal
        channel_pos : int
            Channel position
        do_plot : bool
            If True, plot the results
        kwargs :
            Test signal parameters : 't_offset', 't_max', 'amplitude', 'f_sin' (for sinusoids only)

        Returns
        -------
        spikelist : SpikeList
            Output spikelist

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
            spikelist = self.process_input(x_test, do_plot=do_plot)
        else:
            spikelist = self.process_input(x_test, do_plot=False)
            if do_plot:
                self.plot_channel_evolution(x_test, channel_pos)
        return spikelist

    # def process_input_block_ver(self, input_sig, block_len, plot_spikes=1):
    #     n_blocks = int(np.ceil(len(input_sig) / block_len))
    #     block_duration = block_len / self.fs
    #     zi, t_last_spikes = [], []
    #     v_init = np.zeros(self.n_channels)
    #     spikes = []
    #     for i in range(0, n_blocks):
    #         block_i = input_sig[i*block_len: np.min([len(input_sig), (i+1)*block_len])]
    #         block_i_filtered, zi = self.filterbank.filter_signal(block_i, do_plot=0, zi_in=zi)
    #         block_i_rectified = self.rectifierbank.rectify_signal(block_i_filtered, do_plot=0)
    #         block_i_compressed = self.compressionbank.compress_signal(block_i_rectified, do_plot=0)
    #         v_out_i, spikes_i, t_last_spikes = self.lifbank.filter_signal(block_i_compressed, do_plot=0, v_init=v_init,
    #                                                                       t_start=i*block_duration,
    #                                                                       t_last_spike=t_last_spikes)
    #         v_init = v_out_i[:, -1]
    #         spikes.append(spikes_i)
    #     spikes = np.vstack(np.array(spikes))
    #     spike_list = SpikeList(time=spikes[:, 0], channel=spikes[:, 1], n_channels=self.n_channels, name=self.__str__())
    #     if plot_spikes:
    #         spike_list.plot()
    #     return spike_list, _


def load_cochlea(dirpath, filename):
    """ Load the cochlea in pickle format (.p) defined by `filename` in `dirpath` """
    with open(path.join(dirpath, filename), 'rb') as f:
        return pickle.load(f)


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

