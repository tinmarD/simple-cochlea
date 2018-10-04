import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io as sio
import os
import pymuvr
sns.set()
sns.set_context('paper')


class SpikeList:
    def __init__(self, time=[], channel=[], pattern_id=[], potential=[], n_channels=0, name='', tmin=0,
                 tmax=[], pattern_names={}):
        time, channel, pattern_id, potential = np.array(time), np.array(channel), np.array(pattern_id, dtype=int),\
                                               np.array(potential)
        if time.size == 0 and not channel.size == 0 or channel.size == 0 and not time.size == 0:
            raise ValueError('SpikeList time must be associated with a channel')
        if time.size > 0 and channel.size > 0:
            time = np.array([time]) if time.ndim == 0 else np.array(time)
            channel = np.array([channel]) if channel.ndim == 0 else np.array(channel)
            if n_channels == 0:
                n_channels = channel.max()
            if len(time) != len(channel):
                raise ValueError('Spikelist arguments time and channel must have the same length')
            if pattern_id.size > 0 and pattern_id.ndim == 0:
                pattern_id = pattern_id * np.ones(len(time), dtype=int)
            if pattern_id.size > 0 and len(pattern_id) != len(time):
                raise ValueError('Spikelist argument pattern_id must either be empty, a scalar or a vector with '
                                 'the same length that time and channel arguments')
            if potential.size > 0 and potential.ndim == 0:
                potential = potential * np.ones(len(time))
            if potential.size > 0 and len(potential) != len(time):
                raise ValueError('Spikelist argument potential must either be empty, a scalar or a vector with '
                                 'the same length that time and channel arguments')
        if not isinstance(name, str):
            raise ValueError('Spikelist argument name must be a string')
        if tmin and not np.isscalar(tmin):
            raise ValueError('SpikeList argument tmin must be a scalar')
        if tmax and not np.isscalar(tmax):
            raise ValueError('SpikeList argument tmax must be a scalar')
        if time.size == 0:
            tmax = 0
            n_spikes = 0
        else:
            if not tmax:
                tmax = time.max()
            n_spikes = len(time)
        if pattern_id.size == 0:
            pattern_id = np.nan * np.ones(n_spikes, dtype=int)
        if not potential.any():
            potential = np.zeros(n_spikes)
        self.time = time
        self.channel = channel.astype(int)
        self.pattern_id = pattern_id
        self.potential = potential
        self.n_spikes = len(time)
        self.n_channels = n_channels
        self.name = name
        self.tmin = tmin if tmin else 0
        self.tmax = tmax
        if pattern_names:
            self.pattern_names = pattern_names
        else:
            patterns = np.unique(self.pattern_id[~np.isnan(self.pattern_id)])
            self.pattern_names = dict(zip(patterns, ['pattern {}'.format(i+1) for i in range(0, len(patterns))]))

    def __str__(self):
        description = 'SpikeList class with {} spikes, {} channels'.format(self.n_spikes, self.n_channels)
        if self.name:
            description += '\n - name : {}'.format(self.name)
        patterns = np.unique(self.pattern_id[~np.isnan(self.pattern_id)])
        n_patterns = len(patterns)
        if n_patterns > 0:
            description += '\n - {} patterns'.format(n_patterns)
        if self.pattern_names:
            description += ' : {}'.format(self.pattern_names)
        return description

    def __len__(self):
        return self.n_spikes

    def __getitem__(self, item):
        pattern_ids = np.unique(self.pattern_id[item])
        pattern_ids = pattern_ids[~np.isnan(pattern_ids)]
        pattern_names = [self.pattern_names[pat_id] for pat_id in pattern_ids]
        return SpikeList(self.time[item], self.channel[item], self.pattern_id[item], self.potential[item],
                         self.n_channels, self.name, self.tmin, self.tmax, dict(zip(pattern_ids, pattern_names)))

    def sort(self, field):
        if not isinstance(field, str) or field.lower() not in ['time', 'channel', 'pattern_id', 'potential']:
            raise ValueError('Argument field must be a string, either time, channel, pattern_id, or potential')
        if field is 'time':
            sort_vect = self.time.argsort()
        elif field is 'channel':
            sort_vect = self.channel.argsort()
        elif field is 'pattern_id':
            sort_vect = self.pattern_id.argsort()
        elif field is 'potential':
            sort_vect = self.potential.argsort()
        return self[sort_vect]

    def add_time_offset(self, t_offset):
        if not np.isscalar(t_offset):
            raise ValueError('Argument t_offset should be a scalar')
        spklist_offset = self
        spklist_offset.tmin += t_offset
        spklist_offset.tmax += t_offset
        spklist_offset.time += t_offset
        return spklist_offset

    def select_spike(self, time_min=[], time_max=[], channel_sel=[], potential_min=[], potential_max=[],
                     pattern_id_sel=[]):
        time_min, time_max = np.array(time_min), np.array(time_max)
        potential_min, potential_max = np.array([potential_min]), np.array([potential_max])
        channel_sel, pattern_id_sel = np.array(channel_sel), np.array(pattern_id_sel)
        if time_min.size > 1 or time_max.size > 1 or channel_sel.size > 1 or pattern_id_sel.size > 1:
            raise ValueError('Arguments should be scalar or single value arrays')
        sel_ind = np.ones(self.n_spikes, dtype=bool)
        if time_min.size > 0:
            sel_ind = self.time >= time_min
        if time_max.size > 0:
            sel_ind = sel_ind & (self.time < time_max)
        if channel_sel.size > 0:
            sel_ind = sel_ind & np.in1d(self.channel, channel_sel)
        if potential_min.size > 0:
            sel_ind = sel_ind & (self.potential >= potential_min)
        if potential_max.size > 0:
            sel_ind = sel_ind & (self.potential < potential_max)
        if pattern_id_sel.size > 0 and self.pattern_id.size > 0:
            sel_ind = sel_ind & np.in1d(self.pattern_id, pattern_id_sel)
        sel_ind_pos = np.where(sel_ind)[0]
        if sel_ind_pos.size == 0:
            spikelist_sel = SpikeList(n_channels=self.n_channels, tmin=self.tmin, tmax=self.tmax)
        else:
            spikelist_sel = self[sel_ind_pos]
        n_spikes = np.sum(sel_ind)
        return sel_ind, spikelist_sel, n_spikes

    def set_pattern_id_from_time_limits(self, t_start, t_end, pattern_id, pattern_dict):
        t_start, t_end, pattern_id = np.array(t_start), np.array(t_end), np.array(pattern_id)
        t_start = np.array([t_start]) if t_start.ndim == 0 else t_start
        t_end = np.array([t_end]) if t_end.ndim == 0 else t_end
        pattern_id = np.array([pattern_id]) if pattern_id.ndim == 0 else pattern_id
        if not t_start.size == t_end.size == pattern_id.size:
            raise ValueError('Arguments t_start, t_end and pattern_id must have the same size')
        unique_pattern_id, index = np.unique(pattern_id, return_index=True)
        for pat_id_i in unique_pattern_id:
            self.pattern_names[pat_id_i] = pattern_dict[pat_id_i]
        n_periods = len(t_start)
        for i in range(0, n_periods):
            sel_ind_i, _, _ = self.select_spike(time_min=t_start[i], time_max=t_end[i]+0.001)
            self.pattern_id[sel_ind_i] = pattern_id[i]

    def set_pattern_id_from_time_limits_old(self, t_start, t_end, pattern_id, pattern_names=[]):
        t_start, t_end, pattern_id = np.array(t_start), np.array(t_end), np.array(pattern_id)
        t_start = np.array([t_start]) if t_start.ndim == 0 else t_start
        t_end = np.array([t_end]) if t_end.ndim == 0 else t_end
        pattern_id = np.array([pattern_id]) if pattern_id.ndim == 0 else pattern_id
        if not len(t_start) == len(t_end) == len(pattern_id):
            raise ValueError('Arguments t_start, t_end and pattern_id must have the same length')
        unique_pattern_id, index = np.unique(pattern_id, return_index=True)
        for i, pat_id_i in enumerate(unique_pattern_id):
            self.pattern_names[pat_id_i] = pattern_names[index[i]]
        n_periods = len(t_start)
        for i in range(0, n_periods):
            sel_ind_i, _, _ = self.select_spike(time_min=t_start[i], time_max=t_end[i]+0.001)
            self.pattern_id[sel_ind_i] = pattern_id[i]

    def get_pattern_results(self, t_start, t_end, pattern_id, pattern_names=[], min_potential=[], do_plot=True,
                            fig_title=''):
        """ Compute the number of spikes per segment. Segments are defined by ``t_start`` and ``t_end``. There might be
        multiples patterns, defined by ``pattern_id`` and ``pattern_names``.
        If ``min_potential`` is defined, select only the spikes whose potential is higher than this.
        If ``do_plot`` is true, plot the results.

        Parameters
        ----------
        t_start : array
            Starting times of the segments (s)
        t_end : array
            Ending times of the segments (s)
        pattern_id : array
            Pattern id of the segments
        pattern_names : array | None
            Pattern name of the segments
        min_potential : float | None
            If defined select only spikes whose potential is higher than this
        do_plot : bool | None (default: True)
            If true, plot the results
        fig_title : str | None
            Figure's title

        Returns
        -------
        n_spikes_mean : array (size n_patterns)
            Mean number of spikes across repetition for each pattern
        n_spikes_per_chunk : array  (size n_segments)
            Number of spikes for each segment
        n_active_chan_mean : array (size n_pattern)
            Mean number of active channel across repetition for each pattern
        n_active_chan_per_chunk : array (size n_segments)
            Number of active channel for each segment
        """
        t_start, t_end, pattern_id = np.array(t_start), np.array(t_end), np.array(pattern_id)
        t_start = np.array([t_start]) if t_start.ndim == 0 else t_start
        t_end = np.array([t_end]) if t_end.ndim == 0 else t_end
        pattern_id = np.array([pattern_id]) if pattern_id.ndim == 0 else pattern_id
        if not len(t_start) == len(t_end) == len(pattern_id):
            raise ValueError('Arguments t_start, t_end and pattern_id must have the same length')
        unique_pattern_id, index = np.unique(pattern_id, return_index=True)
        n_patterns = unique_pattern_id.size
        n_chunks = pattern_id.size
        n_spikes_per_chunk = np.zeros(n_chunks, dtype=int)
        n_active_chan_per_chunk = np.zeros(n_chunks, dtype=int)
        n_spikes_mean, n_active_chan_mean = np.zeros(n_patterns), np.zeros(n_patterns)
        n_repets_i = np.zeros(n_patterns, dtype=int)
        n_repets = np.zeros(n_chunks, dtype=int)
        for i in range(n_chunks):
            _, spikelist_i, n_spikes_per_chunk[i] = self.select_spike(t_start[i], t_end[i], potential_min=min_potential)
            n_active_chan_per_chunk[i] = np.unique(spikelist_i.channel).size
            pat_pos = np.where(unique_pattern_id == pattern_id[i])[0]
            n_repets_i[pat_pos] += 1
            n_repets[i] = n_repets_i[pat_pos]
        for i, id in enumerate(unique_pattern_id):
            n_spikes_mean[i] = np.mean(n_spikes_per_chunk[pattern_id == id])
            n_active_chan_mean[i] = np.mean(n_active_chan_per_chunk[pattern_id == id])
        if do_plot:
            d = {'index': np.arange(n_chunks), 'n_spikes': n_spikes_per_chunk,
                 'n_active_channels': n_active_chan_per_chunk, 'pattern_name': pattern_names, 'n_repet': n_repets}
            df = pd.DataFrame(data=d)
            g = sns.factorplot(x='n_repet', y='n_spikes', hue='pattern_name', data=df, kind='bar', legend_out=True)
            if min_potential:
                fig_title += ' - min_potential = {}'.format(min_potential)
            g.set(title=fig_title)
        return n_spikes_mean, n_spikes_per_chunk, n_active_chan_mean, n_active_chan_per_chunk

    def epoch(self, t_start, t_end):
        t_start, t_end = np.array(t_start), np.array(t_end)
        t_start = np.array([t_start]) if t_start.ndim == 0 else t_start
        t_end = np.array([t_end]) if t_end.ndim == 0 else t_end
        if len(t_start) != len(t_end):
            raise ValueError('Arguments t_start, t_end must have the same length')
        n_epochs = len(t_start)
        spike_list_epochs = []
        for i in range(0, n_epochs):
            _, epoch_i, _ = self.select_spike(time_min=t_start[i], time_max=t_end[i])
            spike_list_epochs.append(epoch_i)
        return SpikeList_list(spike_list_epochs, t_start, t_end)

    def epoch_on_triggers(self, t_triggers, time_pre, time_post):
        return self.epoch(t_start=t_triggers - time_pre, t_end=t_triggers+time_post)

    def plot(self, bin_duration=0.002, potential_thresh=[], ax_list=[], minplot=0, tau_lif=[], pattern_id_sel=[],
             color=[]):
        """ Plot the spikelist. The main central axis plots the spike list, one dot for each spike. Each pattern has
        a different color. The bottom axis plots the histogram of the spikes for all channels. The right axis sums up
        the spikes over time for each channel. The left axis plot the median ISI (Inter-Spike Interval), and if the
        ``tau_lif`` parameter is given, the ratio  ISI_med / Tau_Lif is also plotted.

        Parameters
        ----------
        bin_duration : float (default: 0.002s)
            Bin duration in seconds for the time histogram, in the bottom axis.
        potential_thresh : float | none (default)
            Potential threshold, only spike whose potential is higher than this value will be plotted. If none, plot
            all the spikes.
        ax_list : list | none (default)
            Axis handles to plot on. If none, create new axes.
        minplot : bool (default: False)
            If True, plot only the central plot.
        tau_lif : float | array | none (default)
            Tau parameter of the LIF bank of the cochlea. If provided, the ratio ISI_med / Tau_Lif is plotted on the
            left axis.
        pattern_id_sel : int | none (default)
            Pattern to select. If provided, the spikes of the selected pattern will appears on a different color in
            the left and right axes.
        color : list | array | none (default)
            Color of the patterns. If none, use default color

        Returns
        -------
        ax_list : list
            List of the axes [central, bottom, right, left]

        """
        if self.n_spikes == 0:
            print('No Spikes in spikelist {}'.format(self.name))
            return
        if ax_list and np.array(ax_list).size == 4:
            ax0, ax1, ax3, ax4 = ax_list[0], ax_list[1], ax_list[2], ax_list[3]
        if ax_list and np.array(ax_list).size == 1:
            ax0 = ax_list
        if potential_thresh:
            _, spklist_sel, _ = self.select_spike(potential_min=potential_thresh)
            if not spklist_sel:
                print('Empty spike list')
                return
        else:
            spklist_sel = self
        tau_lif = np.array(tau_lif)
        if tau_lif.size > 0 and not tau_lif.size == self.n_channels:
            raise ValueError('Argument tau_lif should have a length of n_channels')
        patterns = np.unique(spklist_sel.pattern_id[~np.isnan(spklist_sel.pattern_id)])
        n_patterns = len(patterns)
        if pattern_id_sel:
            if not np.isscalar(pattern_id_sel):
                print('Parameter ``pattern_id_sel`` must be a scalar')
                pattern_id_sel = []
            elif pattern_id_sel not in np.unique(self.pattern_id):
                print('No pattern with id {}'.format(pattern_id_sel))
                pattern_id_sel = []
            else:
                _, spike_list_pattern, _ = self.select_spike(pattern_id_sel=pattern_id_sel)
                pattern_sel_pos = int(np.where(np.unique(self.pattern_id) == pattern_id_sel)[0])
        if not ax_list:
            fig = plt.figure()
        ax0 = plt.subplot2grid((4, 7), (0, 1), rowspan=3, colspan=5) if not ax_list else ax0
        marker_size = 3 if self.n_spikes > 5000 else 5
        if n_patterns == 0:
            ax0.plot(spklist_sel.time, spklist_sel.channel, '.', markersize=marker_size)
            base_color = sns.color_palette(n_colors=1)[0]
        else:
            base_color = sns.color_palette(n_colors=n_patterns + 1)[0]
            patterns_colors = sns.color_palette(n_colors=n_patterns+1)[1:]
            legend_str = []
            # Plot spikes not belonging to a pattern
            if spklist_sel.time[np.isnan(spklist_sel.pattern_id)].any():
                ax0.plot(spklist_sel.time[np.isnan(spklist_sel.pattern_id)],
                         spklist_sel.channel[np.isnan(spklist_sel.pattern_id)],
                         '.', color='k', markersize=marker_size)
                legend_str.append(' ')
            for i, pattern_id in enumerate(patterns):
                color_i = color if color else patterns_colors[i]
                ax0.plot(spklist_sel.time[spklist_sel.pattern_id == pattern_id],
                         spklist_sel.channel[spklist_sel.pattern_id == pattern_id],
                         '.', color=color_i, markersize=marker_size)
                legend_str.append(spklist_sel.pattern_names[pattern_id])
            ax0.legend(legend_str, frameon=True, framealpha=0.8, loc='upper right')
        ax0.set_ylim(0, self.n_channels)
        ax0.set(title='Raster plot - {}'.format(self.name))
        if minplot:
            return [ax0, [], [], []]
        else:
            ax1 = plt.subplot2grid((4, 7), (3, 1), rowspan=1, colspan=5, sharex=ax0) if not ax_list else ax1
            ax1.hist(spklist_sel.time, bins=int((self.tmax - self.tmin) / bin_duration), color=base_color, rwidth=1)
            ax1.set_xlim(self.tmin, self.tmax)
            ax1.set(xlabel='Time (s)', ylabel='count')
            ax3 = plt.subplot2grid((4, 7), (0, 6), rowspan=3, sharey=ax0) if not ax_list else ax3
            spikes_per_channel = [np.sum(spklist_sel.channel == i) for i in range(0, self.n_channels)]
            ax3.barh(range(0, self.n_channels), spikes_per_channel, height=1, color=base_color, linewidth=0)
            if pattern_id_sel:
                spikes_per_channel_pattern = [np.sum(spike_list_pattern.channel == i) for i in range(0, self.n_channels)]
                ax3.barh(range(0, self.n_channels), spikes_per_channel_pattern, height=1,
                         color=patterns_colors[pattern_sel_pos],  linewidth=0)
            ax3.set_ylim((0, self.n_channels))
            ax3.set(xlabel='count', ylabel='Channel')
            ax3.invert_xaxis()
            ax3.yaxis.set_label_position("right")
            ax3.yaxis.tick_right()
            ax4 = plt.subplot2grid((4, 7), (0, 0), rowspan=3, colspan=1, sharey=ax0) if not ax_list else ax4
            ax4.set(ylabel='Channel', xlabel='Median ISI (ms)')
            median_isi = spklist_sel.get_median_isi()
            ax4.barh(range(0, self.n_channels), 1000 * median_isi, 1, color=base_color, linewidth=0)
            if pattern_id_sel:
                median_isi_pattern = spike_list_pattern.get_median_isi()
                ax4.barh(range(0, self.n_channels), 1000 * median_isi_pattern, 1,
                         color=patterns_colors[pattern_sel_pos], linewidth=0)
            ax4.autoscale(axis='y', tight=True)
            ax4.set_xlim([0, 5])
            ax5 = ax4.twiny()
            ax5.grid(False)
            if tau_lif.size > 0:
                if pattern_id_sel:
                    ax5.plot(median_isi_pattern / tau_lif, np.arange(0, self.n_channels), 'r', alpha=0.5)
                else:
                    ax5.plot(median_isi / tau_lif, np.arange(0, self.n_channels), 'r', alpha=0.5)
                ax5.plot([1, 1], [0, self.n_channels], 'r', ls='--', alpha=0.4)
                ax5.autoscale(axis='y', tight=True)
                ax5.set(xlabel='$ISI_{{med}} / \\tau_{{LIF}}$')
                ax5.set_xlim([0, 10])
            return [ax0, ax1, ax3, ax4]

    def plot_channel_selectivity(self, title_str=''):
        chan_pattern_spikes, chan_pref_pattern, chan_selectivity = self.get_channel_selectivity()
        if chan_pattern_spikes.size == 0:
            return
        n_spikes_on_pref_pattern = np.max(chan_pattern_spikes, 1)
        f = plt.figure()
        ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=4, colspan=1)
        ax1.bar(range(0, self.n_channels), np.sum(chan_pattern_spikes, 1), width=1, color=[0.2, 0.2, 0.2])
        legend_str = ['Total']
        patterns_colors = sns.color_palette(n_colors=len(self.pattern_names))
        for i, pat_id in enumerate(self.pattern_names):
            chan_sel = np.where(chan_pref_pattern == pat_id)[0]
            if chan_sel.size > 0:
                ax1.bar(chan_sel, n_spikes_on_pref_pattern[chan_sel], width=1, color=patterns_colors[i])
                legend_str.append(self.pattern_names[pat_id])
        ax1.legend(legend_str, loc='upper right')
        ax1.set(ylabel='Number of spikes', title='Number of spikes during preferred pattern - {}'.format(title_str))
        ax2 = plt.subplot2grid((5, 1), (4, 0), rowspan=1, colspan=1, sharex=ax1)
        ax2.bar(range(0, self.n_channels), chan_selectivity, width=1)
        ax2.set(xlabel='Channel', title='Pattern Selectivity')
        ax1.autoscale(axis='x', tight=True)

    def get_median_isi(self):
        median_isi = np.zeros(self.n_channels)
        for i in range(0, self.n_channels):
            spike_times_i = self.time[self.channel == i]
            if len(spike_times_i) > 1:
                median_isi[i] = np.median(spike_times_i[1:] - spike_times_i[:-1])
        return median_isi

    def get_mean_isi(self):
        mean_isi = np.zeros(self.n_channels)
        for i in range(0, self.n_channels):
            spike_times_i = self.time[self.channel == i]
            if len(spike_times_i) > 1:
                mean_isi[i] = np.median(spike_times_i[1:] - spike_times_i[:-1])
        return mean_isi

    def export(self, export_path='.', export_name='spike_list_0_', format='mat'):
        spklist_ordered = self.sort('time')
        spklist_ordered.channel = spklist_ordered.channel + 1
        spike_mat = np.vstack([spklist_ordered.time, spklist_ordered.channel, spklist_ordered.pattern_id,
                               spklist_ordered.potential]).T
        sio.savemat(os.path.join(export_path, export_name), mdict={'spike_list': spike_mat})

    def get_channel_selectivity(self):
        if not self.pattern_names:
            print('No patterns defined - cannot compute channel selectivity')
            return np.array([]), np.array([]), np.array([])
        if self.n_spikes == 0:
            print('No spikes - cannot compute channel selectivity')
            return np.array([]), np.array([]), np.array([])
        n_patterns = len(self.pattern_names)
        chan_pattern_spikes = np.zeros((self.n_channels, n_patterns))
        pattern_spike_true = np.zeros((n_patterns, self.n_spikes), dtype=bool)
        for j, pat_id in enumerate(self.pattern_names):
            pattern_spike_true[j, :] = self.pattern_id == pat_id
        for i in range(0, self.n_channels):
            for j in range(0, n_patterns):
                chan_pattern_spikes[i, j] = np.sum((self.channel == i) & pattern_spike_true[j, :])
        chan_pref_pattern = np.argmax(chan_pattern_spikes, 1)
        chan_selectivity = np.zeros(self.n_channels)
        chan_with_spikes = np.sum(chan_pattern_spikes, 1) > 0
        chan_selectivity[chan_with_spikes] = np.max(chan_pattern_spikes, 1)[chan_with_spikes] / \
                                             np.sum(chan_pattern_spikes, 1)[chan_with_spikes]
        return chan_pattern_spikes, chan_pref_pattern, chan_selectivity

    def to_dataframe(self):
        df = pd.DataFrame({'time':self.time, 'channel':self.channel, 'pattern_id':self.pattern_id},
                          columns=['time', 'channel', 'pattern_id'])
        return df

    def convert_to_packet(self, packet_size, overlap=0):
        if not np.isscalar(packet_size):
            raise ValueError('Argument packet size must be a scalar')
        if overlap < 0 or overlap > 1:
            raise ValueError('Argument overlap must be between 0 and 1')
        n_overlap = np.round(packet_size * overlap)
        if n_overlap == packet_size:
            print('Set overlap to packet_size - 1 : {}'.format(packet_size-1))
            n_overlap = packet_size - 1
        if n_overlap == 0:
            n_max_packets = int(np.ceil(self.n_spikes / packet_size))
        else:
            n_max_packets = int(np.ceil(self.n_spikes / (packet_size - n_overlap)))
        if not n_overlap == 0:
            print('Overlap is not handled correctly - set it to 0')
            n_overlap = 0

        spklist_sorted = self.sort('time')
        # Binary matrix size (n_channel, n_packets)
        packet_matrix = np.zeros((self.n_channels, n_max_packets))
        # Matrix (n_channel, n_packets) containing either 0 or the spike's index in the original spikelist
        index_matrix = np.zeros((self.n_channels, n_max_packets))
        # Maximal spike time for each packet
        packet_time = np.zeros(n_max_packets)
        i_start, i_packet = 0, 0
        while i_start < self.n_spikes-1:
            # Get the packet_size first unique spikes in the list (sorted by time)
            k_start = 3
            unique_chan, spk_ind = np.unique(spklist_sorted.channel[i_start:min(i_start + k_start * packet_size,
                                                                                spklist_sorted.n_spikes)],
                                             return_index=1)
            time_sort_vect = np.argsort(spk_ind)
            unique_chan = unique_chan[time_sort_vect]
            spk_ind = spk_ind[time_sort_vect]
            while len(unique_chan) < packet_size:
                k_start += 1
                unique_chan, spk_ind = np.unique(
                    spklist_sorted.channel[i_start:min(i_start + k_start * packet_size, self.n_spikes)],
                    return_index=1)
                if i_start + k_start * packet_size > self.n_spikes:
                    break
            # unique_chan : position of the channels that spiked
            # spk_ind : relative index of the unique spikes in the spike list (with offset i_start) [sorted by time]
            unique_chan = unique_chan.astype(int)
            unique_chan, spk_ind = unique_chan[0:packet_size], spk_ind[0:packet_size]
            packet_matrix[unique_chan, i_packet] = 1
            index_matrix[unique_chan, i_packet] = spk_ind
            packet_time[i_packet] = spklist_sorted.time[i_start + spk_ind[-1]]
            # Increment i_start
            i_start = int(i_start + 1 + spk_ind[-1] - n_overlap)
            # if n_overlap:
            #     i_start = i_start + 1 + spk_ind[packet_size]
            # else:
            #     i_start = i_start + packet_size - n_overlap
            i_packet += 1
            if len(unique_chan) < packet_size:
                break
        packet_matrix = packet_matrix[:, 0:i_packet-1]
        index_matrix = index_matrix[:, 0:i_packet-1]
        packet_time = packet_time[0:i_packet-1]
        return packet_matrix, index_matrix, packet_time


class SpikeList_list:
    def __init__(self, spikelist_list, t_start=[], t_end=[]):
        t_start, t_end = np.array(t_start), np.array(t_end)
        if t_start.ndim == 0:
            t_start = np.array([t_start])
        if t_end.ndim == 0:
            t_end = np.array([t_end])
        if not type(spikelist_list) == list:
            spikelist_list = [spikelist_list]
        if t_start.size > 0 and t_end.size > 0 and not len(spikelist_list) == len(t_start) == len(t_end):
            raise ValueError('Arguments spikelist_list and t_start and t_end must have the same length')
        self.n_spikelist = len(spikelist_list)
        self.spikelist_list = spikelist_list
        self.n_spikes = np.array([s.n_spikes for s in spikelist_list])
        self.duration = np.array([s.tmax - s.tmin for s in spikelist_list])
        name_list = []
        n_channels_list = []
        for i in range(0, self.n_spikelist):
            if t_start.size > 0:
                self.spikelist_list[i] = spikelist_list[i].add_time_offset(-t_start[i])
                self.spikelist_list[i].tmin = 0
                self.spikelist_list[i].tmax = t_end[i] - t_start[i]
            else:
                self.spikelist_list[i] = spikelist_list[i]
            name_list.append(spikelist_list[i].name)
            n_channels_list.append(spikelist_list[i].n_channels)
        self.tmin = 0
        self.tmax = np.max(np.array([s.tmax - s.tmin for s in spikelist_list]))
        if len(np.unique(np.array(name_list))) == 1:
            self.name = name_list[0]
        else:
            self.name = ''
        if not len(np.unique(np.array(n_channels_list))) == 1:
            raise ValueError('All spike lists should have the same number of channels')
        else:
            self.n_channels = n_channels_list[0]

    def __str__(self):
        return 'SpikeList_list containing {} spike lists with {} channels'.format(self.n_spikelist, self.n_channels)

    def __getitem__(self, item):
        if np.isscalar(item):
            return self.spikelist_list[item]
        elif type(item) == slice:
            spikelist_list_sel = self.spikelist_list[item]
        elif type(item) == list:
            item = np.array(item)
        if type(item) == np.ndarray:
            if item.dtype == 'bool':
                item = np.where(item)[0]
            spikelist_list_sel = [self.spikelist_list[it] for it in item]
        return SpikeList_list(spikelist_list_sel)

    def plot(self, n_max_rows=3, n_max_cols=5, plot_vr_dist=0):
        plot_vr_dist = 1 if not plot_vr_dist == 0 else 0
        n_rows = int(np.ceil(self.n_spikelist / (n_max_cols-plot_vr_dist)))
        n_cols = self.n_spikelist + plot_vr_dist if n_rows == 1 else n_max_cols
        fig = plt.figure()
        ax_list, i = list(), 0
        for k, spklist in enumerate(self.spikelist_list):
            if i == n_cols-1 and plot_vr_dist:
                i += 1
            if i == 0:
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                plt.text(0.3, 0.95, self.name, fontsize=12, transform=plt.gcf().transFigure)
            else:
                ax = fig.add_subplot(n_rows, n_cols, i + 1, sharex=ax_list[0], sharey=ax_list[0])
            spklist.plot(ax_list=ax, minplot=1)
            ax_list.append(ax)
            ax.set(xlabel='Time (s)', title='Epoch {}'.format(i))
            ax.legend('Epoch {} - {} spikes'.format(k, spklist.n_spikes))
            if i == 0 or i / n_cols == int(i / n_cols):
                ax.set(ylabel='Channel')
            i += 1
        if plot_vr_dist:
            _, vr_dist, _, global_vr_dist = self.compute_vanrossum_distance()
            ax = fig.add_subplot(n_rows, n_cols, n_cols, sharey=ax_list[0])
            ax.barh(range(0, self.n_channels), vr_dist, height=1)
            ax.autoscale(axis='y', tight=True)
            ax.invert_xaxis()
            ax.set(xlabel='Van Rossum Distance', title='Mean VR distance : {:.3f}'.format(global_vr_dist))

    def plot_superpose(self, plot_fano_factor=1, plot_vr_dist=1):
        f = plt.figure()
        ax = plt.subplot2grid((1, 6), (0, 1), rowspan=1, colspan=4)
        colors = sns.color_palette(n_colors=self.n_spikelist)
        legend_str = ['Epoch {} - {} spikes'.format(i, self.spikelist_list[i].n_spikes) for i in range(0, self.n_spikelist)]
        for i, spklist in enumerate(self.spikelist_list):
            ax.plot(spklist.time, spklist.channel, '.', color=colors[i], alpha=0.7)
        ax.set(xlabel='Time (s)', ylabel='Channel', title=self.name)
        ax.legend(legend_str)
        if plot_fano_factor:
            fano_factor, mean_fano_factor = self.compute_fano_factor()
            ax_ff = plt.subplot2grid((1, 6), (0, 0), rowspan=1, colspan=1, sharey=ax)
            ax_ff.barh(range(0, self.n_channels), fano_factor, height=1)
            ax_ff.autoscale(axis='y', tight=True)
            ax_ff.set(xlabel='Fano Factor', ylabel='Channel',
                      title='Mean Fano Factor : {:.3f}'.format(mean_fano_factor))
        if plot_vr_dist:
            _, vr_dist, _, global_vr_dist = self.compute_vanrossum_distance()
            ax_vr = plt.subplot2grid((1, 6), (0, 5), rowspan=1, colspan=1, sharey=ax)
            ax_vr.barh(range(0, self.n_channels), vr_dist, height=1)
            ax_vr.autoscale(axis='y', tight=True)
            ax_vr.invert_xaxis()
            ax_vr.set(xlabel='Van Rossum Distance', title='Mean VR distance : {:.3f}'.format(global_vr_dist))

    def compute_channel_with_spikes(self):
        """ Get proportion of channel with spikes in all epochs

        Returns
        -------
        channel_with_spikes_ratio : array

        """
        spikes_in_channel = np.zeros((self.n_channels, self.n_spikelist))
        for i, spklist in enumerate(self.spikelist_list):
            spikes_in_channel[np.unique(spklist.channel), i] = 1
        channel_with_spikes_ind = spikes_in_channel.sum(1) == self.n_spikelist
        channel_with_spikes_ratio = np.sum(channel_with_spikes_ind) / self.n_channels
        return channel_with_spikes_ratio

    def compute_spike_per_channel_variation(self):
        """ Compute the coefficient of variation of the number of spikes per channel for each spike_list

        Returns
        -------
        var_coeff, var_coeff_mean
        """
        var_coeff = np.zeros(self.n_spikelist)
        for i, spklist in enumerate(self.spikelist_list):
            spk_per_channel = np.array([np.sum(spklist.channel == i_chan) for i_chan in range(0, self.n_channels)])
            var_coeff[i] = spk_per_channel.std() / spk_per_channel.mean() if spk_per_channel.sum() > 0 else 0
        return var_coeff, var_coeff.mean()

    def compute_vanrossum_distance(self, cos=0.1, tau=0.001):
        """
        Compute the van Rossum distance (see: A Novel Spike Distance  - M. C. W. van Rossum). Return also the distance
        normalized by sqrt((M+N)/2) where M and N are the number of spikes in the 2 spike trains. This way the
        normalized distance between 2 uncorrelated Poisson spike trains in 1. The result is divided by sqrt(2) due to
        the implementation in pymuvr to get similar results as in the original paper.
        The function return the mean of these distance across each comparison of the spikelist_list for each channel.
        :param cos: ??? - does not seem to have much influence
        :param tau: Time constant for the exponential tail
        :return: mean_vr_dist, mean_vr_dist_norm, global_vr_dist, global_vr_dist_norm
        """
        mean_vr_dist, spikes_in_chan = np.zeros(self.n_channels), np.zeros(self.n_channels)
        mean_vr_dist_norm = np.zeros(self.n_channels)
        up_i, up_j = np.triu_indices(self.n_spikelist, 1)
        for i_chan in range(0, self.n_channels):
            obs = []
            for spklist in self.spikelist_list:
                obs.append([list(spklist.time[spklist.channel == i_chan])])
                spk_counts = np.array([np.sum(spklist.channel == i_chan) for spklist in self.spikelist_list])
            if np.array(obs).size > 0:
                vr_dist = pymuvr.square_dissimilarity_matrix(obs, cos, tau, 'distance')
                mean_vr_dist[i_chan] = np.mean(vr_dist[up_i, up_j])
                norm_mat = np.sqrt((np.tile(spk_counts, [len(obs), 1]) + np.tile(spk_counts, [len(obs), 1]).T) / 2)
                vr_dist_norm = vr_dist / (norm_mat + 1e-8)
                mean_vr_dist_norm[i_chan] = np.mean(vr_dist_norm[up_i, up_j])
                spikes_in_chan[i_chan] = 1
        global_vr_dist = mean_vr_dist[spikes_in_chan.astype(bool)].mean()
        global_vr_dist_norm = mean_vr_dist_norm[spikes_in_chan.astype(bool)].mean()
        return mean_vr_dist, mean_vr_dist_norm, global_vr_dist, global_vr_dist_norm

    def compute_fano_factor(self):
        """
        Compute the Fano Factor (~dispersion) of the number of spikes in each spike list, for each channel.
        Fano factor is defined as the variance divided by the mean of a random process
        The global fano factor returned is the mean across channels
        :return: fano_factor, global_fano_factor
        """
        fano_factor = np.zeros(self.n_channels)
        for i_chan in range(0, self.n_channels):
            spk_counts = np.array([np.sum(spklist.channel == i_chan) for spklist in self.spikelist_list])
            if spk_counts.sum() > 0:
                fano_factor[i_chan] = spk_counts.var() / spk_counts.mean()
            else:
                fano_factor[i_chan] = 0
            if np.sum(fano_factor > 0) > 0:
                global_fano_factor = fano_factor[fano_factor > 0].mean()
            else:
                global_fano_factor = 0
        return fano_factor, global_fano_factor


def import_spikelist_from_mat(import_path, n_channels):
    mat_file = sio.loadmat(import_path)
    try:
        spike_list_data = mat_file['spike_list']
    except:
        try:
            spike_list_data = mat_file[list(mat_file.keys())[-1]]
        except:
            raise ValueError('Could not read spike list')
    if spike_list_data.shape[1] < 4:
        potential = np.zeros(spike_list_data.shape[0])
    else:
        potential = spike_list_data[:, 3]
    return SpikeList(spike_list_data[:, 0], spike_list_data[:, 1] - 1, spike_list_data[:, 2], potential,
                     n_channels=n_channels)


def dual_spikelist_plot(spike_list_in, spike_list_out, potential_thresh_in=0, potential_thresh_out=0, tau_lif=[],
                        pattern_id_sel_in=[], pattern_id_sel_out=[]):
    f = plt.figure()
    ax0 = plt.subplot2grid((8, 7), (0, 1), rowspan=3, colspan=5)
    ax1 = plt.subplot2grid((8, 7), (3, 1), rowspan=1, colspan=5, sharex=ax0)
    ax2 = plt.subplot2grid((8, 7), (0, 6), rowspan=3, sharey=ax0)
    ax3 = plt.subplot2grid((8, 7), (0, 0), rowspan=3, colspan=1, sharey=ax0)
    ax_list = spike_list_in.plot(ax_list=[ax0, ax1, ax2, ax3], potential_thresh=potential_thresh_in, tau_lif=tau_lif,
                                 pattern_id_sel=pattern_id_sel_in)
    if spike_list_in.n_spikes > 0:
        ax_list[1].set(xlabel='')
        ax0 = ax_list[0]
    ax4 = plt.subplot2grid((8, 7), (4, 1), rowspan=3, colspan=5, sharex=ax0, sharey=ax0)
    ax5 = plt.subplot2grid((8, 7), (7, 1), rowspan=1, colspan=5, sharex=ax0)
    ax6 = plt.subplot2grid((8, 7), (4, 6), rowspan=3, sharey=ax0)
    ax7 = plt.subplot2grid((8, 7), (4, 0), rowspan=3, colspan=1)
    spike_list_out.plot(ax_list=[ax4, ax5, ax6, ax7], potential_thresh=potential_thresh_out,
                        pattern_id_sel=pattern_id_sel_out)
    ax4.legend([])
    ax4.set_xlim((spike_list_in.tmin, spike_list_in.tmax))
    return [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]

