import matplotlib.pyplot as plt
import spikelist


class SpikePacket:
    def __init__(self, spike_list, packet_size, overlap=0):
        if not type(spike_list) == spikelist.SpikeList:
            raise ValueError('Argument spike_list must be a SpikeList object')
        if not type(packet_size) == int:
            raise ValueError('Argument packet_size must be a scalar integer')
        self.spike_list = spike_list
        self.packet_size = packet_size
        self.overlap = 0
        self.packets, self.spike_ind, self.packet_time = spike_list.convert_to_packet(packet_size, overlap)
        self.n_spikes = len(spike_list)
        self.n_packets = len(self.packet_time)
        self.n_channels = spike_list.n_channels
        self.name = 'Spike Packet List '.format(spike_list.name)

    def __str__(self):
        desc_str = self.name
        desc_str += '\n{} packets, {} spikes - packet size : {}, overlap : {}'.format(self.n_packets, self.n_spikes,
                                                                                      self.packet_size, self.overlap)
        return desc_str

    def plot(self):
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.imshow(self.packets, origin='lower', aspect='auto')
        ax.set(xlabel='Packet number', ylabel='Channel', title=self.name)


# def overlay_same_packet_pattern(spike_packet_list, pattern_ids):
#     if not type(spike_packet_list) == list:
#         raise ValueError('Argument spike_packet_list should a be a list of SpikePacket objects')
#     if not type(spike_packet_list[0]) == SpikePacket:
#         raise ValueError('Argument spike_packet_list should a be a list of SpikePacket objects')
#     unique_pattern_ids = np.unique(pattern_ids)
#     n_max_trials = np.max([len(np.where(pattern_ids == pat_id)[0]) for pat_id in unique_pattern_ids])
#     n_patterns = np.unique(pattern_ids).size
#     f = plt.figure()
#     trial_colors = sns.color_palette(n_colors=n_max_trials)
#     for i in range(0, n_patterns):
#         ax = f.add_subplot(1, n_patterns, i+1)
#         pattern_i_ind = np.where(pattern_ids == unique_pattern_ids[i])[0]
#         spkpacket_pattern_i = [spike_packet_list[k] for k in pattern_i_ind]
#         legend_str = []
#         for j, spkpacket in enumerate(spkpacket_pattern_i):
#             # Shift time origin so that each trial start at the same time
#             ax.imshow(spkpacket.packets, origin='lower', aspect='auto', alpha=0.2)
#         ax.set_ylim(0, spkpacket.n_channels)
#         ax.set(xlabel='time', ylabel='channel', title=spkpacket.spike_list.pattern_names[unique_pattern_ids[i]])
#         # ax.legend(legend_str, loc='upper right')




