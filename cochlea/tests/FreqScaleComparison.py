import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from cochlea_utils import *
import cochlea
from scipy import signal

n_channels = 50
fs = 44100
fmin = 200
fmax = 10000
forder = 2

wn_erb, cf_erb = erbscale(fs, fmin, fmax, n_channels)
wn_music, cf_music = musicscale(fs, fmin, fmax, n_channels)
wn_linear, cf_linear = linearscale(fs, fmin, fmax, n_channels)
bw_erb = (wn_erb[:, 1] - wn_erb[:, 0]) * fs / 2
bw_music = (wn_music[:, 1] - wn_music[:, 0]) * fs / 2
bw_linear = (wn_linear[:, 1] - wn_linear[:, 0]) * fs / 2


f = plt.figure()
ax = f.add_subplot(111)
marksize = 5
ax.plot(range(0, n_channels), cf_erb, marker="o")
# ax.plot(range(0, n_channels), cf_music, marker="o")
ax.plot(range(0, n_channels), cf_linear, marker="o")
ax.legend(['Erb', 'Linear'])
ax.set(xlabel='Channel number', ylabel='Filter center frequency (Hz)', title='Central Frequency Evolution')
ax2 = f.add_subplot(122, sharex=ax)
ax2.plot(range(0, n_channels), bw_erb, marker="o")
# ax2.plot(range(0, n_channels), bw_music, marker="o")
ax2.plot(range(0, n_channels), bw_linear, marker="o")
ax2.legend(['Erb', 'Linear'])
ax2.set(xlabel='Channel number', ylabel='Filter bandwidth (Hz)', title='Bandwidth Evolution')

