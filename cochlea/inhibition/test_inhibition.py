import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set()
sns.set_context('paper')

vmin, vmax = -1, 1
a = 0.4
v1 = np.linspace(vmin, vmax, 1000)
v2 = v1
n_pnts = v1.size

# Inhibition 1 : Forward
i1_sf, i1_sb, i1_shuntf, i1_shuntb = np.zeros((4, n_pnts, n_pnts))
i2_sf, i2_sb, i2_shuntf, i2_shuntb = np.zeros((4, n_pnts, n_pnts))
for i, v2_i in enumerate(v2):
    for j, v1_i in enumerate(v1):
        i1_sf[i, j] = v1_i - a*v2_i
        i2_sf[i, j] = v2_i - a*v1_i
        i1_sb[i, j] = (v1_i - a*v2_i) / (1 - a**2)
        i2_sb[i, j] = (v2_i - a*v1_i) / (1 - a**2)
        i1_shuntf[i, j] = v1_i / (1 + a*v2_i)
        i2_shuntf[i, j] = v2_i / (1 + a*v1_i)

f = plt.figure()
ax = f.add_subplot(111)
im = ax.pcolormesh(v1, v2, i1_sf, cmap="RdBu_r", norm=mpl.colors.Normalize(vmin=i1_sf.min(), vmax=i1_sf.max()))
plt.colorbar(im)
ax.set(xlabel='v1', ylabel='v2', title='Forward Inhibition : i1=v1 - a*v2, a={}'.format(a))

f = plt.figure()
ax = f.add_subplot(111)
im = ax.pcolormesh(v1, v2, i1_sb, cmap="RdBu_r", norm=mpl.colors.Normalize(vmin=i1_sb.min(), vmax=i1_sb.max()))
plt.colorbar(im)
ax.set(xlabel='v1', ylabel='v2', title='Backward Inhibition : i1=v1 - a*i2, a={}'.format(a))

f = plt.figure()
ax = f.add_subplot(111)
im = ax.pcolormesh(v1, v2, i1_shuntf, cmap="RdBu_r", norm=mpl.colors.Normalize(vmin=i1_shuntf.min(), vmax=i1_shuntf.max()))
plt.colorbar(im)
ax.set(xlabel='v1', ylabel='v2', title='Shunting Forward : i1=v1 / (1 + a*v2), a={}'.format(a))


# Contrast Enhancement : Output ratio / Input ratio i.e.: (i1/i2) / (v1/v2)
# v2 = r * v1 --> v1/v2 = 1/r with 0 < r < 1 i.e v1 > v2
N = 1000
r = np.linspace(0.05, a-0.03, N)
magn_sf = (1 - a*r) / (r-a) * r
magn_shuntf = np.zeros((v1.size, N))  # magn_shuntf = f(v1, r)
for i, v1_i in enumerate(v1):
    for j, r_i in enumerate(r):
        magn_shuntf[i, j] = (1/r_i) * (1 + a*v1_i) / (1 + a*r_i*v1_i) * r_i

f = plt.figure()
ax = f.add_subplot(111)
ax.plot(1/r, magn_sf)
ax.set(xlabel='v1/v2', ylabel='i1/i2', title='Magnification - Subtractive Forward')

f = plt.figure()
ax = f.add_subplot(111)
im = ax.pcolormesh(1/r, v1, magn_shuntf, cmap="RdBu_r", norm=mpl.colors.Normalize(vmin=magn_shuntf.min(), vmax=magn_shuntf.max()))
plt.colorbar(im)
ax.set(xlabel='v1/v2', ylabel='v1', title='Magnification - Shunting Forward')

# for i, r_i in enumerate(r):
#     magn_sf[i] = (1 - a*r_i) / (r_i - a)
#     magn_sb[i] = magn_sf[i]


