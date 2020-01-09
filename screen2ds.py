"""1-D scatering screen and its dynamic and secondary spectra.

Create a dynamic wave field based on a set of scattering points.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u, constants as const
from astropy.visualization import quantity_support

from fields import dynamic_field


plt.ion()
quantity_support()
plt.clf()
np.random.seed(1234)

# Set scalings.
lobs = 1. * u.m
fobs = const.c / lobs
d_eff = 1 * u.kpc
mu_eff = 100 * u.mas / u.yr

# Create scattering screen, composed of a centred and an offset Gaussian.
th1 = np.linspace(-8, 8, 64, endpoint=False) << u.mas
th2 = np.array([4.5, 4.7, 4.8, 4.9, 5., 5.2, 5.5, 6.5, 8.5]) << u.mas
sig = 1.5*u.mas
a1 = 0.3*np.exp(-0.5*(th1/sig)**2)
a2 = 0.03 * np.exp(-0.5*((th2-6*u.mas)/sig)**2)
th = np.hstack((th1.value, th2.value)) << u.mas
a = np.hstack((a1.value, a2.value))
th_perp = np.hstack((-th1.value/20, .2*np.ones(th2.shape))) << u.mas
realization = a * np.random.normal(size=th.shape+(2,)).view('c16').squeeze(-1)
# Make direct line of sight a bit brighter.
realization[th1.size // 2] = 1
# realization[th.size // 2-4] = 0.1

# plt.plot(th, realization.real)

f = np.linspace(-0.5, 0.5, 200, endpoint=False) << u.MHz
f += fobs
t = (np.linspace(-500, 500, 80, endpoint=False) << u.s).to(u.minute)


ax1 = plt.subplot(131)
plt.plot(th, th_perp, '+')
plt.xlim(th.min()*1.05, th.max()*1.04)
plt.ylim(th.min()*1.04, th.max()*1.05)
ax1.set_aspect(1.)

dynwave = dynamic_field(th, th_perp, realization, d_eff, mu_eff, f, t)

# Just for fun, add noise.
noise = np.random.normal(scale=0.05,
                         size=dynwave.shape[-2:]+(2,)).view('c16').squeeze(-1)
axes = tuple(range(0, dynwave.ndim-2))
dynspec = np.abs(dynwave[...].sum(axes) + noise)**2
dynspec /= dynspec.mean()

plt.subplot(132)
ds_extent = (t[0].value, t[-1].value, f[0].value, f[-1].value)
plt.imshow(dynspec, origin=0, aspect='auto', extent=ds_extent, cmap='Greys')
plt.xlabel(t.unit.to_string('latex'))
plt.ylabel(f.unit.to_string('latex'))
plt.colorbar()


# And turn it into a secondary spectrum.
sec = np.fft.fft2(dynspec)
sec /= sec[0, 0]
tau = np.fft.fftfreq(dynspec.shape[0], f[1] - f[0]).to(u.us)
fd = np.fft.fftfreq(dynspec.shape[1], t[1] - t[0]).to(u.mHz)

sec = np.fft.fftshift(sec)
tau = np.fft.fftshift(tau) << tau.unit
fd = np.fft.fftshift(fd) << fd.unit

ss = np.maximum(np.abs(sec)**2, 1e-30)

plt.subplot(133)
sec_extent = (fd[0].value, fd[-1].value, tau[0].value, tau[-1].value)
plt.imshow(np.log10(ss), origin=0, aspect='auto', extent=sec_extent,
           cmap='Greys', vmin=-7, vmax=0)
plt.xlabel(fd.unit.to_string('latex'))
plt.ylabel(tau.unit.to_string('latex'))
plt.colorbar()
