"""1-D scatering screen and its dynamic and secondary spectra.

Create a dynamic wave field based on a set of scattering points.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u, constants as const
from astropy.visualization import quantity_support


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
th = np.linspace(-10*u.mas, 10*u.mas, 64, endpoint=False)
sig = 1.5*u.mas
a = 0.3*np.exp(-0.5*(th/sig)**2) + 0.03 * np.exp(-0.5*((th-5*u.mas)/sig)**2)
realization = a * np.random.normal(size=th.shape+(2,)).view('c16').squeeze(-1)
# Make direct line of sight a bit brighter.
realization[th.size // 2] = 1
# realization[th.size // 2-4] = 0.1

# plt.plot(th, realization.real)

# Construct the dynamic spectrum directly
f = np.linspace(-0.5, 0.5, 200, endpoint=False) << u.MHz
f += fobs
t = (np.linspace(-500, 500, 80, endpoint=False) << u.s).to(u.minute)
th_t = th[..., np.newaxis, np.newaxis] + mu_eff * t
tau_t = ((d_eff / (2*const.c)) * th_t ** 2).to(
    u.s, equivalencies=u.dimensionless_angles())
dynwave = (realization[..., np.newaxis, np.newaxis]
           * np.exp(-2j * np.pi * f[:, np.newaxis] * tau_t))

# Just for fun, add noise.
noise = np.random.normal(scale=0.05,
                         size=dynwave.shape[-2:]+(2,)).view('c16').squeeze(-1)
dynspec = np.abs(dynwave[...].sum(0) + noise)**2
dynspec /= dynspec.mean()

plt.subplot(121)
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

plt.subplot(122)
sec_extent = (fd[0].value, fd[-1].value, tau[0].value, tau[-1].value)
plt.imshow(np.log10(ss), origin=0, aspect='auto', extent=sec_extent, cmap='Greys',
           vmin=-7, vmax=0)
plt.xlabel(fd.unit.to_string('latex'))
plt.ylabel(tau.unit.to_string('latex'))
plt.colorbar()
