"""1-D scatering screen and its dynamic and secondary spectra.

Creates a one-dimensional, gaussian screen, and attempts to create
corresponding dynamic and secondary spectra both by inserting points
in a secondary wavefield and FTing, and by calculating the dynamic
spectrum directly, as a superposition of delayed fields.  The result
turns out to be sensitive to the gridding of the secondary spectrum.

In doppler frequency, one can easily bin correctly, but not so in tau
unless one takes very fine bins.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u, constants as const
from astropy.visualization import quantity_support


plt.ion()
quantity_support()
plt.clf()
np.random.seed(123456)


# Set scalings.
lobs = 1.*u.m
fobs = const.c / lobs
fd_scale = 10*u.mHz / (10. * u.mas)
tau_scale = 100*u.us / (10.*u.mas)**2
mu_eff = (fd_scale * lobs / (2. * const.c * tau_scale)).to(u.mas/u.yr)


# Create scattering screen, composed of a centred and an offset Gaussian.
th = np.linspace(-10*u.mas, 10*u.mas, 40, endpoint=False)
sig = 1.5*u.mas
a = 0.3*np.exp(-0.5*(th/sig)**2) + 0.03 * np.exp(-0.5*((th-5*u.mas)/sig)**2)
realization = a * np.random.normal(size=th.shape+(2,)).view('c16').squeeze(-1)
# Make direct line of sight a bit brighter.
realization[th.size // 2] = 1
# # For tests where realization is 0.
# realization[th.size // 2 - 4] = 1

# plt.plot(th, realization.real)

# Create a secondary spectrum and fill spots approximately.
tau_grid = np.linspace(-200*u.us, 200*u.us, 200, endpoint=False)
dtau = tau_grid[1] - tau_grid[0]
fd_grid = np.linspace(-20*u.mHz, 20*u.mHz, 80, endpoint=False)
dfd = fd_grid[1] - fd_grid[0]
wavefield = np.zeros(tau_grid.shape+fd_grid.shape, 'c16')
tau = th**2 * tau_scale
fd = th * fd_scale
itau = np.round(((tau - tau_grid[0]) / dtau).to_value(1)).astype(int)
ifd = np.round(((fd - fd_grid[0]) / dfd).to_value(1)).astype(int)
# Add geometric delay at observed frequency to the realization.
to_add = realization * np.exp(-2j*np.pi*fobs*tau)
np.add.at(wavefield, (itau, ifd), to_add)

sec_extent = (fd_grid[0].value, fd_grid[-1].value,
              tau_grid[0].value, tau_grid[-1].value)

wave_power = np.maximum(np.abs(wavefield)**2, 1e-30)
plt.subplot(221)
plt.imshow(np.log10(wave_power), origin=0, aspect='auto',
           extent=sec_extent, cmap='Greys', vmax=0, vmin=-5)
plt.xlabel(fd_grid.unit.to_string('latex'))
plt.ylabel(tau_grid.unit.to_string('latex'))

# Use the approximate secondary spectrum to create a dynamic spectrum.
dynwave = np.fft.fft2(np.fft.fftshift(wavefield))
f_grid = np.fft.fftfreq(tau_grid.size, dtau).to(u.MHz)
t_grid = np.fft.fftfreq(fd_grid.size, dfd).to(u.minute)

dynwave = np.fft.fftshift(dynwave)
t_grid = np.fft.fftshift(t_grid) << t_grid.unit
f_grid = np.fft.fftshift(f_grid) << f_grid.unit

ds_extent = (t_grid[0].value, t_grid[-1].value,
             f_grid[0].value, f_grid[-1].value)

# Just for fun, add noise.
noise = np.random.normal(scale=0.1,
                         size=dynwave.shape+(2,)).view('c16').squeeze(-1)

dynspec = np.abs(dynwave+noise)**2
dynspec /= dynspec.mean()

plt.subplot(222)
plt.imshow(dynspec, origin=0, aspect='auto', extent=ds_extent, cmap='Greys')
plt.xlabel(t_grid.unit.to_string('latex'))
plt.ylabel(f_grid.unit.to_string('latex'))


# Now calculate the corresponding "observed" secondary spectrum.
secspec = np.fft.ifft2(dynspec)
secspec /= secspec[0, 0]
secspec = np.fft.ifftshift(secspec)
secpower = np.maximum(np.abs(secspec)**2, 1e-30)
plt.subplot(223)
plt.imshow(np.log10(secpower), origin=0, aspect='auto',
           extent=sec_extent, vmin=-7, vmax=0, cmap='Greys')
plt.xlabel(fd_grid.unit.to_string('latex'))
plt.ylabel(tau_grid.unit.to_string('latex'))

# For comparison, contruct dynamic spectrum directly
f = f_grid[:, np.newaxis, np.newaxis] + fobs
t = t_grid[:, np.newaxis]
th_t = th + t * mu_eff

dynw = realization * np.exp(-2j * np.pi * tau_scale * f * th_t**2)

dynspec2 = np.abs(dynw[...].sum(-1))**2

plt.subplot(224)
plt.imshow(dynspec2, origin=0, aspect='auto', extent=ds_extent, cmap='Greys')
plt.xlabel(t_grid.unit.to_string('latex'))
plt.ylabel(f_grid.unit.to_string('latex'))
