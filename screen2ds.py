"""1-D scatering screen and its dynamic and secondary spectra.

Create a dynamic wave field based on a set of scattering points.

Runs as a script to create a simulated dynamic spectrum, plotting
the result, and with the grid of theta that might be used to
reproduce it overlaid.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u, constants as const
from astropy.time import Time
from astropy.visualization import quantity_support

from scintillometry.io import hdf5

from fields import dynamic_field, theta_grid, theta_theta_indices


plt.ion()
quantity_support()
plt.clf()
np.random.seed(654321)

# Set scalings.
fobs = 330. * u.MHz
d_eff = 1 * u.kpc
mu_eff = 100 * u.mas / u.yr

simple = True

# Create scattering screen, composed of a centred and an offset Gaussian.
sig = 1.5*u.mas
if simple:
    th = np.linspace(-7, 7, 28, endpoint=False) << u.mas
    th_perp = np.zeros_like(th)
    a = (0.3*np.exp(-0.5*(th/sig)**2)
         + 0.03*np.exp(-0.5*((th-5*u.mas)/sig)**2)).to_value(1)
else:
    th1 = np.linspace(-8, 8, 32, endpoint=False) << u.mas
    a1 = 0.3*np.exp(-0.5*(th1/sig)**2)
    th2 = np.array([4.5, 4.7, 4.8, 4.9, 5., 5.2, 5.5, 6.5, 8.5]) << u.mas
    a2 = 0.03 * np.exp(-0.5*((th2-5*u.mas)/sig)**2)
    th = np.hstack((th1.value, th2.value)) << u.mas
    a = np.hstack((a1.value, a2.value))
    th_perp = np.hstack((-th1.value/20, .2*np.ones(th2.shape))) << u.mas

realization = a * np.random.normal(size=th.shape+(2,)).view('c16').squeeze(-1)
# Make direct line of sight a bit brighter.
# TODO: this image should not really have proper motion!
realization[np.where(th == 0)] = 1
# realization[th.size // 2+8] = 0.1
# realization[th.size // 2+4] = 0.1

# Normalize so we should get a dynamic spectrum of unity mean
realization /= np.sqrt((np.abs(realization)**2).sum())

# plt.plot(th, realization.real)

# Smallest theta corresponds to 0.15 us -> 6.5 MHz of bandwidth
# But also need resolution of 0.013 MHz -> factor 500 -> too much.
# Instead rely on f_d for those small theta.

f = fobs + np.linspace(-0.5*u.MHz, 0.5*u.MHz, 200, endpoint=False)
t = np.linspace(-10*u.minute, 10*u.minute, 100, endpoint=False)

ax1 = plt.subplot(131)
plt.scatter(th, th_perp, marker='o', s=np.maximum(np.abs(realization*40), 0.5))
plt.xlim(th.min()*1.05, th.max()*1.05)
plt.ylim(th.min()*1.05, th.max()*1.05)
ax1.set_aspect(1.)

dynwave = dynamic_field(th, th_perp, realization, d_eff, mu_eff, f, t)
axes = tuple(range(0, dynwave.ndim-2))
dynspec = np.abs(dynwave[...].sum(axes))**2

# Just for fun, add noise.  Add as gaussian noise, since in a real
# observation, the dynamic spectrum would consist of folded data of
# a lot of background-subtracted pulses, so intensities would no longer
# be chi2 distributed.
noise = 0.01
dynspec += noise * np.random.normal(size=dynspec.shape)

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


# Plot secondary spectrum, with thetas from a default grid that would be
# used for it overlaid.
plt.subplot(133)

tau_max = (1./(f[3]-f[0])).to(u.us)
th_g = theta_grid(d_eff, mu_eff, f, t, tau_max=tau_max)
fd_g = (d_eff/const.c*mu_eff*fobs*th_g).to(
    u.mHz, equivalencies=u.dimensionless_angles())
tau_g = (d_eff/(2*const.c)*th_g**2).to(
    u.us, equivalencies=u.dimensionless_angles())
i0, i1 = theta_theta_indices(th_g)
plt.plot(fd_g[i0]-fd_g[i1], tau_g[i0]-tau_g[i1], 'bo', ms=0.2)
plt.plot(fd_g, tau_g, 'ro', ms=0.4)
sec_extent = (fd[0].value, fd[-1].value, tau[0].value, tau[-1].value)
plt.imshow(np.log10(ss), origin=0, aspect='auto', extent=sec_extent,
           cmap='Greys', vmin=-7, vmax=0)
plt.xlabel(fd.unit.to_string('latex'))
plt.ylabel(tau.unit.to_string('latex'))
plt.colorbar()


# Save the simulated dynamic spectrum for later use.
with hdf5.open('dynspec.h5', 'w', sample_shape=dynspec.shape[:1],
               sample_rate=(1/(t[1]-t[0])).to(u.mHz),
               samples_per_frame=dynspec.shape[-1],
               dtype=dynspec.dtype, time=Time.now(), frequency=f,
               sideband=1) as fw:
    fw.write(dynspec.T)
    fw.fh_raw.create_dataset('realization', data=realization)
    fw.fh_raw.create_dataset('theta', data=th.value)
    fw.fh_raw.attrs['noise'] = noise
