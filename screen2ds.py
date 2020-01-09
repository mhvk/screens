"""1-D scatering screen and its dynamic and secondary spectra.

Create a dynamic wave field based on a set of scattering points.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u, constants as const
from astropy.visualization import quantity_support


def expand2(*arrays):
    """Add two unity axes to all arrays."""
    return [np.reshape(array, np.shape(array)+(1, 1))
            for array in arrays]


# Construct the dynamic spectrum directly
def get_dynwave(theta_par, theta_perp, realization, d_eff, mu_eff, f, t):
    """Given a set of scattering points, construct the dynamic wave field.

    Parameters
    ----------
    theta_par : ~astropy.units.Quantity
        Angles of the scattering point in the direction parallel to ``mu_eff``
    theta_perp : ~astropy.units.Quantity
        Angles perpendiculat to ``mu_eff``.
    realization : array-like
        Complex amplitudes of the scattering points
    d_eff : ~astropy.units.Quantity
        Effective distance.  Should be constant; if different for
        different points, no screen-to-screen scattering is taken into
        account.
    mu_eff : ~astropy.units.Quantity
        Effective proper motion (``v_eff / d_eff``), parallel to ``theta_par``.
    t : ~astropy.units.Quantity
        Times for which the dynamic wave spectrum should be calculated.
    f : ~astropy.units.frequency
        Frequencies for which the spectrum should be calculated.

    Returns
    -------
    dynwave : array
        Delayed wave field array, with last axis time, second but last
        frequency, and earlier axes as given by the other parameters.
    """
    theta_par, theta_perp, realization, d_eff, mu_eff = expand2(
        theta_par, theta_perp, realization, d_eff, mu_eff)
    th_par = theta_par + mu_eff * t
    tau_t = (d_eff / (2*const.c)) * (th_par**2 + theta_perp**2)
    phase = (f[:, np.newaxis] * u.cycle * tau_t).to_value(
        u.one, u.dimensionless_angles())
    return realization * np.exp(-1j * phase)


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

dynwave = get_dynwave(th, th_perp, realization, d_eff, mu_eff, f, t)

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
