# Licensed under the GPLv3 - see LICENSE
"""Simulate a 1-D scatering screen and show its dynamic and secondary spectra.

Run as a script to see the effect of different dispersion measure gradients
over the screen, and how these can be undone by appropriate frequency-
dependent time shifts.

"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u, constants as const
from scipy.signal.windows import tukey

from baseband_tasks.dm import DispersionMeasure

from screens.fields import dynamic_field, phasor
from screens.dynspec import DynamicSpectrum as DS
from screens.conjspec import ConjugateSpectrum as CS

plt.ion()

np.random.seed(654321)

# Set scalings.
fobs = 1320. * u.MHz
d_eff = 0.25 * u.kpc
mu_eff = 100 * u.mas / u.yr

# Pick a sparse, strongly modulated screen with many arclets, or a dense,
# weakly modulated screen with only one arclet (latter more instructive).
arclets = False
# Possible scale to multiply theta values with.  Time and frequency
# are scaled correspondingly, i.e., scale = 1/7 means 49 times larger
# frequency range (2->98 MHz) and 7 times larger time range.
scale = 1  # Only 5 MHz at 1320 MHz, all transforms very similar.
scale = 1/10  # 500 MHz, nu t only good one.

# Create scattering screen, using a centred Gaussian for the amplitudes.
sig = 3*u.mas
if arclets:
    th = np.linspace(-7, 7, 28, endpoint=False) << u.mas
    a = (0.3*np.exp(-0.5*(th/sig)**2)
         + 0.03*np.exp(-0.5*((th-5*u.mas)/sig)**2)).to_value(u.one)
    realization = a * np.random.normal(size=th.shape+(2,)).view('c16').squeeze(-1)
    # Add a bright spot.
    realization[-3] = 0.5
else:
    th = np.linspace(-10, 10, 28*16, endpoint=False) << u.mas
    a = 0.01*np.exp(-0.5*(th/sig)**2).to_value(u.one)
    # For smooth arc, just randomize phases.
    realization = a * np.exp(2j*np.pi*np.random.uniform(size=th.shape))
    # On purpose introduce a bright spot and a gap, to check visibility.
    realization[4*16] = 0.03
    realization[-5*16:-5*16+8] = 0

# Make direct line of sight bright.
# TODO: this image should not really have proper motion!
realization[np.where(th == 0)] = 1

# Normalize so we should get a dynamic spectrum of unity mean
realization /= np.sqrt((np.abs(realization)**2).sum())

th *= scale
f = fobs + np.linspace(-2.5*u.MHz, 2.5*u.MHz, 400, endpoint=False) / scale**2
f <<= u.GHz
t = np.linspace(-30*u.minute, 30*u.minute, 200,
                endpoint=False)[:, np.newaxis] / scale / 2
df = f[1] - f[0]
dt = t[1, 0] - t[0, 0]

# Calculate dynamic spectrum.
dynwave = dynamic_field(th, 0., realization, d_eff, mu_eff, f, t)

# DM gradient
dt_max = [0, 20, 40] * u.ns
dt_dm1 = DispersionMeasure(1.).time_delay(f.min(), fobs)
dm_gradient = ((dt_max / dt_dm1) / th.max() * (u.pc / u.cm**3)).to(u.pc/u.cm**3/u.mas)
for i, (dmg, _dt) in enumerate(zip(dm_gradient, dt_max)):
    ddm = DispersionMeasure(dmg * th[:, np.newaxis, np.newaxis])
    phase_factor = ddm.phase_factor(f, fobs)
    dynspec = np.abs((dynwave * phase_factor).sum(0))**2
    # Add gaussian noise. OK since in a real observation, the dynamic spectrum
    # would consist of folded data of a lot of background-subtracted pulses, so
    # intensities would be chi2 distributed with large N, i.e., near Gaussian.
    noise = 0.02
    dynspec += noise * np.random.normal(size=dynspec.shape)
    # Normalize.
    dynspec /= dynspec.mean()
    # Smooth edges to reduce peakiness in sec. spectrum.
    alpha_nu = 0.25
    alpha_t = 0.5  # Bit larger so nu-t transform also is OK.
    taper = (tukey(dynspec.shape[-1], alpha=alpha_nu)
             * tukey(dynspec.shape[0], alpha=alpha_t)[:, np.newaxis])
    dynspec = (dynspec - 1.0) * taper + 1.0

    ds = DS(dynspec, f=f, t=t, noise=noise)

    # For plotting purposes, show the dynamic spectrum as seen by the nu t transform.
    tt = t * fobs / f
    _ds = np.stack([np.interp(_t, t[:, 0], _d) for _t, _d in zip(tt.T, dynspec.T)]).T
    ds_t = DS(_ds, f=f, t=t, noise=noise)

    # Make a plain nu t transform
    nut = CS.from_dynamic_spectrum(dynspec, f=f, t=t*(f/fobs))
    nut.tau <<= u.us
    # As well as one in which we take account of the frequency-dependent
    # delay introduced by the DM gradient.
    conv = (d_eff/const.c*mu_eff*f).to(1/(u.ks*u.mas), u.dimensionless_angles())
    phase_gradient = DispersionMeasure(dmg * u.mas).phase_delay(f, fobs) / u.mas
    delay = (phase_gradient / conv).to(t.unit, equivalencies=[(u.cycle, None)])
    nut2 = CS.from_dynamic_spectrum(dynspec, f=f, t=(t+delay)*(f/fobs))
    nut2.tau <<= u.us

    plt.subplot(3, 3, 1+i)
    ds_extent = ((t[0, 0] - dt/2).value, (t[-1, 0] + dt/2).value,
                 (f[0] - df/2).value, (f[-1] + df/2).value)
    plt.imshow(ds_t.dynspec.T, origin='lower', aspect='auto', extent=ds_extent,
               cmap='Greys')
    plt.xlabel(rf"$t(\nu/\bar{{\nu}})\ ({ds_t.t.unit.to_string('latex')[1:-1]})$")
    plt.ylabel(rf"$\nu\ ({ds_t.f.unit.to_string('latex')[1:-1]})$")
    plt.title(rf"$d DM/d\theta={dmg.to_string(format='latex', precision=2)[1:]}")
    plt.colorbar()

    plt.subplot(3, 3, 3+1+i)
    dfd = nut.fd[1, 0] - nut.fd[0, 0]
    dtau = nut.tau[1] - nut.tau[0]
    ss_extent = ((nut.fd[0, 0] - dfd/2).value, (nut.fd[-1, 0] + dfd/2).value,
                 (nut.tau[0] - dtau/2).value, (nut.tau[-1] + dtau/2).value)
    plt.imshow(np.log10(nut.secspec.T), origin='lower', aspect='auto',
               extent=ss_extent, cmap='Greys', vmin=-9, vmax=-2)
    plt.xlabel(rf"$f_{{D}}\ ({nut.fd.unit.to_string('latex')[1:-1]})$")
    plt.ylabel(rf"$\tau\ ({nut.tau.unit.to_string('latex')[1:-1]})$")
    plt.title(rf"$\nu - \nu t$")
    plt.colorbar()

    plt.subplot(3, 3, 6+1+i)
    plt.imshow(np.log10(nut2.secspec.T), origin='lower', aspect='auto',
               extent=ss_extent, cmap='Greys', vmin=-9, vmax=-2)
    plt.xlabel(rf"$f_{{D}}\ ({nut2.fd.unit.to_string('latex')[1:-1]})$")
    plt.ylabel(rf"$\tau\ ({nut2.tau.unit.to_string('latex')[1:-1]})$")
    plt.title(rf"$\nu - \nu t_{{corr}}$")
    plt.colorbar()
