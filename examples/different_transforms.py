# Licensed under the GPLv3 - see LICENSE
"""Simulate a 1-D scatering screen and show its dynamic and secondary spectra.

Run as a script to see the effect of different transforms on blending
for a large frequency range: straight fourier transform on frequency
and time, after rebinning on a constant wavelength grid, and after
rebinning time to frequency times time (nu-t transform).

"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u, constants as const
from scipy.signal.windows import tukey

from screens.fields import dynamic_field
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
# scale = 1  # Only 5 MHz at 1320 MHz, all transforms very similar.
scale = 1/10  # 500 MHz, nu t only good one.

# Create scattering screen, using a centred Gaussian for the amplitudes.
sig = 3*u.mas
if arclets:
    th = np.linspace(-10, 10, 28, endpoint=False) << u.mas
    a = (0.3*np.exp(-0.5*(th/sig)**2)
         + 0.03*np.exp(-0.5*((th-6*u.mas)/sig)**2)).to_value(u.one)
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
t = np.linspace(-30*u.minute, 30*u.minute, 200,
                endpoint=False)[:, np.newaxis] / scale / 2
df = f[1] - f[0]
dt = t[1, 0] - t[0, 0]

# Calculate dynamic spectrum.
dynspec = np.abs(dynamic_field(th, 0., realization, d_eff, mu_eff, f, t).sum(0))**2
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

# And turn it into a regular secondary spectrum (straight FT)
cs = CS.from_dynamic_spectrum(ds)
cs.tau <<= u.us  # nicer than 1/MHz
cs.fd <<= u.mHz  # nicer than 1/min
dfd = cs.fd[1, 0] - cs.fd[0, 0]
dtau = cs.tau[1] - cs.tau[0]

plt.subplot(231)
# TODO: really, should just use a WCS!
ds_extent = ((t[0, 0] - dt/2).value, (t[-1, 0] + dt/2).value,
             (f[0] - df/2).value, (f[-1] + df/2).value)
plt.imshow(ds.dynspec.T, origin='lower', aspect='auto', extent=ds_extent,
           cmap='Greys')
plt.xlabel(rf"$t\ ({ds.t.unit.to_string('latex')[1:-1]})$")
plt.ylabel(rf"$f\ ({ds.f.unit.to_string('latex')[1:-1]})$")
plt.title(rf"$\nu - t$")
plt.colorbar()

plt.subplot(234)
ss_extent = ((cs.fd[0, 0] - dfd/2).value, (cs.fd[-1, 0] + dfd/2).value,
             (cs.tau[0] - dtau/2).value, (cs.tau[-1] + dtau/2).value)
plt.imshow(np.log10(cs.secspec.T), origin='lower', aspect='auto', extent=ss_extent,
           cmap='Greys', vmin=-9, vmax=-2)
plt.xlabel(rf"$f_{{D}}\ ({cs.fd.unit.to_string('latex')[1:-1]})$")
plt.ylabel(rf"$\tau\ ({cs.tau.unit.to_string('latex')[1:-1]})$")
plt.colorbar()

# Rebin frequency to wavelength.
w = np.linspace(const.c / f[0], const.c / f[-1], f.shape[-1]).to(u.cm)
dw = w[1] - w[0]
_ds = np.stack([np.interp(const.c/w, f, _d) for _d in dynspec])
ds_w = DS(_ds, f=w, t=t, noise=noise)
# And turn it into a secondary spectrum (straight FT)
cs_w = CS.from_dynamic_spectrum(ds_w)
cs_w.fd <<= u.mHz
dfl = cs_w.tau[1] - cs_w.tau[0]

plt.subplot(232)
ds_w_extent = ds_extent[:2] + ((w[0] - dw/2).value, (w[-1] + dw/2).value)
plt.imshow(ds_w.dynspec.T, origin='lower', aspect='auto', extent=ds_w_extent,
           cmap='Greys')
plt.xlabel(rf"$t\ ({ds_w.t.unit.to_string('latex')[1:-1]})$")
plt.ylabel(rf"$\lambda\ ({ds_w.f.unit.to_string('latex')[1:-1]})$")
plt.title(rf"$\lambda - t$")
plt.colorbar()

plt.subplot(235)
ss_w_extent = ss_extent[:2] + (
    (cs_w.tau[0] - dfl/2).value, (cs_w.tau[-1]+dfl/2).value)
plt.imshow(np.log10(cs_w.secspec.T), origin='lower', aspect='auto',
           extent=ss_w_extent, cmap='Greys', vmin=-9, vmax=-2)
plt.xlabel(rf"$f_{{D}}\ ({cs_w.fd.unit.to_string('latex')[1:-1]})$")
plt.ylabel(rf"$f_{{\lambda}}\ ({cs_w.tau.unit.to_string('latex_inline')[1:-1]})$")
plt.colorbar()

# Rebin time to t / f so it becomes a nu t transform
tt = t * f.mean() / f
_ds = np.stack([np.interp(_t, t[:, 0], _d) for _t, _d in zip(tt.T, dynspec.T)]).T
ds_t = DS(_ds, f=f, t=t, noise=noise)

nut = CS.from_dynamic_spectrum(ds_t)
nut.tau <<= u.us
nut.fd <<= u.mHz
# For comparison, should be nearly the same.
# nut2 = CS.from_dynamic_spectrum(dynspec, f=f, t=t*f/f.mean(), fd=nut.fd[:, 0])
# nut2.tau <<= u.us
# nut2.fd <<= u.mHz

plt.subplot(233)
plt.imshow(ds_t.dynspec.T, origin='lower', aspect='auto', extent=ds_extent,
           cmap='Greys')
plt.xlabel(rf"$t(\nu/\bar{{\nu}})\ ({ds_t.t.unit.to_string('latex')[1:-1]})$")
plt.ylabel(rf"$\nu\ ({ds_t.f.unit.to_string('latex')[1:-1]})$")
plt.title(rf"$\nu - \nu t$")
plt.colorbar()

plt.subplot(236)
plt.imshow(np.log10(nut.secspec.T), origin='lower', aspect='auto', extent=ss_extent,
           cmap='Greys', vmin=-9, vmax=-2)
plt.xlabel(rf"$f_{{D}}\ ({nut.fd.unit.to_string('latex')[1:-1]})$")
plt.ylabel(rf"$\tau\ ({nut.tau.unit.to_string('latex')[1:-1]})$")
plt.colorbar()
