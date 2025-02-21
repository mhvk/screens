# Licensed under the GPLv3 - see LICENSE
"""Fit part of the Brisken dynamic spectrum.

Example runs on Brisken file 'dynamic_spectrum_arar.npz'
# On CITA machines can use
# ln -s /mnt/scratch-lustre/simard/GB057/dynamic_spectra/dynamic_spectrum_arar.npz
"""  # noqa:E501

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from astropy.visualization import quantity_support

from screens import DynamicSpectrum
from screens.modeling import DynamicSpectrumModel
from screens.visualization import ThetaTheta


quantity_support()


a = np.load('dynamic_spectrum_arar.npz')
f_full = a['f_MHz'] << u.MHz
t_full = a['t_s'] << u.s
sel_t, sel_f = slice(292, 505), slice(50000, 50300)
t = t_full[sel_t]
f = f_full[sel_f]
dynspec = DynamicSpectrum(a['I'][sel_t, sel_f], t=t[:, np.newaxis],
                          f=f, noise=1)
ds = DynamicSpectrumModel(dynspec, d_eff=1.4*u.kpc,
                          mu_eff=50*u.mas/u.yr)


tau_max = 350*u.us
# This stores the best mu_eff, theta, magnification on the instance.
ds.locate_mu_eff(np.arange(30, 70, 2) << u.mas/u.yr, tau_max=tau_max,
                 oversample_tau=1.2, oversample_fd=1.4)

r = ds.curvature

plt.subplot(3, 4, 1)
plt.plot(r['mu_eff'], r['redchi2'])
plt.plot(ds.mu_eff, r['redchi2'].min(), 'r+')

th_th = ds.theta_theta()
th = ds.theta
th_kwargs = dict(extent=(th[0].value, th[-1].value)*2,
                 origin='lower', vmin=-7, vmax=0, cmap='Greys')
th_th_proj = ThetaTheta(ds.theta)
plt.subplot(3, 4, 2, projection=th_th_proj)
plt.imshow(np.log10(np.maximum(np.abs(th_th)**2, 1e-30)), **th_kwargs)
plt.xlabel(th.unit.to_string('latex'))
plt.ylabel(th.unit.to_string('latex'))

ds_kwargs = dict(extent=(ds.t[0, 0].value, ds.t[-1, 0].value,
                         ds.f[0].value, ds.f[-1].value),
                 origin='lower', aspect='auto', cmap='Greys',
                 vmin=0, vmax=7)
dynspec_r = ds.model()
dynspec_r *= ds.dynspec.mean()/dynspec_r.mean()
plt.subplot(3, 4, 3)
plt.imshow(dynspec_r.T, **ds_kwargs)

res_kwargs = ds_kwargs.copy()
res_kwargs['vmin'], res_kwargs['vmax'] = -5, 5
plt.subplot(3, 4, 4)
plt.imshow((ds.dynspec-dynspec_r).T/ds.noise, **res_kwargs)

plt.subplot(3, 4, 7)
plt.imshow(ds.dynspec.T, **ds_kwargs)

(raw_mag_fit, raw_mag_err,
 raw_mu_eff_fit, raw_mu_eff_err) = ds.fit(
     guess='curvature', verbose=3,
     ftol=0.1/ds.dynspec.size)

plt.subplot(3, 4, 5)
plt.plot(th, np.abs(raw_mag_fit), th, np.abs(raw_mag_err))

rd = np.sqrt(ds.pcovar.diagonal())
rc = ds.pcovar/rd/rd[:, np.newaxis]
plt.subplot(3, 4, 6)
plt.imshow(rc, vmin=-1, vmax=1)

plt.subplot(3, 4, 8)
plt.imshow(ds.residuals(raw_mag_fit, raw_mu_eff_fit).T,
           **res_kwargs)

(cln_mag_fit, cln_mag_err,
 cln_mu_eff_fit, cln_mu_eff_err) = ds.cleanup_fit()

plt.subplot(3, 4, 9)
plt.plot(th, np.abs(cln_mag_fit), th, np.abs(cln_mag_err))

cd = np.sqrt(ds.ccovar.diagonal())
cc = ds.ccovar/cd/cd[:, np.newaxis]
plt.subplot(3, 4, 10)
plt.imshow(cc, vmin=-1, vmax=1)

plt.subplot(3, 4, 11)
plt.imshow(ds.model(cln_mag_fit, cln_mu_eff_fit).T,
           **ds_kwargs)
plt.subplot(3, 4, 12)
plt.imshow(ds.residuals(cln_mag_fit, cln_mu_eff_fit).T,
           **res_kwargs)

plt.show()
