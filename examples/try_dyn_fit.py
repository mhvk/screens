# Licensed under the GPLv3 - see LICENSE
"""Fit a simulated dynamic spectrum directly using a 1D screen.

This presumes a dynamic spectrum has been generated using screen2ds.
It does *not* use the theta information from that file, but does
assume approximate knowledge of d_eff and mu_eff (only fitting for
the latter).
"""

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from astropy.visualization import quantity_support

from screens.visualization import ThetaTheta
from screens import DynamicSpectrum


plt.ion()
quantity_support()

dyn_chi2 = DynamicSpectrum.fromfile('dynspec.h5', d_eff=1.*u.kpc,
                                    mu_eff=100*u.mas/u.yr)
dyn_chi2.theta = dyn_chi2.theta_grid(
    tau_max=(1./(dyn_chi2.f[3]-dyn_chi2.f[0])).to(u.us))
dyn_chi2.locate_mu_eff(np.arange(98, 103) << u.mas/u.yr)

r = dyn_chi2.curvature

plt.clf()
plt.subplot(3, 4, 1)
plt.plot(r['mu_eff'], r['redchi2'])

th_th = dyn_chi2.theta_theta()
th = dyn_chi2.theta
th_kwargs = dict(extent=(th[0].value, th[-1].value)*2,
                 origin=0, vmin=-7, vmax=0, cmap='Greys')
th_th_proj = ThetaTheta(dyn_chi2.theta)
plt.subplot(3, 4, 2, projection=th_th_proj)
plt.imshow(np.log10(np.maximum(np.abs(th_th)**2, 1e-30)), **th_kwargs)
plt.xlabel(th.unit.to_string('latex'))
plt.ylabel(th.unit.to_string('latex'))

ds_kwargs = dict(extent=(dyn_chi2.t[0].value, dyn_chi2.t[-1].value,
                         dyn_chi2.f[0].value, dyn_chi2.f[-1].value),
                 origin=0, aspect='auto', cmap='Greys',
                 vmin=0, vmax=7)
ibest = r['redchi2'].argmin()
dynspec_r = dyn_chi2.model(r['recovered'][ibest],
                           mu_eff=r['mu_eff'][ibest])
dynspec_r *= dyn_chi2.dynspec.mean()/dynspec_r.mean()
plt.subplot(3, 4, 3)
plt.imshow(dynspec_r.T, **ds_kwargs)

res_kwargs = ds_kwargs.copy()
res_kwargs['vmin'], res_kwargs['vmax'] = -5, 5
plt.subplot(3, 4, 4)
plt.imshow((dyn_chi2.dynspec-dynspec_r).T/dyn_chi2.noise, **res_kwargs)

plt.subplot(3, 4, 7)
plt.imshow(dyn_chi2.dynspec.T, **ds_kwargs)

(raw_mag_fit, raw_mag_err,
 raw_mu_eff_fit, raw_mu_eff_err) = dyn_chi2.fit(
     guess='curvature', verbose=3,
     ftol=0.1/dyn_chi2.dynspec.size)

plt.subplot(3, 4, 5)
plt.plot(th, np.abs(raw_mag_fit), th, np.abs(raw_mag_err))

rd = np.sqrt(dyn_chi2.pcovar.diagonal())
rc = dyn_chi2.pcovar/rd/rd[:, np.newaxis]
plt.subplot(3, 4, 6)
plt.imshow(rc, vmin=-1, vmax=1)

plt.subplot(3, 4, 8)
plt.imshow(dyn_chi2.residuals(raw_mag_fit, raw_mu_eff_fit).T,
           **res_kwargs)

(cln_mag_fit, cln_mag_err,
 cln_mu_eff_fit, cln_mu_eff_err) = dyn_chi2.cleanup_fit()

plt.subplot(3, 4, 9)
plt.plot(th, np.abs(cln_mag_fit), th, np.abs(cln_mag_err))

cd = np.sqrt(dyn_chi2.ccovar.diagonal())
cc = dyn_chi2.ccovar/cd/cd[:, np.newaxis]
plt.subplot(3, 4, 10)
plt.imshow(cc, vmin=-1, vmax=1)

plt.subplot(3, 4, 11)
plt.imshow(dyn_chi2.model(cln_mag_fit, cln_mu_eff_fit).T,
           **ds_kwargs)
plt.subplot(3, 4, 12)
plt.imshow(dyn_chi2.residuals(cln_mag_fit, cln_mu_eff_fit).T,
           **res_kwargs)
