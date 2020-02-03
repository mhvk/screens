# Licensed under the GPLv3 - see LICENSE
"""Do eigenvalue decomposition on a simulated secondary spectrum.

This presumes a dynamic spectrum has been generated using screen2ds.
It does *not* use the theta information from that file, but does
assume d_eff and mu_eff are already known.

TODO: actually try to locate mu_eff.
"""
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from astropy.visualization import quantity_support
from scipy.linalg import eigh

from screens.visualization import ThetaTheta
from screens import DynamicSpectrum, SecondarySpectrum


plt.ion()
plt.clf()
quantity_support()

dyn_spec = DynamicSpectrum.fromfile('dynspec.h5', d_eff=1.*u.kpc,
                                    mu_eff=100*u.mas/u.yr)
sec_spec = SecondarySpectrum.from_dynamic_spectrum(dyn_spec)

sec_kwargs = dict(extent=(sec_spec.fd[0].value, sec_spec.fd[-1].value,
                          sec_spec.tau[0].value, sec_spec.tau[-1].value),
                  cmap='Greys', vmin=-7, vmax=0, origin=0, aspect='auto')
plt.subplot(321)
plt.imshow(np.log10(np.abs(sec_spec.secspec)**2).T, **sec_kwargs)

conserve = True

mu_eff = 100*u.mas/u.yr
th_th = sec_spec.theta_theta(mu_eff, conserve=conserve)

th_kwargs = sec_kwargs.copy()
th_kwargs['extent'] = (sec_spec.theta[0].value,
                       sec_spec.theta[-1].value)*2
th_kwargs['aspect'] = 'equal'
th_th_proj = ThetaTheta(sec_spec.theta)
ax = plt.subplot(322, projection=th_th_proj)
ax.imshow(np.log10(np.maximum(np.abs(th_th)**2, 1e-30)).T, **th_kwargs)

secspec = sec_spec.secspec.copy()
secspec[(sec_spec.fd == 0) | (sec_spec.tau == 0)] = 0
# try recoving just plain power
w_a, v_a = eigh(np.abs(th_th)**2, eigvals=(sec_spec.theta.size-1,)*2)
rec_a = v_a[:, -1] * np.sqrt(w_a[-1])
th_th_rp = rec_a[:, np.newaxis] * rec_a
ax = plt.subplot(324, projection=th_th_proj)
ax.imshow(np.log10(np.maximum(th_th_rp, 1e-30)).T, **th_kwargs)

sec_rp = sec_spec.model(rec_a, mu_eff, conserve=conserve)
sec_p = np.abs(secspec)**2
sec_p_noise = sec_p[:30, :30].std()
print("just power, red chi2 = ",
      ((np.abs(secspec)**2-sec_rp)**2).mean() / sec_p_noise**2)
plt.subplot(323)
plt.imshow(np.log10(np.maximum(sec_rp, 1e-30)).T, **sec_kwargs)

# try recovering phases as well.
w, v = eigh(th_th, eigvals=(sec_spec.theta.size-1,)*2)
recovered = v[:, -1] * np.sqrt(w[-1])
th_th_r = recovered[:, np.newaxis] * recovered
ax = plt.subplot(326, projection=th_th_proj)
ax.imshow(np.log10(np.maximum(np.abs(th_th_r)**2, 1e-30)).T, **th_kwargs)

sec_r = sec_spec.model(recovered, mu_eff, conserve=conserve)
sec_noise = secspec[:30, :30].std()
print("also phase, red chi2 = ",
      (np.abs(secspec-sec_r)**2).mean() / sec_noise**2)
print(" power red chi2 = ",
      ((np.abs(secspec)**2-np.abs(sec_r)**2)**2).mean() / sec_p_noise**2)
plt.subplot(325)
plt.imshow(np.log10(np.maximum(np.abs(sec_r)**2, 1e-30)).T, **sec_kwargs)

# Try via dynspec.  Generically, this seems to be worse, especially with
# flux conservation turned on.
dyn_spec.theta = sec_spec.theta
ds_model = dyn_spec.model(recovered)
sec_dr = np.fft.fft2(ds_model)
sec_dr /= sec_dr[0, 0]
sec_dr = np.fft.fftshift(sec_dr)
sec_dr[(sec_spec.fd == 0) | (sec_spec.tau == 0)] = 0
print("via dyn spec, red chi2 = ",
      (np.abs(secspec-sec_dr)**2).mean() / sec_noise**2)
