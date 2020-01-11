import numpy as np
from astropy import units as u
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt

from scintillometry.io import hdf5

from fields import dynamic_field, theta_theta, clean_theta_theta


plt.ion()
quantity_support()
plt.clf()


with hdf5.open('dynspec.h5') as fh:
    dynspec = fh.read().T
    f = fh.frequency
    fobs = f[f.shape[0]//2]
    t = (np.arange(-fh.shape[0] // 2, fh.shape[0] // 2)
         / fh.sample_rate).to(u.minute)
    realization = fh.fh_raw['realization'][:]
    th = fh.fh_raw['theta'][:] << u.mas
    noise = fh.fh_raw.attrs['noise']

# Display simulated dynamic spectrum.
ds_kwargs = dict(extent=(t[0].value, t[-1].value, f[0].value, f[-1].value),
                 origin=0, aspect='auto', cmap='Greys', vmin=0, vmax=6)
plt.subplot(321)
plt.imshow(dynspec, **ds_kwargs)
plt.xlabel(t.unit.to_string('latex'))
plt.ylabel(f.unit.to_string('latex'))
plt.colorbar()

d_eff = 1 * u.kpc
mu_eff = 100 * u.mas / u.yr

th_th = theta_theta(th, d_eff, mu_eff, dynspec, f, t)

# Clean up near the diagonal
th_th = clean_theta_theta(th_th, k=1, clean_cross=True)

# Show inferred theta-theta.
th_kwargs = dict(extent=(th[0].value, th[-1].value)*2,
                 origin=0, vmin=-7, vmax=0, cmap='Greys')
plt.subplot(322)
plt.imshow(np.log10(np.maximum(np.abs(th_th)**2, 1e-30)), **th_kwargs)
plt.xlabel(th.unit.to_string('latex'))
plt.ylabel(th.unit.to_string('latex'))


# Calculate eigenvectors for inferred theta-theta.

w, v = np.linalg.eigh(th_th)

assert w[-1] == w.max()

# Ideally, the eigenvalue is 1, but we want a normalized solution anyway,
# so just use properly normalized eigenvector.
recovered = v[:, -1]

# Show the theta-theta implied by largest eigenvector.
plt.subplot(324)
plt.imshow(np.log10(np.abs(np.outer(recovered, recovered.conj())**2)),
           **th_kwargs)
plt.xlabel(th.unit.to_string('latex'))
plt.ylabel(th.unit.to_string('latex'))

# As well as the corresponding dynamic spetrum.
plt.subplot(323)
dynwave_r = dynamic_field(th, 0, recovered, d_eff, mu_eff, f, t)
dynspec_r = np.maximum(np.abs(dynwave_r.sum(0)) ** 2, 1e-30)

# Mean of dynamic spectra should equal sum of all recovered powers.
# Since we normalize that to (close to) 1, just rescale similarly here.
dynspec_r *= dynspec.mean()/dynspec_r.mean()
plt.imshow(dynspec_r, **ds_kwargs)
plt.xlabel(t.unit.to_string('latex'))
plt.ylabel(f.unit.to_string('latex'))
plt.colorbar()

print('Recovered red. chi2 ', ((dynspec-dynspec_r)**2).mean() / noise**2)

# Also show theta-theta corresponding to actual realization.
plt.subplot(326)
th_th_real = np.outer(realization, realization.conj())
plt.imshow(np.log10(np.abs(th_th_real**2)), **th_kwargs)
plt.xlabel(th.unit.to_string('latex'))
plt.ylabel(th.unit.to_string('latex'))

w_real, v_real = np.linalg.eigh(th_th_real)

assert w_real[-1] == w_real.max()

recovered_real = v_real[:, -1]

# and the dynamic spectrum implied by its largest eigenvector.
plt.subplot(325)
dynwave_real = dynamic_field(th, 0, recovered_real, d_eff, mu_eff, f, t)
dynspec_real = np.maximum(np.abs(dynwave_real.sum(0)) ** 2, 1e-30)
plt.imshow(dynspec_real, **ds_kwargs)
plt.xlabel(t.unit.to_string('latex'))
plt.ylabel(f.unit.to_string('latex'))
plt.colorbar()

print('Recovered red. chi2 ', ((dynspec-dynspec_real)**2).mean() / noise**2)
