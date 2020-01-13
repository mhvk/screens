import numpy as np
import scipy.linalg
from astropy import units as u
from astropy.table import QTable
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt
import h5py

from scintillometry.io import hdf5

from fields import dynamic_field, theta_theta, theta_grid


with hdf5.open('dynspec.h5') as fh:
    dynspec = fh.read().T
    f = fh.frequency
    fobs = f[f.shape[0]//2]
    t = (np.arange(-fh.shape[0] // 2, fh.shape[0] // 2)
         / fh.sample_rate).to(u.minute)
    realization = fh.fh_raw['realization'][:]
    th = fh.fh_raw['theta'][:] << u.mas
    noise = fh.fh_raw.attrs['noise']


d_eff = 1 * u.kpc
mu_eff = 100 * u.mas/u.yr

# Make a grid that steps roughly with dtau at large theta.
# Assue we're not too close to max tau in secondary spectrum.
tau_max = (1./(f[3]-f[0])).to(u.us)
th_r = theta_grid(d_eff, mu_eff, f, t, tau_max=tau_max)

r = QTable([np.linspace(90, 110, 41) << u.mas/u.yr], names=['mu_eff'])
r['w'] = 0.
r['th_ms'] = 0.
r['redchi2'] = 0.
r['recovered'] = np.zeros((1, th_r.size), complex)
r.meta['theta'] = th_r
for i, mu_eff in enumerate(r['mu_eff']):
    th_th = theta_theta(th_r, d_eff, mu_eff, dynspec, f, t)
    w, v = scipy.linalg.eigh(th_th, eigvals=(th_r.size-1, th_r.size-1))
    recovered = v[:, -1]

    th_ms = (np.abs((th_th - (recovered[:, np.newaxis]
                              * recovered.conj())))**2).mean()
    dynwave_r = dynamic_field(th_r, 0, recovered, d_eff, mu_eff, f, t)
    dynspec_r = np.abs(dynwave_r.sum(0)) ** 2
    # Mean of dynamic spectra should equal sum of all recovered powers.
    # Since we normalize that to (close to) 1, just rescale similarly here.
    dynspec_r *= dynspec.mean()/dynspec_r.mean()
    redchi2 = ((dynspec-dynspec_r)**2).mean() / noise**2
    r['redchi2'][i] = redchi2
    r['th_ms'][i] = th_ms
    r['w'][i] = w[-1]
    r['recovered'][i] = recovered
    print(f'{mu_eff} {w[-1]} {th_ms} {redchi2}')

# Work-around for astropy 3.2 bug that does not allow overwriting path.
with h5py.File('dynspec.h5', 'r+') as h5:
    h5.pop('curvature', None)
    h5.pop('curvature.__table_column_meta__', None)

r.write('dynspec.h5', serialize_meta=True, path='curvature', append=True)


plt.ion()
quantity_support()
plt.clf()
plt.plot(r['mu_eff'], r['redchi2'])
