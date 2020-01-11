import numpy as np
from astropy import units as u
from astropy.table import QTable
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt
import h5py

from scintillometry.io import hdf5

from fields import dynamic_field, theta_theta, clean_theta_theta


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

r = QTable([np.linspace(70, 130, 61) << u.mas/u.yr], names=['mu_eff'])
r['w'] = 0.
r['th_ms'] = 0.
r['redchi2'] = 0.
r['recovered'] = np.zeros((1, th.size), complex)
for i, mu_eff in enumerate(r['mu_eff']):
    th_th = theta_theta(th, d_eff, mu_eff, dynspec, f, t)
    th_th = clean_theta_theta(th_th, k=1)
    w, v = np.linalg.eigh(th_th)
    recovered = v[:, -1]

    th_ms = (np.abs((th_th - (recovered[:, np.newaxis]
                              * recovered.conj())))**2).mean()
    dynwave_r = dynamic_field(th, 0, recovered, d_eff, mu_eff, f, t)
    dynspec_r = np.abs(dynwave_r.sum(0)) ** 2
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
