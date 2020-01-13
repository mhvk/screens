import numpy as np
from astropy import units as u
from astropy.table import QTable
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from scintillometry.io import hdf5

from fields import dynamic_field


plt.ion()
quantity_support()


class DynSpecChi2:
    def __init__(self, theta, d_eff, mu_eff, dynspec, f, t):
        self.theta = theta
        self.d_eff = d_eff
        self.mu_eff = mu_eff
        self.dynspec = dynspec
        self.f = f
        self.t = t
        self._mu_eff_old = None
        self.dyn_wave = None

    def __call__(self, pars):
        prt = False
        realization = pars[:-1].view(complex)
        mu_eff = pars[-1] << self.mu_eff.unit
        # realization = pars.view(complex)
        # mu_eff = self.mu_eff
        if mu_eff != self.mu_eff or self.dyn_wave is None:
            self.mu_eff = mu_eff
            self.dyn_wave = dynamic_field(self.theta, 0., 1.,
                                          self.d_eff, self.mu_eff,
                                          self.f, self.t)
            prt = True
        dyn_wave = self.dyn_wave * realization[:, np.newaxis, np.newaxis]
        dynspec_r = np.abs(dyn_wave.sum(0))**2
        redchi2 = ((dynspec-dynspec_r)**2).mean() / noise**2
        if prt:
            print(mu_eff, redchi2)
        return redchi2


with hdf5.open('dynspec.h5') as fh:
    dynspec = fh.read().T
    f = fh.frequency
    fobs = f[f.shape[0]//2]
    t = (np.arange(-fh.shape[0] // 2, fh.shape[0] // 2)
         / fh.sample_rate).to(u.minute)
    realization = fh.fh_raw['realization'][:]
    th = fh.fh_raw['theta'][:] << u.mas
    noise = fh.fh_raw.attrs['noise']


r = QTable.read('dynspec.h5', path='curvature')
th_r = r.meta['theta']

d_eff = 1.*u.kpc

ibest = r['redchi2'].argmin()
mu_eff = r['mu_eff'][ibest]
recovered = r['recovered'][ibest]

guesses = np.hstack([recovered.view(recovered.real.dtype).ravel(),
                     mu_eff.value])

dyn_chi2 = DynSpecChi2(th_r, d_eff, mu_eff, dynspec, f, t)

sol = minimize(dyn_chi2, guesses)
