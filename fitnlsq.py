import numpy as np
from astropy import units as u, constants as const
from astropy.table import QTable
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

from scintillometry.io import hdf5

from fields import dynamic_field


plt.ion()
quantity_support()


class DynSpecChi2:
    def __init__(self, theta, d_eff, mu_eff, dynspec, f, t, fit_mu_eff=False):
        self.theta = theta
        self.d_eff = d_eff
        self.mu_eff = mu_eff
        self.dynspec = dynspec
        self.f = f
        self.t = t
        self.fit_mu_eff = fit_mu_eff
        self._mu_eff_old = None
        self.dyn_wave = None

    def parse_pars(self, pars):
        if self.fit_mu_eff:
            realization = pars[:-1].view(complex)
            mu_eff = pars[-1] << self.mu_eff.unit
            return realization, mu_eff
        else:
            return pars.view(complex), self.mu_eff

    def residuals(self, pars):
        prt = False  # not self.fit_mu_eff
        mag, mu_eff = self.parse_pars(pars)
        if mu_eff != self.mu_eff or self.dyn_wave is None:
            self.mu_eff = mu_eff
            self.dyn_wave = dynamic_field(self.theta, 0., 1.,
                                          self.d_eff, self.mu_eff,
                                          self.f, self.t)
            prt = True
        dyn_wave_sum = (self.dyn_wave
                        * mag[:, np.newaxis, np.newaxis]).sum(0)
        dynspec_r = (dyn_wave_sum.view('2f8')**2).sum(-1)
        resids = (dynspec - dynspec_r).ravel() / noise
        if prt:
            redchi2 = (resids**2).sum()
            print(mu_eff, redchi2)
        return resids

    def dmu_eff(self, pars):
        mag, _ = self.parse_pars(pars)
        dexpphdmu = (-1j * self.d_eff/const.c
                     * u.cycle * self.f[:, np.newaxis] * self.t).to_value(
                         1./self.mu_eff.unit, u.dimensionless_angles())
        magtheta = mag * self.theta.to_value(u.radian)
        dyn_wave_sum = (self.dyn_wave * mag[:, np.newaxis, np.newaxis]).sum(0)
        ddyn_wave_sum_dmu = (
            self.dyn_wave * magtheta[:, np.newaxis, np.newaxis]
            * dexpphdmu).sum(0)
        f = dyn_wave_sum.view('2f8') * ddyn_wave_sum_dmu.view('2f8')
        return 2.*f.sum(-1)

    def jacobian(self, pars):
        assert self.dyn_wave is not None
        mag, _ = self.parse_pars(pars)
        # Use Wirtinger derivatives to calculate derivative to x, y
        magdw = (self.dyn_wave * mag[:, np.newaxis, np.newaxis]).sum(0)
        dmag = self.dyn_wave * magdw.conj()
        dmagc = self.dyn_wave.conj() * magdw
        cjacobian = np.zeros_like(dmag)
        cjacobian.real = (dmag+dmagc).real
        cjacobian.imag = (-dmag+dmagc).imag
        jacobian = (cjacobian.view('2f8').transpose(1, 2, 0, 3)
                    .reshape(self.dynspec.size, -1))
        if self.fit_mu_eff:
            dmu_eff = self.dmu_eff(pars)
            jacobian = np.concatenate(
                [jacobian, dmu_eff.reshape(dynspec.size, 1)], axis=-1)
        return -jacobian / noise


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

dyn_chi2 = DynSpecChi2(th_r, d_eff, mu_eff, dynspec, f, t, True)


guesses = recovered.view(recovered.real.dtype).ravel()
if dyn_chi2.fit_mu_eff:
    guesses = np.hstack([guesses, mu_eff.value])

sol = least_squares(dyn_chi2.residuals, guesses, dyn_chi2.jacobian, verbose=2)
