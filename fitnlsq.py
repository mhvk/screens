import numpy as np
from astropy import units as u, constants as const
from astropy.table import QTable
from astropy.utils import lazyproperty
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from scintillometry.io import hdf5

from fields import dynamic_field


plt.ion()
quantity_support()


class DynSpecChi2:
    def __init__(self, theta, d_eff, mu_eff, dynspec, f, t, noise):
        self.theta = theta
        self.d_eff = d_eff
        self.mu_eff = mu_eff
        self.dynspec = dynspec
        self.f = f
        self.t = t
        self.noise = noise

    @lazyproperty
    def dyn_wave(self):
        return dynamic_field(self.theta, 0., 1., self.d_eff, self.mu_eff,
                             self.f, self.t)

    def model(self, magnification, mu_eff):
        if mu_eff is not None and mu_eff != self.mu_eff:
            self.mu_eff = mu_eff
            # Force recalculation by next access.
            del self.dyn_wave

        dyn_wave_sum = (self.dyn_wave
                        * magnification[:, np.newaxis, np.newaxis]).sum(0)
        dynspec_r = (dyn_wave_sum.view('2f8')**2).sum(-1)
        return dynspec_r

    def residuals(self, magnification, mu_eff):
        model = self.model(magnification, mu_eff)
        return (dynspec - model) / self.noise

    def jacobian(self, magnification, mu_eff):
        # Use Wirtinger derivatives to calculate derivative to x, y
        magdw = (self.dyn_wave
                 * magnification[:, np.newaxis, np.newaxis]).sum(0)
        jac_mag = ((2. * self.dyn_wave.conj() * magdw)
                   .transpose(1, 2, 0).reshape(self.dynspec.size, -1))
        if mu_eff is not None:
            dexpphdmu = (-1j * self.d_eff/const.c
                         * u.cycle * self.f[:, np.newaxis] * self.t).to_value(
                             1./self.mu_eff.unit, u.dimensionless_angles())
            magtheta = magnification * self.theta.to_value(u.radian)
            ddyn_wave_sum_dmu = (
                self.dyn_wave * magtheta[:, np.newaxis, np.newaxis]
                * dexpphdmu).sum(0)
            dmu_eff = 2.*(magdw.view('2f8')
                          * ddyn_wave_sum_dmu.view('2f8')).sum(-1)
            jac_mu = dmu_eff.reshape(dynspec.size, 1)

        else:
            jac_mu = None

        jacobian = self._combine_pars(jac_mag, jac_mu)
        return jacobian

    def _combine_pars(self, magnification, mu_eff):
        pars = (magnification.view('2f8')
                .reshape(magnification.shape[:-1]+(-1,)))
        if mu_eff is not None:
            mu_eff_par = np.atleast_1d(mu_eff)
            pars = np.concatenate([pars, mu_eff_par], axis=-1)

        return pars

    def _separate_pars(self, pars):
        pars = np.asanyarray(pars)
        if len(pars) > 2*len(self.theta):
            magnification = pars[:-1].view(complex)
            mu_eff = pars[-1] << self.mu_eff.unit
            return magnification, mu_eff
        else:
            return pars.view(complex), None

    def _model(self, unused_x_data, *pars):
        magnification, mu_eff = self._separate_pars(pars)
        old_mu_eff = self.mu_eff
        model = self.model(magnification, mu_eff)
        if self._prt and old_mu_eff != self.mu_eff:
            print(self.mu_eff, ((self.dynspec-model)**2).sum()/noise**2)

        return model.ravel()

    def _jacobian(self, unused_x_data, *pars):
        magnification, mu_eff = self._separate_pars(pars)
        return self.jacobian(magnification, mu_eff)

    def fit(self, mag_guesses, mu_eff_guess=None, verbose=2, **kwargs):
        self._prt = verbose > 2
        if mu_eff_guess is not None:
            mu_eff_guess = mu_eff_guess.to_value(self.mu_eff.unit)
        guesses = self._combine_pars(mag_guesses, mu_eff_guess)
        # Default method of 'lm' does not give reliable error estimates.
        kwargs.setdefault('method', 'trf')
        if kwargs['method'] != 'lm':
            kwargs['verbose'] = min(verbose, 2)
        self.popt, self.pcovar = curve_fit(
            self._model, xdata=None, ydata=self.dynspec.ravel(),
            p0=guesses, sigma=np.broadcast_to(noise, self.dynspec.size),
            jac=self._jacobian, **kwargs)
        magnifications, mu_eff = self._separate_pars(self.popt)
        sig_mag, sig_mu_eff = self._separate_pars(np.sqrt(np.diag(self.pcovar)))
        return magnifications, sig_mag, mu_eff, sig_mu_eff


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

dyn_chi2 = DynSpecChi2(th_r, d_eff, mu_eff, dynspec, f, t, noise)

mag_fit, mag_err, mu_fit, mu_err = dyn_chi2.fit(recovered, mu_eff)
