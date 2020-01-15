import numpy as np
from astropy import units as u, constants as const
from astropy.table import QTable
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import eigh
import h5py

from scintillometry.io import hdf5

from fields import dynamic_field, theta_grid, theta_theta


plt.ion()
quantity_support()


class DynamicSpectrum:
    _mu_eff_old = None

    def __init__(self, dynspec, f, t, noise, d_eff, mu_eff,
                 theta=None, magnification=None):
        self.dynspec = dynspec
        self.f = f
        self.t = t
        self.noise = noise
        self.d_eff = d_eff
        self.mu_eff = mu_eff
        self.theta = theta
        self.magnification = magnification

    @classmethod
    def fromfile(cls, filename, d_eff=None, mu_eff=None):
        with hdf5.open(filename) as fh:
            dynspec = fh.read().T
            f = fh.frequency
            t = (np.arange(-fh.shape[0] // 2, fh.shape[0] // 2)
                 / fh.sample_rate).to(u.minute)
            noise = fh.fh_raw.attrs['noise']

        self = cls(dynspec, f, t, noise, d_eff, mu_eff)
        self.filename = filename

        try:
            self.curvature = QTable.read(filename, path='curvature')
            self.theta = self.curvature.meta['theta']
            self.d_eff = self.curvature.meta['d_eff']
            self.mu_eff = self.curvature.meta['mu_eff']
        except Exception:
            pass

        return self

    def theta_grid(self, tau_max=None, oversample=1.4):
        return theta_grid(self.d_eff, self.mu_eff, self.f, self.t,
                          tau_max=tau_max, oversample=oversample)

    def dynamic_bases(self, mu_eff=None):
        if mu_eff is None:
            mu_eff = self.mu_eff

        if mu_eff != self._mu_eff_old:
            self._dyn_wave = dynamic_field(self.theta, 0., 1.,
                                           self.d_eff, mu_eff, self.f, self.t)
            self._mu_eff_old = mu_eff

        return self._dyn_wave

    def locate_mu_eff(self, mu_eff_trials=None):
        if mu_eff_trials is None:
            mu_eff_trials = np.linspace(self.mu_eff * 0.8,
                                        self.mu_eff * 1.2, 41)
        r = QTable([mu_eff_trials], names=['mu_eff'])
        r['w'] = 0.
        r['th_ms'] = 0.
        r['redchi2'] = 0.
        r['recovered'] = np.zeros((1, self.theta.size), complex)
        for i, mu_eff in enumerate(r['mu_eff']):
            th_th = theta_theta(self.theta, self.d_eff, mu_eff,
                                self.dynspec, self.f, self.t)
            w, v = eigh(th_th, eigvals=(self.theta.size-1,)*2)
            recovered = v[:, -1]

            th_ms = (np.abs((th_th - (recovered[:, np.newaxis]
                                      * recovered.conj())))**2).mean()
            dynspec_r = self.model(recovered, mu_eff=mu_eff)
            # Mean of dynamic spectra should equal sum of all recovered powers.
            # Since we normalize that to (close to) 1, just rescale similarly.
            dynspec_r *= self.dynspec.mean()/dynspec_r.mean()
            redchi2 = ((self.dynspec-dynspec_r)**2).mean() / self.noise**2
            r['redchi2'][i] = redchi2
            r['th_ms'][i] = th_ms
            r['w'][i] = w[-1]
            r['recovered'][i] = recovered
            print(f'{mu_eff} {w[-1]} {th_ms} {redchi2}')

        r.meta['theta'] = self.theta
        r.meta['d_eff'] = self.d_eff
        r.meta['mu_eff'] = r['mu_eff'][r['redchi2'].argmin()]

        self.curvature = r

        if getattr(self, 'filename', None) is not None:
            # Work-around for astropy 3.2 bug that prevents overwriting path.
            with h5py.File(self.filename, 'r+') as h5:
                h5.pop('curvature', None)
                h5.pop('curvature.__table_column_meta__', None)

            r.write(self.filename, serialize_meta=True, path='curvature',
                    append=True)
        return r

    def model(self, magnification=None, mu_eff=None):
        if magnification is None:
            magnification = self.magnification

        dyn_wave_sum = (self.dynamic_bases(mu_eff)
                        * magnification[:, np.newaxis, np.newaxis]).sum(0)
        dynspec_r = (dyn_wave_sum.view('2f8')**2).sum(-1)
        return dynspec_r

    def residuals(self, magnification=None, mu_eff=None):
        model = self.model(magnification, mu_eff)
        return (self.dynspec - model) / self.noise

    def jacobian(self, magnification, mu_eff):
        r"""Derivatives of the model dynamic spectrum to all parameters.

        The model dynamic spectrum is

        .. math::

           w(f, t) &= \sum_k \mu_k \exp[-j\phi(\theta_i, \mu_{eff}, f, t)]
                   &\equiv \sum_k \mu_k \Phi_k \\
           DS(f,t) &= |w(f, t)|^2 = w \bar{w}

        It is easiest to do the derivatives via Wirtinger derivatives:

        .. math::

           \frac{\partial}{\partial x}
           &= \left(\frac{\partial}{\partial z}
                  + \frac{\partial}{\partial\bar{z}}\right) \\
           \frac{\partial}{\partial y}
           &= i \left(\frac{\partial}{\partial z}
                     - \frac{\partial}{\partial\bar{z}}\rigt)

        Using this for the magnifications:

        .. math::

           \frac{\partial DS}{\partial\mu}
           &= \bar{w}\frac{\partial w}{\partial\mu} = \bar{w}\Phi \\
           \frac{\partial DS}{\partial\bar{\mu}}
           &= w \frac{\partial\bar{w}}{\partial\bar{\mu}} = w \bar{\Phi} \\
           \frac{\partial DS}{\partial x}
           &= \left(\bar{w} \Phi + w \bar{Phi}\right) = 2Re(w\bar{Phi}) \\
           \frac{\partial DS}{\partial y}
           &= i\left(\bar{w} \Phi - w \bar{Phi}\right) = 2Im(w\bar{Phi})

        And for :math:`mu_{eff}`:

           \frac{\partial w}{\partial\mu_{eff}}
           &= \sum_k\mu_k\Phi_k
                \frac{\partial -i\phi_k}{\partial\mu_{eff}} \\
           \frac{\partial\bar{w}}{\partial\mu_{eff}}
           &= \sum_k\bar{\mu}_k\bar{Phi}_k
               \frac{\partial i\phi_k}{\partial\mu_{eff}} \\
           \frac{\partial DS}{\partial\mu_{eff}}
           &= \bar{w}\frac{\partial w}{\partial\mu_{eff}}
            + w \frac{\partial\bar{w}}{\partial\mu_{eff}} \\
           &= 2\sum_k\mu_k Phi_k \frac{\partial -i\phi}{\partial\mu_{eff}}

        """
        dyn_wave = self.dynamic_bases(mu_eff)
        magdw = dyn_wave * magnification[:, np.newaxis, np.newaxis]
        magdw_sum = magdw.sum(0)
        jac_mag = ((2. * self._dyn_wave.conj() * magdw_sum)
                   .transpose(1, 2, 0).reshape(self.dynspec.size, -1))
        if mu_eff is not None:
            assert mu_eff == self._mu_eff_old
            dphidmu = (-1j * self.d_eff/const.c
                       * u.cycle * self.f[:, np.newaxis] * self.t
                       * self.theta[:, np.newaxis, np.newaxis]).to_value(
                           1./self.mu_eff.unit, u.dimensionless_angles())
            ddyn_wave_sum_dmu = (magdw * dphidmu).sum(0)
            dmu_eff = 2.*(magdw_sum.view('2f8')
                          * ddyn_wave_sum_dmu.view('2f8')).sum(-1)
            jac_mu = dmu_eff.reshape(self.dynspec.size, 1)

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
            print(self.mu_eff, ((self.dynspec-model)**2).sum()/self.noise**2)

        return model.ravel()

    def _jacobian(self, unused_x_data, *pars):
        magnification, mu_eff = self._separate_pars(pars)
        return self.jacobian(magnification, mu_eff)

    def fit(self, guess=None, verbose=2, **kwargs):
        if guess is None:
            mu_eff_guess = self.mu_eff
            mag_guess = self.magnification
        elif guess == 'curvature':
            ibest = self.curvature['redchi2'].argmin()
            mu_eff_guess = self.curvature['mu_eff'][ibest]
            mag_guess = self.curvature['recovered'][ibest]
        else:
            mag_guess, mu_eff_guess = guess

        self._prt = verbose > 2
        if mu_eff_guess is not None:
            mu_eff_guess = mu_eff_guess.to_value(self.mu_eff.unit)

        guesses = self._combine_pars(mag_guess, mu_eff_guess)
        # Default method of 'lm' does not give reliable error estimates.
        kwargs.setdefault('method', 'trf')
        if kwargs['method'] != 'lm':
            kwargs['verbose'] = min(verbose, 2)
        # Note: for typical runs, times dominated by svd decompositions
        # inside fitting routines.
        popt, pcovar = curve_fit(
            self._model, xdata=None, ydata=self.dynspec.ravel(),
            p0=guesses, sigma=np.broadcast_to(self.noise, self.dynspec.size),
            jac=self._jacobian, **kwargs)
        mag_fit, mu_eff_fit = self._separate_pars(popt)
        mag_err, mu_eff_err = self._separate_pars(np.sqrt(np.diag(pcovar)))
        return mag_fit, mag_err, mu_eff_fit, mu_eff_err


dyn_chi2 = DynamicSpectrum.fromfile('dynspec.h5', d_eff=1.*u.kpc,
                                    mu_eff=100*u.mas/u.yr)
dyn_chi2.theta = dyn_chi2.theta_grid(
    tau_max=(1./(dyn_chi2.f[3]-dyn_chi2.f[0])).to(u.us))
if not hasattr(dyn_chi2, 'curvature'):
    dyn_chi2.locate_mu_eff()

r = dyn_chi2.curvature

plt.clf()
plt.subplot(121)
plt.plot(r['mu_eff'], r['redchi2'])

mag_fit, mag_err, mu_eff_fit, mu_eff_err = dyn_chi2.fit(guess='curvature')
dyn_chi2.magnification = mag_fit
dyn_chi2.mu_eff = mu_eff_fit
plt.subplot(122)
plt.imshow(dyn_chi2.residuals(), aspect='auto', origin=0, vmin=-5, vmax=5)
