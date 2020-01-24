import numpy as np
from astropy import units as u, constants as const
from astropy.table import QTable
from scipy.optimize import curve_fit
from scipy.linalg import eigh
import h5py

from scintillometry.io import hdf5

from fields import dynamic_field, theta_theta, theta_grid


class DynamicSpectrum:
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
            dynspec = fh.read()
            f = fh.frequency
            t = (np.arange(-fh.shape[0] // 2, fh.shape[0] // 2)
                 / fh.sample_rate).to(u.minute)[:, np.newaxis]
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

    def theta_grid(self, oversample_tau=1.3, oversample_fd=1.3**2, **kwargs):
        kwargs.setdefault('d_eff', self.d_eff)
        kwargs.setdefault('mu_eff', self.mu_eff)
        kwargs.setdefault('fobs', self.f.mean())
        kwargs.setdefault('dtau', (1./self.f.ptp()).to(u.us)
                          / oversample_tau)
        kwargs.setdefault('dfd', (1./self.t.ptp()).to(u.mHz)
                          / oversample_fd)
        kwargs.setdefault('tau_max', 1/(self.f[2]-self.f[0]))
        kwargs.setdefault('fd_max', 1/(self.t[2]-self.t[0]))
        return theta_grid(**kwargs)

    def dynamic_bases(self, mu_eff=None):
        if mu_eff is None:
            mu_eff = self.mu_eff

        if mu_eff != getattr(self, '_mu_eff_old', None):
            self._dyn_wave = dynamic_field(self.theta, 0., 1.,
                                           self.d_eff, mu_eff, self.f, self.t)
            self._mu_eff_old = mu_eff

        return self._dyn_wave

    def theta_theta(self, mu_eff=None):
        if mu_eff is None:
            mu_eff = self.mu_eff

        return theta_theta(self.theta, self.d_eff, mu_eff,
                           self.dynspec, self.f, self.t)

    def locate_mu_eff(self, mu_eff_trials=None, verbose=True):
        if mu_eff_trials is None:
            mu_eff_trials = np.linspace(self.mu_eff * 0.8,
                                        self.mu_eff * 1.2, 21)
        r = QTable([mu_eff_trials], names=['mu_eff'])
        r['w'] = 0.
        r['th_ms'] = 0.
        r['redchi2'] = 0.
        r['recovered'] = np.zeros((1,)+self.theta.shape, complex)
        for i, mu_eff in enumerate(r['mu_eff']):
            th_th = self.theta_theta(mu_eff)
            w, v = eigh(th_th, eigvals=(self.theta.size-1,)*2)
            recovered = v[:, -1]
            recovered0 = recovered[self.theta == 0]

            recovered *= recovered0.conj()/np.abs(recovered0)

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
            if verbose:
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
                        * magnification[..., np.newaxis, np.newaxis]).sum(-3)
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

        It is easiest to use Wirtinger derivatives:

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
           &= \bar{w} \frac{\partial w}{\partial\mu_{eff}}
            +  w \frac{\partial\bar{w}}{\partial\mu_{eff}} \\
           &= 2\sum_k\mu_k Phi_k \frac{\partial -i\phi}{\partial\mu_{eff}}

        """
        dyn_wave = self.dynamic_bases(mu_eff)
        magdw = dyn_wave * magnification[:, np.newaxis, np.newaxis]
        magdw_sum = magdw.sum(0)
        jac_mag = ((2. * self._dyn_wave.conj() * magdw_sum)
                   .transpose(1, 2, 0).reshape(self.dynspec.size, -1)
                   .view([('mag_real', 'f8'), ('mag_imag', 'f8')]))
        if mu_eff is not None:
            dphidmu = (-1j * self.d_eff/const.c
                       * u.cycle * self.f * self.t
                       * self.theta[:, np.newaxis, np.newaxis]).to(
                           1./self.mu_eff.unit, u.dimensionless_angles())
            ddyn_wave_sum_dmu = (magdw * dphidmu).sum(0)
            dmu_eff = 2.*(magdw_sum.view('2f8')
                          * ddyn_wave_sum_dmu.view('2f8')).sum(-1)
            jac_mu = dmu_eff.reshape(self.dynspec.size, 1)

        else:
            jac_mu = None

        return jac_mag, jac_mu

    def _combine_pars(self, magnification, mu_eff, mu_eff_scale):
        msh = magnification.shape[:-1]
        magnification = magnification.view('2f8')
        theta0 = self.theta == 0
        mag0real = magnification[..., theta0, 0].reshape(msh+(-1,))
        others = magnification[..., ~theta0, :].reshape(msh+(-1,))
        if mu_eff is not None:
            mu_eff_par = mu_eff.to_value(mu_eff_scale).reshape(msh+(1,))
            pars = np.concatenate([others, mag0real, mu_eff_par], axis=-1)
        else:
            pars = np.concatenate([others, mag0real], axis=-1)

        return pars

    def _separate_pars(self, pars, mu_eff_scale=None):
        pars = np.asanyarray(pars)
        if len(pars) > 2*len(self.theta)-1:
            if mu_eff_scale is None:
                mu_eff_scale = self.mu_eff_guess
            mu_eff = pars[-1] * mu_eff_scale
            pars = pars[:-1]
        else:
            mu_eff = None

        amp0 = pars[-1]
        others = pars[:-1].view(complex)
        magnification = np.zeros(self.theta.shape, complex)
        theta0 = self.theta == 0
        magnification[theta0] = amp0
        magnification[~theta0] = others
        return magnification, mu_eff

    def _separate_covar(self, pcovar, mu_eff_scale=None):
        """Complex covariance and pseudo-covariance.

        Of magnitudes with themselves and magnitudes with mu_eff

        Note: not quite sure this is done correctly (or useful)!
        https://en.wikipedia.org/wiki/Complex_random_vector#Covariance_matrix_and_pseudo-covariance_matrix
        """
        if len(pcovar) > 2*len(self.theta)-1:
            if mu_eff_scale is None:
                mu_eff_scale = self.mu_eff_guess
            mu_eff_cov, mu_eff_var = self._separate_pars(
                pcovar[-1], mu_eff_scale**2)
            pcovar = pcovar[:-1, :-1] * mu_eff_scale
        else:
            mu_eff_cov = mu_eff_var = None

        r0 = pcovar[-1:, :-1]
        r0 = np.concatenate([r0, np.zeros_like(r0)], axis=0)
        c0 = pcovar[:-1, -1:]
        c0 = np.concatenate([c0, np.zeros_like(c0)], axis=1)
        others = pcovar[:-1, :-1]
        var0 = np.array([[pcovar[-1, -1], 0.],
                         [0., 0.]])
        i0 = np.argmax(self.theta == 0) * 2
        mag_cov_reim = np.block(
            [[others[:i0, :i0], c0[:i0], others[:i0, i0:]],
             [r0[:, :i0], var0, r0[:, i0:]],
             [others[i0:, :i0], c0[i0:], others[i0:, i0:]]])
        mag_cov_xx = mag_cov_reim[0::2, 0::2]
        mag_cov_yx = mag_cov_reim[1::2, 0::2]
        mag_cov_xy = mag_cov_reim[0::2, 1::2]
        mag_cov_yy = mag_cov_reim[1::2, 1::2]
        mag_covar = (mag_cov_xx + mag_cov_yy
                     + 1j*(mag_cov_yx - mag_cov_xy))
        mag_pseudo = (mag_cov_xx - mag_cov_yy
                      + 1j*(mag_cov_yx + mag_cov_xy))

        return mag_covar, mag_pseudo, mu_eff_cov, mu_eff_var

    def _model(self, unused_x_data, *pars):
        magnification, mu_eff = self._separate_pars(pars)
        model = self.model(magnification, mu_eff)
        if self._prt:
            print(mu_eff if mu_eff is not None else self.mu_eff,
                  ((self.dynspec-model)**2).sum()/self.noise**2)

        return model.ravel()

    def _jacobian(self, unused_x_data, *pars):
        magnification, mu_eff = self._separate_pars(pars)
        jac_mag, jac_mu_eff = self.jacobian(magnification, mu_eff)
        return self._combine_pars(jac_mag, jac_mu_eff,
                                  1/self.mu_eff_guess)

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
        self.mag_guess = mag_guess
        self.mu_eff_guess = mu_eff_guess
        guesses = self._combine_pars(mag_guess, mu_eff_guess,
                                     self.mu_eff_guess)
        # Default method of 'lm' does not give reliable error estimates.
        kwargs.setdefault('method', 'trf')
        if kwargs['method'] != 'lm':
            kwargs['verbose'] = min(verbose, 2)
            kwargs['x_scale'] = 'jac'
        # Note: typically, run-time is dominated by svd decompositions
        # inside fitting routines.
        self.popt, self.pcovar = curve_fit(
            self._model, xdata=None, ydata=self.dynspec.ravel(),
            p0=guesses, sigma=np.broadcast_to(self.noise, self.dynspec.size),
            jac=self._jacobian, **kwargs)

        self.raw_mag_fit, self.raw_mu_eff_fit = self._separate_pars(self.popt)
        perr = np.sqrt(np.diag(self.pcovar))
        self.raw_mag_err, self.raw_mu_eff_err = self._separate_pars(perr)
        return (self.raw_mag_fit, self.raw_mag_err,
                self.raw_mu_eff_fit, self.raw_mu_eff_err)

    def cleanup_fit(self, max_abs_err=0.1, max_rel_err=1000., verbose=True):
        jm = self._jacobian(None, *self.popt)
        # Get bases for jacobian using SVD
        _, jms, jmvt = np.linalg.svd(jm, full_matrices=False)
        # Project optimal parameters on SVD bases
        pproj = jmvt.conj() @ self.popt
        # Remove basis which carry very little information.
        # For this purpose, magnifications cannot physically be
        # more than 1, and mu_eff is also scaled to be around 1,
        # so remove all that have inferred error = 1/jms > max_err.
        jms /= self.noise
        max_err = np.minimum(max_abs_err, max_rel_err/jms[0])
        ok = jms > 1./max_err
        self.copt = jmvt[ok].T @ pproj[ok]
        self.ccovar = (jmvt[ok].T / jms[ok]**2) @ jmvt[ok]
        self.cln_mag_fit, self.cln_mu_eff_fit = self._separate_pars(self.copt)
        cerr = np.sqrt(np.diag(self.ccovar))
        self.cln_mag_err, self.cln_mu_eff_err = self._separate_pars(cerr)
        if verbose:
            msg = "{} correction, mu_eff={:+10.4f}±{:6.4f} mas/yr, chi2={}"
            print(msg.format("Before",
                             self.raw_mu_eff_fit.to_value(u.mas/u.yr),
                             self.raw_mu_eff_err.to_value(u.mas/u.yr),
                             (self.residuals(self.raw_mag_fit,
                                             self.raw_mu_eff_fit)**2).sum()))
            print(msg.format("After",
                             self.cln_mu_eff_fit.to_value(u.mas/u.yr),
                             self.cln_mu_eff_err.to_value(u.mas/u.yr),
                             (self.residuals(self.cln_mag_fit,
                                             self.cln_mu_eff_fit)**2).sum()))

        return (self.cln_mag_fit, self.cln_mag_err,
                self.cln_mu_eff_fit, self.cln_mu_eff_err)
