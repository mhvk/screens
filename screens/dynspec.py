# Licensed under the GPLv3 - see LICENSE
import numpy as np
from astropy import units as u, constants as const
from astropy.table import QTable
from scipy.optimize import curve_fit
from numpy.linalg import eigh

from .fields import dynamic_field, theta_grid, theta_theta_indices


__all__ = ['DynamicSpectrum']


class DynamicSpectrum:
    """Dynamic spectrum and methods to fit it.

    The code is meant to be agnostic to which axes are which, but some may
    assume a shape of ``(..., time_axis, frequency_axis)``.

    Parameters
    ----------
    dynspec : `~numpy.ndarray`
        Intensities as a function of time and frequency.
    t : `~astropy.units.Quantity`
        Times of the dynamic spectrum.  Should have the proper shape to
        broadcast with ``dynspec``.
    f : `~astropy.units.Quantity`
        Frequencies of the dynamic spectrum.  Should have the proper shape to
        broadcast with ``dynspec``.
    noise : float
        The uncertainty in the intensities in the dynamic spectrum.
    d_eff : `~astropy.units.Quantity`
        Assumed effective distance.  This is used throughout and not fit,
        but can be treated as a scaling parameters.
    mu_eff : `~astropy.units.Quantity`, optional
        Initial guess for the effective proper motion, ``v_eff/d_eff``.
    theta : `~astropy.units.Quantity`, optional
        Grid of theta angles to use for modelling the dynamic spectrum.
        Probably more usefully calculated later using ``theta_grid``.
    magnification :  `~astropy.units.Quantity`, optional
        Magnifications at each ``theta``.  More typically inferred by fitting
        the dynamic spectrum.
    """

    def __init__(self, dynspec, f, t, noise, d_eff, mu_eff=None,
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
    def fromfile(cls, filename, d_eff=None, mu_eff=None, noise=None):
        """Read a dynamic spectrum from an HDF5 file.

        This includes its time and frequency axes.

        Note: this needs the baseband-tasks package for HDF5 file access.
        """
        from baseband.io import hdf5

        with hdf5.open(filename) as fh:
            dynspec = fh.read()
            f = fh.frequency
            t = (np.arange(-fh.shape[0] // 2, fh.shape[0] // 2)
                 / fh.sample_rate).to(u.minute)[:, np.newaxis]
            if noise is None:
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

    def theta_grid(self, oversample_tau=1.3, oversample_fd=1.69, **kwargs):
        """Calculate a grid of theta for modelling the dynamic spectrum.

        Wraps `screens.fields.theta_grid` with defaults from the class.
        See that function for details.

        Note that this does *not* set the angles on the class, so typical
        usage is ``ds.theta = ds.theta_grid()``.
        """
        kwargs.setdefault('d_eff', self.d_eff)
        kwargs.setdefault('mu_eff', self.mu_eff)
        kwargs.setdefault('fobs', self.f.mean())
        kwargs.setdefault('dtau', (1./self.f.ptp()).to(u.us)
                          / oversample_tau)
        kwargs.setdefault('dfd', (1./self.t.ptp()).to(u.mHz)
                          / oversample_fd)
        kwargs.setdefault('tau_max', 1/np.abs(self.f[2]-self.f[0]).min())
        kwargs.setdefault('fd_max', 1/np.abs(self.t[2]-self.t[0]).min())
        return theta_grid(**kwargs)

    def dynamic_bases(self, mu_eff=None):
        """Calculate the amplitude infererence patterns for current fit.

        The power of the sum of these amplitudes is a dynamic spectrum.

        Explicitly assumes that time and frequency are the last two axes
        of the dynamic spectrum.  (TODO: lift this restriction.)

        Parameters
        ----------
        mu_eff : `~astropy.units.Quantity`, optional
            Effective proper motion to use.  Defaults to that stored on
            the instance.  Will *not* update the instance.

        Notes
        -----
        The calculated dynamic bases are cached for a given ``mu_eff``.
        """
        if mu_eff is None:
            mu_eff = self.mu_eff

        if mu_eff != getattr(self, '_mu_eff_old', None):
            self._dyn_wave = dynamic_field(
                self.theta, 0., 1.,
                self.d_eff, mu_eff, self.f, self.t).reshape(
                    (1,)*(self.dynspec.ndim-2)+(-1,)+self.dynspec.shape[-2:])
            self._mu_eff_old = mu_eff

        return self._dyn_wave

    def theta_theta(self, mu_eff=None, theta_grid=None, **kwargs):
        """Create a theta-theta array from the dynamic spectrum.

        For a given grid in ``theta`` (possibly calculated) and a set of pairs
        found using `screens.fields.theta_theta_indices`, this brute-force
        estimates the amplitudes at each pair by cross-multiplying their
        expected signature in the dynamic spectrum.

        Parameters
        ----------
        mu_eff : `~astropy.units.Quantity`, optional
            Effective proper motion to use.  Defaults to that stored on
            the instance.  Will update the instance if given.
        theta_grid : bool, optional
            Whether to calculate a new theta grid, or use the one stored
            on the instance.  By default, calculate it only if ``mu_eff`` is
            passed in.  If `True`, this will update the grid stored on the
            instance.
        **kwargs
            Any further arguments are passed on to
            `~screens.dynspec.DynamicSpectrum.theta_grid`
        """
        if mu_eff is not None:
            self.mu_eff = mu_eff

        if theta_grid or (theta_grid is None and mu_eff is not None):
            self.theta = self.theta_grid(**kwargs)
            self._mu_eff_old = None

        dynwave = self.dynamic_bases()
        # Get intensities by brute-force mapping:
        # dynspec * dynwave[j] * dynwave[i].conj() / sqrt(2) for all j > i
        # Do first product ahead of time to speed up calculation
        # (remove constant parts of input spectrum to eliminate edge effects)
        ddyn = dynwave * np.expand_dims(
            self.dynspec - self.dynspec.mean(), -3)
        # Explicit loop is faster than just broadcasting or using indices
        # for advanced indexing, since it avoids creating a large array.
        result = np.zeros(self.dynspec.shape[:-2] + self.theta.shape * 2,
                          ddyn.dtype)
        indices = theta_theta_indices(self.theta)
        for i, j in zip(*indices):
            amplitude = ((ddyn[..., j, :, :]
                          * dynwave[..., i, :, :].conj()).mean((-2, -1))
                         / np.sqrt(2.))
            result[..., i, j] = amplitude
            result[..., j, i] = amplitude.conj()

        return result

    def locate_mu_eff(self, mu_eff_trials=None, verbose=True, use=True,
                      **kwargs):
        """Try reproducing the dynamic spectrum for a range of proper motion.

        For each proper motion, construct a theta-theta array, calculte the
        largest eigenvalue and use the corresponding eigenvector as the model
        one-dimensional screen.

        Parameters
        ----------
        mu_eff_trials : `~astropy.units.Quantity`
            Proper motions to try.
        verbose : bool
            Whether or not to give summary statistics for each trial.
        use : bool
            Whether to store the best-fit proper motion and corresponding
            theta grid and magnifications on the instance.
        **kwargs
            Further parameters to use in calculating the theta grid for each
            proper motion.

        Returns
        -------
        curvature : `~astropy.table.QTable`
            Table with the following columns:
            - ``mu_eff`` : Input proper motions.
            - ``theta`` : grid in theta used.
            - ``w`` : Largest eigenvalue.
            - ``recovered`` : corresponding eigenvector, i.e., magnifications.
            - ``th_ms`` : Mean-square residual in theta-theta space.
            - ``ndof`` : degrees of freedom ``n_dynspec - n_theta - 2``.
            - ``redchi2`` : reduced chi2 ``((dynspec - model)/noise)**2/ndof``.

        Notes
        -----
        The resulting table is also stored on the instance, as ``curvature``.
        """
        if mu_eff_trials is None:
            mu_eff_trials = np.linspace(self.mu_eff * 0.8,
                                        self.mu_eff * 1.2, 21)
        r = QTable([mu_eff_trials], names=['mu_eff'])
        shape = (len(mu_eff_trials),) + self.dynspec.shape[:-2]
        r['theta'] = np.zeros(len(mu_eff_trials), object)
        r['w'] = np.zeros(shape)
        r['recovered'] = np.zeros(len(mu_eff_trials), object)
        r['th_ms'] = np.zeros(shape)
        r['ndof'] = 0
        r['redchi2'] = np.zeros(shape)
        for i, mu_eff in enumerate(r['mu_eff']):
            th_th = self.theta_theta(mu_eff, **kwargs)
            w, v = eigh(th_th)
            recovered = v[..., -1]
            recovered0 = recovered[..., self.theta == 0]

            recovered *= recovered0.conj()/np.abs(recovered0)

            th_ms = (np.abs(
                th_th - (recovered[..., :, np.newaxis]
                         * recovered[..., np.newaxis, :].conj()))**2).mean()
            dynspec_r = self.model(recovered, mu_eff=mu_eff)
            # Mean of dynamic spectra should equal sum of all recovered powers.
            # Since we normalize that to (close to) 1, just rescale similarly.
            dynspec_r *= (self.dynspec.mean((-2, -1), keepdims=True)
                          / dynspec_r.mean((-2, -1), keepdims=True))
            ndof = (self.dynspec.shape[-1] * self.dynspec.shape[-2]
                    - self.theta.size - 2)
            redchi2 = (((self.dynspec-dynspec_r)**2).sum((-2, -1))
                       / self.noise**2) / ndof
            r['theta'][i] = self.theta
            r['w'][i] = w[..., -1]
            r['recovered'][i] = recovered
            r['th_ms'][i] = th_ms
            r['ndof'][i] = ndof
            r['redchi2'][i] = redchi2
            if verbose:
                print(f'{mu_eff} {w[..., -1]} {ndof} {redchi2}')

        self.curvature = r
        if use:
            ibest = r['redchi2'].argmin(0)
            self.theta = r['theta'][ibest]
            self.magnification = np.array(r['recovered'][ibest])
            if self.dynspec.ndim > 2:
                assert self.dynspec.ndim == 3, 'not implemented yet'
                self.magnification = np.array(
                    [self.magnification[i][i]
                     for i in range(self.dynspec.shape[0])],
                    dtype=object)
            self.mu_eff = r['mu_eff'][ibest]

        return r

    def model(self, magnification=None, mu_eff=None):
        """Calculate a model dynamic spectrum.

        Uses parameters on the instance if not otherwise specified.

        Parameters
        ----------
        magnification : array-like, optional
            Complex magnification for each theta value.
        mu_eff : `~astropy.units.Quantity`, optional
            Effective proper motion.
        """
        if magnification is None:
            magnification = self.magnification

        dyn_wave_sum = (self.dynamic_bases(mu_eff)
                        * magnification[..., np.newaxis, np.newaxis]).sum(-3)
        dynspec_r = (dyn_wave_sum.view('2f8')**2).sum(-1)
        return dynspec_r

    def residuals(self, magnification=None, mu_eff=None):
        """Residuals compared to the model.

        Parameters as for :meth:`~screens.dynspec.DynamicSpectrum.model`:
        """
        model = self.model(magnification, mu_eff)
        return (self.dynspec - model) / self.noise

    def jacobian(self, magnification, mu_eff):
        r"""Derivatives of the model dynamic spectrum to all parameters.

        The model dynamic spectrum is

        .. math::

           w(f, t) &= \sum_k \mu_k \exp[-j\phi(\theta_i, \mu_{\rm eff}, f, t)]
                   &\equiv \sum_k \mu_k \Phi_k \\
           DS(f,t) &= |w(f, t)|^2 = w \bar{w}

        It is easiest to use Wirtinger derivatives:

        .. math::

           \frac{\partial}{\partial x}
           &= \left(\frac{\partial}{\partial z}
                  + \frac{\partial}{\partial\bar{z}}\right) \\
           \frac{\partial}{\partial y}
           &= j \left(\frac{\partial}{\partial z}
                     - \frac{\partial}{\partial\bar{z}}\right)

        Using this for the magnifications:

        .. math::

           \frac{\partial DS}{\partial\mu}
           &= \bar{w}\frac{\partial w}{\partial\mu} = \bar{w}\Phi \\
           \frac{\partial DS}{\partial\bar{\mu}}
           &= w \frac{\partial\bar{w}}{\partial\bar{\mu}} = w \bar{\Phi} \\
           \frac{\partial DS}{\partial x}
           &= \left(\bar{w} \Phi + w \bar{\Phi}\right) = 2\Re(w\bar{\Phi}) \\
           \frac{\partial DS}{\partial y}
           &= j\left(\bar{w} \Phi - w \bar{\Phi}\right) = 2\Im(w\bar{\Phi})

        And for :math:`\mu_{\rm eff}`:

        .. math::

           \frac{\partial w}{\partial\mu_{\rm eff}}
           &= \sum_k\mu_k\Phi_k
                \frac{\partial -i\phi_k}{\partial\mu_{\rm eff}} \\
           \frac{\partial\bar{w}}{\partial\mu_{\rm eff}}
           &= \sum_k\bar{\mu}_k\bar{\Phi}_k
               \frac{\partial i\phi_k}{\partial\mu_{\rm eff}} \\
           \frac{\partial DS}{\partial\mu_{\rm eff}}
           &= \bar{w} \frac{\partial w}{\partial\mu_{\rm eff}}
            +  w \frac{\partial\bar{w}}{\partial\mu_{\rm eff}}
            = -2j\sum_k\mu_k \Phi_k \frac{\partial\phi}{\partial\mu_{\rm eff}}

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
        """Turn complex parameters into the real ones used in the fits."""
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
        """Turn real parameters used in the fits into physical ones."""
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
        """Calculate model for the fit routine."""
        magnification, mu_eff = self._separate_pars(pars)
        model = self.model(magnification, mu_eff)
        if self._prt:
            print(mu_eff if mu_eff is not None else self.mu_eff,
                  ((self.dynspec-model)**2).sum()/self.noise**2)

        return model.ravel()

    def _jacobian(self, unused_x_data, *pars):
        """Calculate jacobian for the fit routine."""
        magnification, mu_eff = self._separate_pars(pars)
        jac_mag, jac_mu_eff = self.jacobian(magnification, mu_eff)
        return self._combine_pars(jac_mag, jac_mu_eff,
                                  1/self.mu_eff_guess)

    def fit(self, guess=None, verbose=2, **kwargs):
        """Fit the dynamic spectrum directly.

        This needs good guesses, such as can be gotten from
        :meth:`screens.dynspec.DynamicSpectrum.locate_mu_eff`.

        Parameters
        ----------
        guess : initial guesses, optional
            If `None`, use the current ``magnification`` and ``mu_eff`` on
            the instance.  If ``curvature``, select the best value from the
            ``curvature`` table on the instance (created by ``locate_mu_eff``).
            If a tuple, guesses for ``magnification`` and ``mu_eff``.
            If a single number, treat it as a guess for ``mu_eff`` and
            determine initial guesses for the magnification using eigen-value
            decomposition.
        verbose : int
            How much information to print during fitting.  A value of ``3``
            forces the class itself to print a reduced chi2 any time ``mu_eff``
            is updated.
        **kwargs
            Any further keyword arguments to pass on to
            :func:`scipy.optimize.curve_fit`.

        Notes
        -----
        This uses the ``model`` and ``jacobian`` methods on the instance.
        """
        if guess is None:
            mu_eff_guess = self.mu_eff
            mag_guess = self.magnification
        elif guess == 'curvature':
            ibest = self.curvature['redchi2'].argmin()
            mu_eff_guess = self.curvature['mu_eff'][ibest]
            mag_guess = self.curvature['recovered'][ibest]
            self.theta = self.curvature['theta'][ibest]
        elif isinstance(guess, tuple):
            mag_guess, mu_eff_guess = guess
        else:
            # Assume mu_eff; note this defines new theta grid.
            mu_eff_guess = guess
            th_th = self.theta_theta(mu_eff_guess)
            _, v = eigh(th_th)
            mag_guess = v[..., -1]
            m0 = mag_guess[..., self.theta == 0]
            mag_guess *= m0.conj()/np.abs(m0)

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
        """Remove highly degenerate configurations from the fit.

        Works by doing an singular value decomposition on the fit and
        removing very small values, those that are either smaller than
        ``max_abs_err`` or ``max_rel_err`` times the largest value.
        """
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
