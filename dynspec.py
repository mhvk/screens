import numpy as np
from astropy import units as u, constants as const
from astropy.table import QTable
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import eigh
import h5py

from scintillometry.io import hdf5

from fields import dynamic_field, theta_theta


def theta_grid(d_eff, mu_eff, f, t, tau_max=None, fd_max=None,
               oversample_tau=1.3, oversample_fd=1.3**2):
    """Make a grid of theta that sample the parabola in a particular way.

    The idea would be to impose the constraint that near tau_max the
    spacing is roughly the spacing allowed by the frequencies, and
    near the origin that allowed by the times.  In practice, one needs
    to oversample in both directions.  The factor 1.3 makes some sense
    from wanting to sample resolution elements with about 3 points
    (rather than 2 for a FFT), but the reason one needs to sample more
    densely in doppler factor is less clear.

    Parameters
    ----------
    d_eff : ~astropy.units.Quantity
        Effective distance.  Should be constant; if different for
        different points, no screen-to-screen scattering is taken into
        account.
    mu_eff : ~astropy.units.Quantity
        Effective proper motion (``v_eff / d_eff``), parallel to ``theta_par``.
    t : ~astropy.units.Quantity
        Times for which the dynamic wave spectrum should be calculated.
    f : ~astropy.units.frequency
        Frequencies for which the spectrum should be calculated.
    tau_max : ~astropy.units.Quantity
        Maximum delay to consider.  If not given, taken as the value
        implied by the frequency resolution (i.e., ``1/(f[2]-f[0])``).
    fd_max : ~astropy.units.Quantity
        Maximum doppler factor to consider.  If not given, taken as the
        value implied by the time resolution (i.e., ``1/(t[2]-t[0])``).
    oversample_tau : float
        Factor by which to oversample pixels in tau (inverse frequency).
        This is a somewhat finicky number: With 1, dynamic spectra are
        underfit, while with 1.5 fitting takes very long as points are
        strongly correlated.
    oversample_fd : float
        Factor by which to oversample pixels in doppler factor (inverse
        time).  Seems best to roughly use the square of ``oversample_tau``,
        not quite clear why.
    """
    # Calculate what the pixels spacings would be in a 2D FFT;
    # Since a FFT can fully describe the data, any spacing in our grid
    # needs to be at least the same.
    fobs = f.mean()
    tau_factor = d_eff/(2.*const.c)
    fd_factor = d_eff*mu_eff*fobs/const.c
    dtau = (1./f.ptp()).to(u.us) / oversample_tau
    dfd = (1./t.ptp()).to(u.mHz) / oversample_fd
    # Curvature of the parabola in these pixel units.
    a_pix = (tau_factor/dtau * (dfd/fd_factor)**2).to_value(
        1, equivalencies=u.dimensionless_angles())
    if tau_max is None:
        tau_max = 1/(f[2]-f[0])
    if fd_max is None:
        fd_max = 1/(t[2]-t[0])
    th_max = min(np.sqrt(tau_max/tau_factor).to(
        u.mas, u.dimensionless_angles()),
                 (fd_max/fd_factor).to(
                     u.mas, u.dimensionless_angles()))
    tau_max_use = (tau_factor * th_max**2).to(u.us, u.dimensionless_angles())
    y_max = (tau_max_use/dtau).to_value(1)
    s_max = round(path_length(a_pix, y_max))
    # Corresponding path length around a parabola
    s = np.arange(1, s_max+1)
    # Guesses for y
    y = np.linspace(1, y_max, s.size)
    d_s = s - path_length(a_pix, y)
    it = 0
    while np.any(np.abs(d_s) > 1e-6) and it < 100:
        dsdy = np.sqrt(1+1/(4*a_pix*y))
        y = np.maximum(y + (d_s)/dsdy, 1e-10)
        d_s = s - path_length(a_pix, y)
        it += 1

    s = np.hstack([-s[::-1], 0, s[:]])
    y = np.hstack([y[::-1], 0, y[:]])
    th_r = (np.sqrt(y/y_max*tau_max_use/tau_factor) * np.sign(s)).to(
        u.mas, u.dimensionless_angles())
    return th_r


def path_length(a, y):
    b = 1/(4*a)
    sq = np.sqrt((b+y)*y)
    return sq + (b/2)*np.log((sq+y)/(sq-y))


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

    def theta_grid(self, **kwargs):
        return theta_grid(self.d_eff, self.mu_eff, self.f, self.t,
                          **kwargs)

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


if __name__ == '__main__':
    plt.ion()
    quantity_support()

    dyn_chi2 = DynamicSpectrum.fromfile('dynspec.h5', d_eff=1.*u.kpc,
                                        mu_eff=100*u.mas/u.yr)
    dyn_chi2.theta = dyn_chi2.theta_grid(
        tau_max=(1./(dyn_chi2.f[3]-dyn_chi2.f[0])).to(u.us))
    # if not hasattr(dyn_chi2, 'curvature'):
    dyn_chi2.locate_mu_eff(np.arange(98, 103) << u.mas/u.yr)

    r = dyn_chi2.curvature

    plt.clf()
    plt.subplot(3, 4, 1)
    plt.plot(r['mu_eff'], r['redchi2'])

    plt.subplot(3, 4, 2)
    th_th = dyn_chi2.theta_theta(r.meta['mu_eff'])
    th = dyn_chi2.theta
    th_kwargs = dict(extent=(th[0].value, th[-1].value)*2,
                     origin=0, vmin=-7, vmax=0, cmap='Greys')
    plt.imshow(np.log10(np.maximum(np.abs(th_th)**2, 1e-30)), **th_kwargs)
    plt.xlabel(th.unit.to_string('latex'))
    plt.ylabel(th.unit.to_string('latex'))

    ds_kwargs = dict(extent=(dyn_chi2.t[0].value, dyn_chi2.t[-1].value,
                             dyn_chi2.f[0].value, dyn_chi2.f[-1].value),
                     origin=0, aspect='auto', cmap='Greys',
                     vmin=0, vmax=7)
    ibest = r['redchi2'].argmin()
    dynspec_r = dyn_chi2.model(r['recovered'][ibest],
                               mu_eff=r['mu_eff'][ibest])
    dynspec_r *= dyn_chi2.dynspec.mean()/dynspec_r.mean()
    plt.subplot(3, 4, 3)
    plt.imshow(dynspec_r.T, **ds_kwargs)

    res_kwargs = ds_kwargs.copy()
    res_kwargs['vmin'], res_kwargs['vmax'] = -5, 5
    plt.subplot(3, 4, 4)
    plt.imshow((dyn_chi2.dynspec-dynspec_r).T/dyn_chi2.noise, **res_kwargs)

    plt.subplot(3, 4, 7)
    plt.imshow(dyn_chi2.dynspec.T, **ds_kwargs)

    (raw_mag_fit, raw_mag_err,
     raw_mu_eff_fit, raw_mu_eff_err) = dyn_chi2.fit(
         guess='curvature', verbose=3,
         ftol=0.1/dyn_chi2.dynspec.size)

    plt.subplot(3, 4, 5)
    plt.plot(th, np.abs(raw_mag_fit), th, np.abs(raw_mag_err))

    rd = np.sqrt(dyn_chi2.pcovar.diagonal())
    rc = dyn_chi2.pcovar/rd/rd[:, np.newaxis]
    plt.subplot(3, 4, 6)
    plt.imshow(rc, vmin=-1, vmax=1)

    plt.subplot(3, 4, 8)
    plt.imshow(dyn_chi2.residuals(raw_mag_fit, raw_mu_eff_fit).T,
               **res_kwargs)

    (cln_mag_fit, cln_mag_err,
     cln_mu_eff_fit, cln_mu_eff_err) = dyn_chi2.cleanup_fit()

    plt.subplot(3, 4, 9)
    plt.plot(th, np.abs(cln_mag_fit), th, np.abs(cln_mag_err))

    cd = np.sqrt(dyn_chi2.ccovar.diagonal())
    cc = dyn_chi2.ccovar/cd/cd[:, np.newaxis]
    plt.subplot(3, 4, 10)
    plt.imshow(cc, vmin=-1, vmax=1)

    plt.subplot(3, 4, 11)
    plt.imshow(dyn_chi2.model(cln_mag_fit, cln_mu_eff_fit).T,
               **ds_kwargs)
    plt.subplot(3, 4, 12)
    plt.imshow(dyn_chi2.residuals(cln_mag_fit, cln_mu_eff_fit).T,
               **res_kwargs)
