# Licensed under the GPLv3 - see LICENSE
import numpy as np
from astropy import units as u, constants as const
from astropy.table import QTable
from scipy.linalg import eigh

from .fields import theta_grid, theta_theta_indices, phasor, expand
from .dynspec import DynamicSpectrum


__all__ = ['ConjugateSpectrum']


class ConjugateSpectrum:
    """Conjugate spectrum and methods to fit it.

    The code is meant to be agnostic to which axes are which, but some may
    assume a shape of ``(..., doppler_axis, delay_axis)``.

    Parameters
    ----------
    conjspec : `~numpy.ndarray`
        Fourier transform of a dynamic spectrum.
    fd : `~astropy.units.Quantity`
        Doppler factors of the conjugate spectrum.  Normally time conjugate
        but can be arbitrary (e.g., conjugate of ``f*t``).  Should have the
        the proper shape to broadcast with ``conjspec``.
    tau : `~astropy.units.Quantity`
        Delays of the conjugate spectrum.  Should have the proper shape to
        broadcast with ``dynspec``.
    noise : float
        The uncertainty in the real and imaginary components of the conjugate
        spectrum.
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
        the secondary spectrum.
    """

    def __init__(self, conjspec, tau, fd, noise=None, d_eff=None, mu_eff=None,
                 theta=None, magnification=None):
        self.conjspec = conjspec
        self.tau = tau
        self.fd = fd
        self.noise = noise
        self.d_eff = d_eff
        self.mu_eff = mu_eff
        self.theta = theta
        self.magnification = magnification

    @classmethod
    def from_dynamic_spectrum(cls, dynspec, normalization='mean', **kwargs):
        """Create a conjugate spectrum from a dynamic one.

        Easiest if the input is a `~screens.dynspec.DynamicSpectrum`
        instance.

        By passing in an explicit time axis using ``t``, one can get a
        different delay factor conjugate.  Particularly useful with
        ``t=f*t``, which takes into account the frequency dependence of
        the time variation of scintles.

        Note that the dynamic spectrum is assumed to have shape
        ``(..., time_axis, frequency_axis)``.

        Parameters
        ----------
        dynspec : array_like or `~screens.dynspec.DynamicSpectrum`
            Input dynamic spectrum for which the fourier transform will
            be calculated.  If it has attributes ``f``, ``t``, ``d_eff``,
            ``theta``, ``magnification``, and ``noise``, those will be
            used as default inputs.  TODO: ``noise`` is likely wrong!
        normalization : 'mean' or None
            Normalize such that the 0, 0 element equals the mean of the
            dynamic spectrum.
        **kwargs
            Other arguments to initialize the conjugate spectrum.
        """
        for key in ('f', 't', 'd_eff', 'mu_eff', 'theta',
                    'magnification', 'noise'):
            val = getattr(dynspec, key, None)
            if val is not None:
                kwargs.setdefault(key, val)

        if isinstance(dynspec, DynamicSpectrum):
            # TODO: give DynamicSpectrum an __array__ method.
            dynspec = dynspec.dynspec

        f = kwargs.pop('f')
        t = kwargs.pop('t')
        fd = kwargs.pop('fd', None)
        if t.size in t.shape and fd is None:  # fast FFT possible.
            conj = np.fft.fftshift(np.fft.fft2(dynspec))
            fd = np.fft.fftshift(np.fft.fftfreq(t.size, t[1]-t[0]).to(u.mHz)
                                 .reshape(t.shape))
        else:
            # Time axis has slow FT or explicit fd given.
            # Time is assumed to be along axis -2.
            # TODO: relax this assumption.
            if fd is None:
                t_step = np.abs(np.diff(t, axis=-2)).min()
                n_t = round((t.ptp()/t_step).to_value(1).item()) + 1
                fd = np.fft.fftshift(np.fft.fftfreq(n_t, t_step).to(u.mHz))

            if fd.ndim == 1:
                fd = expand(fd, n=dynspec.ndim)

            if t.shape[-1] == 1:
                factor = phasor(t, fd, linear_axis=None).conj() * dynspec
            else:
                dt = np.diff(t, axis=-1)
                # Check whether our last axis (generally frequency) is linearly
                # spaced, so we can speed up the calculation.
                linear_axis = -1 if np.allclose(dt, dt[..., :1]) else None
                factor = phasor(t, fd, linear_axis=linear_axis).conj()
                factor *= dynspec

            step1 = factor.sum(-2, keepdims=True).swapaxes(0, -2).squeeze(0)
            conj = np.fft.fftshift(np.fft.fft(step1, axis=-1), axes=-1)
            fd.shape = conj.shape[-2], 1

        if normalization == 'mean':
            normalization = conj[conj.shape[-2] // 2, conj.shape[-1] // 2]
            conj /= normalization

        tau = np.fft.fftshift(np.fft.fftfreq(f.size, f[1]-f[0]).to(u.us))
        tau.shape = f.shape
        self = cls(conj, tau, fd, **kwargs)
        self.f = f
        self.t = t
        self.normalization = normalization
        return self

    def theta_grid(self, oversample_tau=2, oversample_fd=4, **kwargs):
        """Calculate a grid of theta for modelling the secondary spectrum.

        Wraps `screens.fields.theta_grid` with defaults from the class.
        See that function for details.  Oversampling is set relatively high
        here, since covariances between the theta are not as important as
        for fitting a dynamic spectrum.

        Note that this does *not* set the angles on the class, so typical
        usage is ``ds.theta = ds.theta_grid()``.
        """
        kwargs.setdefault('d_eff', self.d_eff)
        kwargs.setdefault('mu_eff', self.mu_eff)
        kwargs.setdefault('fobs', self.f.mean())
        kwargs.setdefault('dtau', (self.tau[1]-self.tau[0]).squeeze()
                          / oversample_tau)
        kwargs.setdefault('dfd', (self.fd[1]-self.fd[0]).squeeze()
                          / oversample_fd)
        kwargs.setdefault('tau_max', self.tau.max() * 2/3)
        kwargs.setdefault('fd_max', self.fd.max() * 2/3)
        return theta_grid(**kwargs)

    def theta_theta(self, mu_eff=None, conserve=False, theta_grid=True,
                    **kwargs):
        """Project the secondary spectrum into theta-theta space.

        For a given grid in ``theta`` (possibly calculated) and a set of pairs
        found using `screens.fields.theta_theta_indices`, interpolate in the
        secondary spectrum to find the theta-theta arrays.

        Parameters
        ----------
        mu_eff : `~astropy.units.Quantity`, optional
            Effective proper motion to use.  Defaults to that stored on
            the instance.  Will update the instance if given.
        conserve : bool
            Whether to conserve flux per surface area.  Doing so reduces
            sensitivity to points near the axes, but means one cannot
            directly use any eigenvectors directly in constructing dynamic
            spectra.
        theta_grid : bool, optional
            Whether to calculate a new theta grid, or use the one stored
            on the instance.  By default, calculate it only if ``mu_eff`` is
            passed in.  If `True`, this will update the grid stored on the
            instance.
        **kwargs
            Any further arguments are passed on to
            `~screens.dynspec.DynamicSpectrum.theta_grid`
        """
        if mu_eff is None:
            mu_eff = self.mu_eff

        if theta_grid:
            self.theta = self.theta_grid(mu_eff=mu_eff, **kwargs)

        conj = self.conjspec
        i0, i1 = theta_theta_indices(self.theta)
        i1 = i1.ravel()
        fobs = self.f.mean()
        tau_factor = self.d_eff/(2.*const.c)
        fd_factor = self.d_eff*mu_eff*fobs/const.c
        tau_th = (tau_factor*self.theta**2).to(u.us, u.dimensionless_angles())
        fd_th = (fd_factor*self.theta).to(u.mHz, u.dimensionless_angles())

        dtau = tau_th[i0] - tau_th[i1]
        dfd = fd_th[i0] - fd_th[i1]
        idtau = np.round(((dtau-self.tau[0])
                          / (self.tau[1]-self.tau[0])).to_value(1)).astype(int)
        idfd = np.round(((dfd-self.fd[0])
                         / (self.fd[1]-self.fd[0])).to_value(1)).astype(int)
        ok = ((idtau >= 0) & (idtau < conj.shape[-1])
              & (idfd >= 0) & (idfd < conj.shape[-2]))
        theta_theta = np.zeros(self.theta.shape*2, conj.dtype)
        amplitude = conj[idfd[ok], idtau[ok]]
        if conserve:
            # Area conversion factor:
            # abs(Δtau[i0]*Δfd[i1]-Δtau[i1]*Δfd[i0])/(Δth[i0]*Δth[i1])
            # Δtau = dtau/dth * Δth = tau_factor * 2 * theta * Δtheta
            # Δfd = dfd/dth * Δth = fd_factor * Δth
            # -> conversion factor
            # abs(dtau/dth[i0]*dfd/dth[i1] - dtau/dth[i1]*dfd/dth[i0])
            # = abs(tau_factor*2*(theta[i0]-theta[i1])*fd_factor)
            # = abs(tau_factor*2*dfd) [see above].
            area = np.abs(tau_factor * 2. * dfd).to_value(
                u.us * u.mHz / u.mas**2, u.dimensionless_angles())
            amplitude[ok] *= area

        theta_theta[i0[ok], i1[ok]] = amplitude
        theta_theta[i1[ok], i0[ok]] = amplitude
        return theta_theta

    def model(self, magnification=None, mu_eff=None, conserve=False):
        """Calculate the expected secondary spectrum for given parameters."""
        if magnification is None:
            magnification = self.magnification
        if mu_eff is None:
            mu_eff = self.mu_eff

        fobs = self.f.mean()
        tau_factor = self.d_eff/(2.*const.c)
        fd_factor = self.d_eff*mu_eff*fobs/const.c
        ifd, itau = np.indices(self.conjspec.shape, sparse=True)
        fd = self.fd[ifd, 0]
        tau = self.tau[itau]
        # On purpose, keep the sign.
        th_tau2 = (tau/tau_factor).to(u.mas**2, u.dimensionless_angles())
        th_fd = (fd/fd_factor).to(u.mas, u.dimensionless_angles())
        # th_fd = th1-th2
        # th_tau = np.sqrt(th1**2-th2**2) = np.sqrt((th1-th2)(th1+th2))
        # -> th1+th2 = th_tau / th_fd
        # -> th1 = (th_tau**2 / th_fd + th_fd) / 2
        # -> th2 = (th_tau**2 / th_fd - th_fd) / 2
        with np.errstate(all='ignore'):
            ths = np.stack([(th_tau2 / th_fd + sign * th_fd) / 2.
                            for sign in (-1, +1)], axis=0) << self.theta.unit
        ith = np.searchsorted(self.theta, ths)
        # Only keep pairs which are inside the grid and not on the axes
        ok = ((fd != 0) & (tau != 0)
              & np.all((ith > 0) & (ith < self.theta.size-1), axis=0))
        ths = ths[:, ok]
        ith = ith[:, ok]
        goupone = self.theta[ith+1] - ths < ths - self.theta[ith]
        ith += goupone
        model = np.zeros_like(self.conjspec, magnification.dtype)
        amplitude = magnification[ith[1]] * magnification[ith[0]].conj()
        if conserve:
            area = (np.abs(tau_factor * 2. * fd_factor * (ths[1] - ths[0]))
                    .to_value(u.us * u.mHz / u.mas**2,
                              u.dimensionless_angles()))
            model[ok] = amplitude / area
        else:
            model[ok] = amplitude
        return model

    def locate_mu_eff(self, mu_eff_trials=None, power=True, verbose=False):
        """Try reproducing the secondary spectrum for a range of proper motion.

        For each proper motion, construct a theta-theta array, calculte the
        largest eigenvalue and use the corresponding eigenvector as the model
        one-dimensional screen.

        Parameters
        ----------
        mu_eff_trials : `~astropy.units.Quantity`
            Proper motions to try.
        power : bool
            Whether to just decompose the powers or the full complex secondary
            spectrum.  The former is faster and varies less quickly with
            proper motion, so can be used to find the global minimum before
            re-running on the full complex values.
        verbose : bool
            Whether or not to give summary statistics for each trial.

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

        Note that the noise in the secondary spectrum is currently ignored.
        """
        if mu_eff_trials is None:
            mu_eff_trials = np.linspace(self.mu_eff * 0.5,
                                        self.mu_eff * 2, 201)
        r = QTable([mu_eff_trials], names=['mu_eff'])
        r['theta'] = np.zeros(len(r), object)
        r['w'] = 0.
        r['recovered'] = np.zeros(len(r), object)
        r['th_ms'] = 0.
        r['redchi2'] = 0.
        for i, mu_eff in enumerate(r['mu_eff']):
            th_th = self.theta_theta(mu_eff)
            if power:
                th_th = np.abs(th_th)**2

            w, v = eigh(th_th, eigvals=(self.theta.size-1,)*2)
            recovered = v[:, -1] * np.sqrt(w[-1])

            if power:
                th_ms = ((th_th - (recovered[:, np.newaxis]
                                   * recovered))**2).mean()
            else:
                recovered0 = recovered[self.theta == 0]
                recovered *= recovered0.conj()/np.abs(recovered0)

                th_ms = (np.abs((th_th - (recovered[:, np.newaxis]
                                          * recovered.conj())))**2).mean()
            conjspec_r = self.model(recovered, mu_eff=mu_eff)
            if power:
                redchi2 = ((np.abs(self.conjspec)**2 - conjspec_r)**2).mean()
            else:
                redchi2 = (np.abs(self.conjspec-conjspec_r)**2).mean()

            r['theta'][i] = self.theta
            r['w'][i] = w[-1]
            r['recovered'][i] = recovered
            r['th_ms'][i] = th_ms
            r['redchi2'][i] = redchi2
            if verbose:
                print(f'{mu_eff} {w[-1]} {th_ms} {redchi2}')

        r.meta['theta'] = self.theta
        r.meta['d_eff'] = self.d_eff
        r.meta['mu_eff'] = r['mu_eff'][r['redchi2'].argmin()]

        self.curvature = r
        return r
