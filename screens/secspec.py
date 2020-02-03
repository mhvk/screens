# Licensed under the GPLv3 - see LICENSE
import numpy as np
from astropy import units as u, constants as const
from astropy.table import QTable
from scipy.linalg import eigh

from .fields import theta_grid, theta_theta_indices, phasor, expand
from .dynspec import DynamicSpectrum


__all__ = ['SecondarySpectrum']


class SecondarySpectrum:
    def __init__(self, secspec, tau, fd, noise, d_eff, mu_eff,
                 theta=None, magnification=None):
        self.secspec = secspec
        self.tau = tau
        self.fd = fd
        self.noise = noise
        self.d_eff = d_eff
        self.mu_eff = mu_eff
        self.theta = theta
        self.magnification = magnification

    @classmethod
    def from_dynamic_spectrum(cls, dynspec, *, fd=None, **kwargs):
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
        if t.size in t.shape and fd is None:  # fast FFT possible.
            sec = np.fft.fft2(dynspec)
            fd = np.fft.fftfreq(t.size, t[1]-t[0]).to(u.mHz).reshape(t.shape)
        else:
            # Time axis has slow FT or explicit fd given.
            # Time is assumed to be along axis -2.
            if fd is None:
                t_step = np.abs(np.diff(t, axis=-2)).min()
                n_t = round((t.ptp()/t_step).to_value(1).item()) + 1
                fd = np.fft.fftfreq(n_t, t_step)

            if fd.ndim == 1:
                fd = expand(fd, n=dynspec.ndim)

            dt = np.diff(t, axis=-1)
            linear_axis = -1 if np.allclose(dt, dt[..., :1]) else None
            factor = phasor(t, fd, linear_axis=linear_axis)
            factor *= dynspec
            step1 = factor.sum(-2, keepdims=True).swapaxes(0, -2).squeeze(0)
            sec = np.fft.fft(step1, axis=-1)
            fd.shape = sec.shape[-2], 1

        sec /= sec[0, 0]
        tau = np.fft.fftfreq(f.size, f[1]-f[0]).to(u.us)
        tau.shape = f.shape

        sec = np.fft.fftshift(sec)
        tau = np.fft.fftshift(tau) << tau.unit
        fd = np.fft.fftshift(fd) << fd.unit
        self = cls(sec, tau, fd, **kwargs)
        self.f = f
        self.t = t
        return self

    def theta_grid(self, oversample_tau=2, oversample_fd=4, **kwargs):
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
        if mu_eff is None:
            mu_eff = self.mu_eff

        if theta_grid:
            self.theta = self.theta_grid(mu_eff=mu_eff, **kwargs)

        sec = self.secspec
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
        ok = ((idtau >= 0) & (idtau < sec.shape[-1])
              & (idfd >= 0) & (idfd < sec.shape[-2]))
        theta_theta = np.zeros(self.theta.shape*2, sec.dtype)
        amplitude = sec[idfd[ok], idtau[ok]]
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
        if magnification is None:
            magnification = self.magnification
        if mu_eff is None:
            mu_eff = self.mu_eff

        fobs = self.f.mean()
        tau_factor = self.d_eff/(2.*const.c)
        fd_factor = self.d_eff*mu_eff*fobs/const.c
        ifd, itau = np.indices(self.secspec.shape, sparse=True)
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
        model = np.zeros_like(self.secspec, magnification.dtype)
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
        if mu_eff_trials is None:
            mu_eff_trials = np.linspace(self.mu_eff * 0.5,
                                        self.mu_eff * 2, 201)
        r = QTable([mu_eff_trials], names=['mu_eff'])
        r['w'] = 0.
        r['th_ms'] = 0.
        r['redchi2'] = 0.
        r['recovered'] = np.zeros(len(r), object)
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
            secspec_r = self.model(recovered, mu_eff=mu_eff)
            if power:
                redchi2 = ((np.abs(self.secspec)**2 - secspec_r)**2).mean()
            else:
                redchi2 = (np.abs(self.secspec-secspec_r)**2).mean()

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
        return r
