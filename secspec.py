import numpy as np
from astropy import units as u, constants as const
from astropy.table import QTable
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt
from scipy.linalg import eigh

from fields import theta_theta_indices
from dynspec import DynamicSpectrum, theta_grid
from visualization import ThetaTheta


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
    def from_dynamic_spectrum(cls, dynspec, **kwargs):
        for key in ('f', 't', 'd_eff', 'mu_eff', 'theta',
                    'magnification', 'noise'):
            val = getattr(dynspec, key, None)
            if val is not None:
                kwargs.setdefault(key, val)

        if isinstance(dynspec, DynamicSpectrum):
            # TODO: give DynamicSpectrum an __array__ method.
            dynspec = dynspec.dynspec

        sec = np.fft.fft2(dynspec)
        sec /= sec[0, 0]
        f = kwargs.pop('f')
        tau = np.fft.fftfreq(f.size, f[1]-f[0]).to(u.us)
        tau.shape = f.shape
        t = kwargs.pop('t')
        fd = np.fft.fftfreq(t.size, t[1]-t[0]).to(u.mHz)
        fd.shape = t.shape

        sec = np.fft.fftshift(sec)
        tau = np.fft.fftshift(tau) << tau.unit
        fd = np.fft.fftshift(fd) << fd.unit
        self = cls(sec, tau, fd, **kwargs)
        self.f = f
        self.t = t
        return self

    def theta_theta(self, mu_eff=None, conserve=False):
        if mu_eff is None:
            mu_eff = self.mu_eff

        sec = self.secspec
        # TODO: Adjust grid code to take in tau, fd.
        self.theta = theta_grid(self.d_eff, mu_eff, self.f, self.t,
                                tau_max=self.tau.max()*2/3,
                                fd_max=self.fd.max(),
                                oversample_tau=2, oversample_fd=4)
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


if __name__ == '__main__':
    plt.ion()
    plt.clf()
    quantity_support()

    dyn_spec = DynamicSpectrum.fromfile('dynspec.h5', d_eff=1.*u.kpc,
                                        mu_eff=100*u.mas/u.yr)
    sec_spec = SecondarySpectrum.from_dynamic_spectrum(dyn_spec)

    sec_kwargs = dict(extent=(sec_spec.fd[0].value, sec_spec.fd[-1].value,
                              sec_spec.tau[0].value, sec_spec.tau[-1].value),
                      cmap='Greys', vmin=-7, vmax=0, origin=0, aspect='auto')
    plt.subplot(321)
    plt.imshow(np.log10(np.abs(sec_spec.secspec)**2).T, **sec_kwargs)

    conserve = True

    mu_eff = 100*u.mas/u.yr
    th_kwargs = sec_kwargs.copy()
    th_kwargs['extent'] = (sec_spec.theta[0].value,
                           sec_spec.theta[-1].value)*2
    th_kwargs['aspect'] = 'equal'
    th_th = sec_spec.theta_theta(mu_eff, conserve=conserve)
    th_th_proj = ThetaTheta(sec_spec.theta)
    ax = plt.subplot(322, projection=th_th_proj)
    ax.imshow(np.log10(np.maximum(np.abs(th_th)**2, 1e-30)).T, **th_kwargs)

    secspec = sec_spec.secspec.copy()
    secspec[(sec_spec.fd == 0) | (sec_spec.tau == 0)] = 0
    # try recoving just plain power
    w_a, v_a = eigh(np.abs(th_th)**2, eigvals=(sec_spec.theta.size-1,)*2)
    rec_a = v_a[:, -1] * np.sqrt(w_a[-1])
    th_th_rp = rec_a[:, np.newaxis] * rec_a
    ax = plt.subplot(324, projection=th_th_proj)
    ax.imshow(np.log10(np.maximum(th_th_rp, 1e-30)).T, **th_kwargs)

    sec_rp = sec_spec.model(rec_a, mu_eff, conserve=conserve)
    sec_p = np.abs(secspec)**2
    sec_p_noise = sec_p[:30, :30].std()
    print("just power, red chi2 = ",
          ((np.abs(secspec)**2-sec_rp)**2).mean() / sec_p_noise**2)
    plt.subplot(323)
    plt.imshow(np.log10(np.maximum(sec_rp, 1e-30)).T, **sec_kwargs)

    # try recovering phases as well.
    w, v = eigh(th_th, eigvals=(sec_spec.theta.size-1,)*2)
    recovered = v[:, -1] * np.sqrt(w[-1])
    th_th_r = recovered[:, np.newaxis] * recovered
    ax = plt.subplot(326, projection=th_th_proj)
    ax.imshow(np.log10(np.maximum(np.abs(th_th_r)**2, 1e-30)).T, **th_kwargs)

    sec_r = sec_spec.model(recovered, mu_eff, conserve=conserve)
    sec_noise = secspec[:30, :30].std()
    print("also phase, red chi2 = ",
          (np.abs(secspec-sec_r)**2).mean() / sec_noise**2)
    print(" power red chi2 = ",
          ((np.abs(secspec)**2-np.abs(sec_r)**2)**2).mean() / sec_p_noise**2)
    plt.subplot(325)
    plt.imshow(np.log10(np.maximum(np.abs(sec_r)**2, 1e-30)).T, **sec_kwargs)
