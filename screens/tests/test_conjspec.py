"""Testing of the ConjugateSpectrum class.

TODO: make this more complete!
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u

from screens.fields import dynamic_field
from screens.dynspec import DynamicSpectrum
from screens.conjspec import ConjugateSpectrum


class TestConjSpec:
    def setup_class(self):
        self.fobs = 316 * u.MHz
        f = (np.arange(-16, 16)/2.) << u.MHz
        self.f = self.fobs + f
        self.df = self.f[1]-self.f[0]
        self.t = (np.arange(-16, 16) << u.minute)[:, np.newaxis]
        self.dt = self.t[1, 0]-self.t[0, 0]
        self.d_eff = 1*u.kpc
        self.mu_eff = 30*u.mas/u.yr
        self.theta = [0., -0.3, 0.5] << u.mas
        self.dw = dynamic_field(self.theta, 0., 1.,
                                self.d_eff, self.mu_eff, self.f, self.t)
        self.magnification = np.array([1., 0.2, 0.1j])
        ds = np.abs((self.dw * self.magnification[:, np.newaxis, np.newaxis])
                    .sum(0))**2
        self.ds = DynamicSpectrum(ds, self.f, self.t, 0.001,
                                  d_eff=self.d_eff, mu_eff=self.mu_eff,
                                  magnification=self.magnification)
        self.ds.theta = self.theta
        self.conjspec = np.fft.fftshift(np.fft.fft2(self.ds.dynspec))

    def expected_conjspec(self, norm):
        if norm == 'mean':
            return (self.conjspec
                    / (self.ds.dynspec.size * self.ds.dynspec.mean()))
        else:
            return self.conjspec

    @pytest.mark.parametrize('norm', (None, 'mean'))
    def test_from_dynspec(self, norm):
        cs = ConjugateSpectrum.from_dynamic_spectrum(self.ds,
                                                     normalization=norm)
        assert_allclose(cs.conjspec, self.expected_conjspec(norm))
        assert_allclose(cs.tau, np.fft.fftshift(
            np.fft.fftfreq(len(self.f), self.df)), atol=0)
        assert_allclose(cs.fd, np.fft.fftshift(
            np.fft.fftfreq(len(self.t), self.dt))[:, np.newaxis], atol=0)

    @pytest.mark.parametrize('norm', (None, 'mean'))
    def test_from_dynspec_explicit_t(self, norm):
        # Uses phasor instead of fft, so this checks that the phase
        # convention is the same.
        # Need to work relative to first element since FT defines
        # phase that way.
        t = np.broadcast_to(self.t-self.t[0], self.ds.dynspec.shape,
                            subok=True)
        cs = ConjugateSpectrum.from_dynamic_spectrum(self.ds, t=t,
                                                     normalization=norm)
        assert_allclose(cs.conjspec, self.expected_conjspec(norm))

    @pytest.mark.parametrize('norm', (None, 'mean'))
    def test_from_dynspec_explicit_fd(self, norm):
        # Uses phasor instead of fft, so this checks that the phase
        # convention is the same.  Again, work relative to start time
        # as does FFT.
        fd = np.fft.fftshift(np.fft.fftfreq(len(self.t), self.dt))
        cs = ConjugateSpectrum.from_dynamic_spectrum(
            self.ds, fd=fd, t=self.t-self.t[0], normalization=norm)
        assert_allclose(cs.conjspec, self.expected_conjspec(norm))
