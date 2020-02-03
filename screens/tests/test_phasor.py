import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u

import pytest

from ..fields import phasor


class TestPhasor:
    def setup_class(cls):
        cls.fobs = 316 * u.MHz
        f = np.arange(-8, 8) << u.MHz
        cls.f = cls.fobs + f
        cls.t = (np.arange(-16, 16) << u.minute)[:, np.newaxis]
        cls.tau = np.fft.fftfreq(cls.f.size, 1*u.MHz)
        cls.fd = np.fft.fftfreq(cls.t.size, 1*u.minute)
        cls.w_tau = 0.3 * u.us
        cls.w_fd = 4 * u.mHz
        phase = (cls.f*cls.w_tau + cls.t*cls.w_fd)*u.cycle
        cls.dw = np.exp(1j*phase.to_value(u.radian))

    def test_linear(self):
        t = self.t - self.t[0]
        fd = self.fd[:, np.newaxis, np.newaxis]
        full = phasor(t, fd)
        linear = phasor(t, fd)
        assert_allclose(full, linear, atol=1e-8, rtol=0)

    @pytest.mark.parametrize('linear_axis', (None, -2))
    def test_ft_t(self, linear_axis):
        expected = np.fft.fft(self.dw, axis=0)
        # Regular FT is always relative to start of array.
        t = self.t - self.t[0]
        phs = phasor(t, self.fd[:, np.newaxis, np.newaxis],
                     linear_axis=linear_axis)
        result = (self.dw*phs).sum(1)
        assert_allclose(result, expected, atol=1e-8, rtol=0)

    @pytest.mark.parametrize('linear_axis', (None, -1))
    def test_ft_f(self, linear_axis):
        expected = np.fft.fft(self.dw, axis=1)
        # Regular FT is always relative to start of array.
        f = self.f - self.f[0]
        phs = phasor(f, self.tau[:, np.newaxis, np.newaxis],
                     linear_axis=linear_axis)
        result = (self.dw*phs).sum(-1).T
        assert_allclose(result, expected, atol=1e-8, rtol=0)

    def test_ft(self):
        expected = np.fft.fft2(self.dw)
        # Regular FT always has phase relative to start of array.
        t = self.t - self.t[0]
        f = self.f - self.f[0]
        phs_t = phasor(t, self.fd[:, np.newaxis, np.newaxis, np.newaxis],
                       linear_axis=-2)
        phs_f = phasor(f, self.tau[:, np.newaxis, np.newaxis],
                       linear_axis=-1)
        result = (self.dw*phs_t*phs_f).sum((-2, -1))
        assert_allclose(result, expected, atol=1e-8, rtol=0)
