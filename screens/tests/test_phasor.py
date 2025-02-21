import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u

import pytest

from ..fields import phasor


NUMPY_LT_2_0 = np.__version__[0] < "2"


class TestPhasor:

    f_type = np.float64
    c_type = np.complex128
    atol = 1e-10

    def setup_class(cls):
        cls.fobs = cls.f_type(316) * u.MHz
        f = np.arange(-8, 8, dtype=cls.f_type) << u.MHz
        cls.f = cls.fobs + f
        cls.t = (np.arange(-16, 16, dtype=cls.f_type)
                 << u.minute)[:, np.newaxis]
        cls.tau = np.fft.fftfreq(cls.f.size, 1*u.MHz).astype(cls.f_type)
        cls.fd = np.fft.fftfreq(cls.t.size, 1*u.minute).astype(cls.f_type)
        cls.w_tau = cls.f_type(0.3) * u.us
        cls.w_fd = cls.f_type(4) * u.mHz
        phase = (cls.f*cls.w_tau + cls.t*cls.w_fd)*u.cycle
        cls.dw = np.exp(1j*phase.to_value(u.radian))

    def test_setup_dtype(self):
        assert self.fobs.dtype == self.f_type
        assert self.f.dtype == self.f_type
        assert self.t.dtype == self.f_type
        assert self.tau.dtype == self.f_type
        assert self.fd.dtype == self.f_type
        assert self.w_tau.dtype == self.f_type
        assert self.w_fd.dtype == self.f_type
        assert self.dw.dtype == self.c_type

    @pytest.mark.parametrize('linear_axis', (-1, "transform"))
    def test_linear(self, linear_axis):
        t = self.t - self.t[0]
        fd = self.fd[:, np.newaxis, np.newaxis]
        full = phasor(t, fd)
        assert full.dtype == self.c_type
        linear = phasor(t, fd, linear_axis=linear_axis)
        assert linear.dtype == self.c_type
        assert_allclose(full, linear, atol=self.atol, rtol=0)

    @pytest.mark.parametrize('linear_axis', (None, -2, "transform"))
    def test_ft_t(self, linear_axis):
        expected = np.fft.ifft(self.dw, axis=0)
        if NUMPY_LT_2_0:
            expected = expected.astype(self.c_type)
        else:
            assert expected.dtype == self.c_type
        # Regular FT is always relative to start of array.
        t = self.t - self.t[0]
        phs = phasor(t, self.fd[:, np.newaxis, np.newaxis],
                     linear_axis=linear_axis)
        assert phs.dtype == self.c_type
        result = np.mean(self.dw*phs, axis=1)
        assert result.dtype == self.c_type
        assert_allclose(result, expected, atol=self.atol, rtol=0)

    @pytest.mark.parametrize('linear_axis', (None, -1, "transform"))
    def test_ft_f(self, linear_axis):
        expected = np.fft.ifft(self.dw, axis=1)
        if NUMPY_LT_2_0:
            expected = expected.astype(self.c_type)
        else:
            assert expected.dtype == self.c_type
        # Regular FT is always relative to start of array.
        f = self.f - self.f[0]
        phs = phasor(f, self.tau[:, np.newaxis, np.newaxis],
                     linear_axis=linear_axis)
        assert phs.dtype == self.c_type
        result = np.mean(self.dw*phs, axis=-1).T
        assert result.dtype == self.c_type
        assert_allclose(result, expected, atol=self.atol, rtol=0)

    @pytest.mark.parametrize('linear_axis_t,linear_axis_f',
                             [(-2, -1), ("transform", "transform")])
    def test_ft(self, linear_axis_t, linear_axis_f):
        expected = np.fft.ifft2(self.dw)
        if NUMPY_LT_2_0:
            expected = expected.astype(self.c_type)
        else:
            assert expected.dtype == self.c_type
        # Regular FT always has phase relative to start of array.
        t = self.t - self.t[0]
        f = self.f - self.f[0]
        phs_t = phasor(t, self.fd[:, np.newaxis, np.newaxis, np.newaxis],
                       linear_axis=linear_axis_t)
        phs_f = phasor(f, self.tau[:, np.newaxis, np.newaxis],
                       linear_axis=linear_axis_f)
        assert phs_t.dtype == self.c_type
        assert phs_f.dtype == self.c_type
        result = np.mean(self.dw*phs_t*phs_f, axis=(-2, -1))
        assert result.dtype == self.c_type
        assert_allclose(result, expected, atol=self.atol, rtol=0)


class TestPhasorFloat32(TestPhasor):
    f_type = np.float32
    c_type = np.complex64
    atol = 2e-5
