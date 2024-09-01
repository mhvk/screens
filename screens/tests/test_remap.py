# Licensed under the GPLv3 - see LICENSE
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u

from screens.remap import remap_time, lincover
from screens import DynamicSpectrum


def test_lincover():
    a = np.array([1.1, 0., 1.05, -0.1])
    out = lincover(a, 10)
    expected = np.linspace(-0.1+1.2/20, 1.1-1.2/20, 10)
    assert_allclose(out, expected)


class TestRemapTime:
    pb = 0.1022515592973 * u.day  # Double pulsar

    @classmethod
    def position(cls, t, *, a=0.5, delta=0., p=None):
        """Model position along screen."""
        if p is None:
            p = cls.pb
        phase = (t-t.mean())/p
        return np.cos(phase*u.cy) + a*phase

    @staticmethod
    def scint(pos, scale=3.):
        """Super simple interference pattern."""
        return (np.cos(pos*scale*u.cy)**2).value

    @classmethod
    def setup_class(self):
        dt = 10*u.s
        n = int(round(((2 * self.pb) / dt).to_value(u.one)))
        nf = 51
        nover = 5
        ta = np.linspace(0*self.pb, 2*self.pb, n*nover,
                         endpoint=False).reshape(-1, nover)
        # Cyclical position + slope
        pos = self.position(ta)
        f = np.linspace(1, 1.3, nf, endpoint=False) << u.GHz
        self.scale = 3 * f.to_value(u.GHz)
        ds = self.scint(pos[..., np.newaxis], scale=self.scale)
        self.ds = DynamicSpectrum(ds.mean(1), t=ta.mean(1).to(u.min), f=f)
        self.pos = pos.mean(1)
        self.new_pos = (np.arange(0, 100) + 0.5) / 100.

    def test_single_frequency(self):
        remapped, weight = remap_time(self.ds.dynspec[:, 0], self.pos, self.new_pos)
        out = remapped / weight
        expected = self.scint(self.new_pos, self.scale[0])
        assert_allclose(out, expected, atol=0.015)

    def test_all_frequencies(self):
        remapped, weight = remap_time(self.ds.dynspec, self.pos, self.new_pos)
        out = remapped / weight
        expected = self.scint(self.new_pos[:, np.newaxis], self.scale)
        assert_allclose(out, expected, atol=0.015)

    def test_remap_nut(self):
        map_pos = self.pos[:, np.newaxis] * (self.ds.f / self.ds.f[0])
        remapped, weight = remap_time(self.ds.dynspec, map_pos, self.new_pos)
        out = remapped / weight
        expected = self.scint(self.new_pos, self.scale[0])
        expected = np.broadcast_to(expected[:, np.newaxis], out.shape)
        assert_allclose(out, expected, atol=0.015)


class TestRemapTimeComplex(TestRemapTime):
    @staticmethod
    def scint(pos, scale=3.):
        """Super simple interference pattern."""
        return np.exp(1j*(pos*scale*u.cy).to_value(u.rad))
