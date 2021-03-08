import numpy as np
from astropy import units as u, constants as const
from astropy.coordinates import (
    CartesianRepresentation,
    CylindricalRepresentation)
import pytest

from screens.screen import Screen, Screen1D, ZHAT


def repr_isclose(r1, r2, atol=0, rtol=0):
    norm = (r1-r2).norm()
    return u.isclose(norm, 0*norm.unit, atol=atol, rtol=rtol)


def repr_allclose(r1, r2, atol=0, rtol=0):
    return np.all(repr_isclose(r1, r2, atol=atol, rtol=rtol))


class TestSimpleScreen:
    def setup_class(self):
        self.d_psr = 1*u.kpc
        self.d_scr = 0.5*u.kpc
        self.pulsar = Screen(CartesianRepresentation(0, 0, 0, unit='m'),
                             CartesianRepresentation(100., 0, 0, unit='km/s'))
        self.screen = Screen(
            CartesianRepresentation([0., 10.], [0., 0.], [0., 0.], unit='AU'),
            CartesianRepresentation(0., 0., 0., unit='km/s'),
            magnification=np.array([1., 0.5j]))
        self.telescope = Screen(CartesianRepresentation(0, 0, 0, unit='m'),
                                CartesianRepresentation(0, 0, 0, unit='km/s'))

    def test_observe_pulsar(self):
        obs = self.telescope.observe(self.pulsar, distance=self.d_psr)
        assert obs.brightness.shape == ()
        assert obs.brightness == 1.
        assert obs.tau == 0.
        assert obs.taudot == 0.

    def test_screen_pulsar(self):
        d_rel = self.d_psr - self.d_scr
        screened = self.screen.observe(self.pulsar, distance=d_rel)
        assert screened.brightness.shape == (2,)
        assert np.all(screened.brightness == self.screen.brightness)
        theta = ([0, -10]*u.AU)/d_rel
        tau_expected = theta**2 * 0.5 * d_rel / const.c
        assert u.allclose(screened.tau, tau_expected)
        taudot_expected = 100*u.km/u.s * theta / const.c
        assert u.allclose(screened.taudot, taudot_expected)

    def test_observe_screened_pulsar(self):
        d_rel = self.d_psr - self.d_scr
        screened = self.screen.observe(self.pulsar, distance=d_rel)
        obs = self.telescope.observe(screened, distance=self.d_scr)
        assert obs.brightness.shape == (2,)
        assert np.all(obs.brightness == self.screen.brightness)
        d_eff = self.d_psr * self.d_scr / d_rel
        v_eff = self.d_scr / d_rel * 100 * u.km/u.s
        theta = ([0, -10]*u.AU) / self.d_scr
        tau_expected = theta**2 * 0.5 * d_eff / const.c
        assert u.allclose(obs.tau, tau_expected)
        taudot_expected = v_eff * theta / const.c
        assert u.allclose(obs.taudot, taudot_expected)


class TestLinearScreen:
    def setup_class(self):
        self.d_psr = 1.5*u.kpc
        self.pulsar = Screen(CartesianRepresentation(0, 0, 0, unit='m'),
                             CartesianRepresentation(100., 0, 0, unit='km/s'))
        self.telescope = Screen(CartesianRepresentation(0, 0, 0, unit='m'),
                                CartesianRepresentation(0, 0, 0, unit='km/s'))

    @pytest.mark.parametrize('angle', [60*u.deg, 135*u.deg])
    def test_observe_screened_pulsar(self, angle):
        d_scr = 0.5*u.kpc
        d_rel = self.d_psr - d_scr
        screen = Screen1D(
            CylindricalRepresentation(1., angle, 0.).to_cartesian(),
            [1, 2]*u.AU, magnification=0.05)
        screened = screen.observe(self.pulsar, distance=d_rel)
        obs = self.telescope.observe(screened, distance=d_scr)
        assert obs.tau.shape == (2,)
        assert repr_allclose(obs.source.pos, screen.p * screen.normal,
                             atol=1*u.km)
        assert obs.brightness == screen.magnification
        d_eff = self.d_psr * d_scr / d_rel
        # Pulsar moving towards screen, so v_eff is negative.
        v_eff = d_scr / d_rel * -100 * u.km/u.s * np.cos(angle)
        theta = screen.p / d_scr
        tau_expected = theta**2 * 0.5 * d_eff / const.c
        assert u.allclose(obs.tau, tau_expected)
        taudot_expected = v_eff * theta / const.c
        assert u.allclose(obs.taudot, taudot_expected)

        assert u.allclose(obs.source.sigma, 0., atol=1e-10)
        mu_psr = -100 * u.km/u.s / self.d_psr
        assert u.allclose(obs.source.sigma_dot, mu_psr * np.sin(angle))

    def test_observe_doubly_screened_pulsar_parallel(self):
        r"""Test with two linear screens which are exactly parallel.

        Hence light ray like  *      o
                               \    /
                                +--+
        """
        d_s1 = 0.5*u.kpc
        d_s2 = 1.0*u.kpc
        shared_normal = CartesianRepresentation(1., 0., 0., unit=u.one)
        shared_offset = [1]*u.AU
        s1 = Screen1D(shared_normal, shared_offset, magnification=0.5)
        s2 = Screen1D(shared_normal, shared_offset, magnification=0.5)
        obs = self.telescope.observe(
            s1.observe(
                s2.observe(self.pulsar, distance=self.d_psr-d_s2),
                distance=d_s2-d_s1),
            distance=d_s1)
        assert obs.tau.shape == (1, 1)
        assert repr_allclose(obs.source.pos, s1.p * shared_normal, atol=1*u.km)
        assert repr_allclose(obs.source.source.pos,
                             s2.p * shared_normal, atol=1*u.km)
        assert obs.brightness == s1.magnification * s2.magnification
        tau_o1 = 0.5 * (shared_offset/d_s1)**2 * d_s1 / const.c
        tau_2p = 0.5 * (shared_offset
                        / (self.d_psr-d_s2))**2 * (self.d_psr-d_s2) / const.c
        tau_expected = tau_o1+tau_2p
        assert u.allclose(obs.tau, tau_expected)
        # Pulsar moving towards s2, so v_eff is negative.
        taudot_expected = (shared_offset / (self.d_psr-d_s2) *
                           -100 * u.km/u.s / const.c)
        assert u.allclose(obs.taudot, taudot_expected)

    def test_observe_doubly_screened_pulsar_perpendicular(self):
        r"""Test with two linear screens which are exactly perpendicular.

        Hence solution is relatively obvious: offset along line at
        each screen exactly halfway the distance to the line on the
        other screen.

        """
        d_s1 = 0.5*u.kpc
        d_s2 = 1.0*u.kpc
        shared_offset = [1]*u.AU
        s1 = Screen1D(CartesianRepresentation(0., 1., 0., unit=u.one),
                      shared_offset, magnification=0.5)
        s2 = Screen1D(CartesianRepresentation(1., 0., 0., unit=u.one),
                      shared_offset, magnification=0.5)
        obs = self.telescope.observe(
            s1.observe(
                s2.observe(self.pulsar, distance=self.d_psr-d_s2),
                distance=d_s2-d_s1),
            distance=d_s1)
        assert obs.tau.shape == (1, 1)
        assert repr_allclose(
            obs.source.pos,
            shared_offset * CartesianRepresentation(0.5, 1., 0., unit=u.one),
            atol=1*u.km)
        assert repr_allclose(
            obs.source.source.pos,
            shared_offset * CartesianRepresentation(1., 0.5, 0., unit=u.one),
            atol=1*u.km)
        assert obs.brightness == s1.magnification * s2.magnification
        tau_o1 = 0.5 * (obs.source.pos/d_s1).norm()**2 * d_s1 / const.c
        tau_12 = 0.5 * ((obs.source.source.pos - obs.source.pos)
                        / (d_s2-d_s1)).norm()**2 * (d_s2-d_s1) / const.c
        tau_2p = (0.5 * (obs.source.source.pos
                         / (self.d_psr-d_s2)).norm()**2 * (self.d_psr-d_s2)
                  / const.c)
        tau_expected = tau_o1 + tau_12 + tau_2p
        assert u.allclose(obs.tau, tau_expected)
        # Pulsar moving towards s2, so v_eff is negative.
        taudot_expected = (shared_offset / (self.d_psr-d_s2) *
                           -100 * u.km/u.s / const.c)
        assert u.allclose(obs.taudot, taudot_expected)

    def test_observe_doubly_screened_pulsar(self):
        """More points, more random orientations."""
        d1 = 0.5*u.kpc
        d2 = 1.0*u.kpc
        dp = 1.5*u.kpc
        pulsar = Screen(
            pos=CartesianRepresentation([0., 1., 0.]*u.AU),
            vel=CartesianRepresentation(300., 0., 0., unit=u.km/u.s))
        telescope = Screen(CartesianRepresentation([0., 0.5, 0.]*u.AU))
        s1 = Screen1D(
            CylindricalRepresentation(1., -40*u.deg, 0.).to_cartesian(),
            [-0.711, -0.62, -0.53, -0.304, -0.111, -0.052, -0.031,
             0., 0.0201, 0.0514, 0.102, 0.199, 0.3001, 0.409]*u.AU,
            magnification=np.array(
                [0.01, 0.01, 0.02, 0.08, 0.25j, 0.34, 0.4+.1j,
                 1, 0.2-.5j, 0.5j, 0.3, 0.2, 0.09, 0.02]))
        s2 = Screen1D(
            CylindricalRepresentation(1., 70*u.deg, 0.).to_cartesian(),
            [0.85, 1.7]*u.AU, magnification=0.05)
        obs = telescope.observe(
            s1.observe(
                s2.observe(pulsar, distance=dp-d2),
                distance=d2-d1),
            distance=d1)
        # Calculates positions as well.
        tau = obs.tau
        assert tau.shape == (14, 2)

        tpos = obs.pos
        scat1 = obs.source.pos
        scat2 = obs.source.source.pos
        ppos = obs.source.source.source.pos
        ds1t = (ZHAT + (scat1-tpos)/d1)
        ds21 = (ZHAT + (scat2-scat1)/(d2-d1))
        dps2 = (ZHAT + (ppos-scat2)/(dp-d2))
        rthat = tpos / tpos.norm()
        uthat = ZHAT.cross(rthat).ravel()
        r1hat = obs.source.normal
        u1hat = ZHAT.cross(r1hat)
        r2hat = obs.source.source.normal
        u2hat = ZHAT.cross(r2hat)
        # Check that no bending happens along line.
        assert u.allclose(ds1t.dot(u1hat), ds21.dot(u1hat))
        assert u.allclose(ds21.dot(u2hat), dps2.dot(u2hat))
        # Check direction from observer to first point is set right.
        # Alpha angle along telescope to origin line.
        assert u.allclose(obs.alpha, -ds1t.dot(rthat))
        assert u.allclose(obs.sigma, ds1t.dot(uthat))
        # Check locations along lines.
        assert repr_allclose(obs.source.p * r1hat
                             + obs.source.sigma * d1 * u1hat,
                             obs.source.pos, atol=1*u.km)
        assert repr_allclose(obs.source.source.p * r2hat
                             + obs.source.source.sigma * d2 * u2hat,
                             obs.source.source.pos, atol=1*u.km)
        # Check bending angles.
        with u.add_enabled_equivalencies(u.dimensionless_angles()):
            assert u.allclose(np.abs(obs.source.alpha),
                              np.arcsin(ds1t.cross(ds21).norm()
                                        / ds1t.norm()/ds21.norm()))
            assert u.allclose(np.abs(obs.source.source.alpha),
                              np.arcsin(ds21.cross(dps2).norm()
                                        / ds21.norm()/dps2.norm()))
