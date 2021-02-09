import numpy as np
from astropy import units as u, constants as const
from astropy.coordinates import CartesianRepresentation

from screens.screen import Screen


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
