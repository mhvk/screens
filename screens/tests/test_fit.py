"""Testing of the fit routines.

TODO: make this more complete!
"""
import numpy as np
from astropy import units as u

from screens.fields import dynamic_field
from screens.dynspec import DynamicSpectrum


class TestFit:
    def setup_class(cls):
        cls.fobs = 316 * u.MHz
        f = (np.arange(-16, 16)/2.) << u.MHz
        cls.f = cls.fobs + f
        cls.t = (np.arange(-16, 16) << u.minute)[:, np.newaxis]
        cls.d_eff = 1*u.kpc
        cls.mu_eff = 30*u.mas/u.yr
        cls.theta = [0., -0.3, 0.5] << u.mas
        cls.dw = dynamic_field(cls.theta, 0., 1.,
                               cls.d_eff, cls.mu_eff, cls.f, cls.t)
        cls.magnification = np.array([1., 0.2, 0.1j])
        ds = np.abs((cls.dw * cls.magnification[:, np.newaxis, np.newaxis])
                    .sum(0))**2
        cls.ds = DynamicSpectrum(ds, cls.f, cls.t, 0.001,
                                 d_eff=cls.d_eff, mu_eff=cls.mu_eff,
                                 magnification=cls.magnification)
        cls.ds.theta = cls.theta

    def test_jacobian(self):
        """Test derivatives d DS / d mag (real,imag) and d DS / d mu_eff."""
        jac_mag, jac_mu = self.ds.jacobian(self.magnification, self.mu_eff)
        dmag = 1e-6
        for real in True, False:
            for i in range(self.magnification.size):
                magnification = self.magnification.copy()
                magnification[i] += dmag * (1 if real else 1j)
                ds = np.abs((magnification[:, np.newaxis, np.newaxis]
                             * self.dw).sum(0))**2
                ddsdmag = (ds - self.ds.dynspec).reshape(ds.size) / dmag
                jac = jac_mag['mag_real' if real else 'mag_imag'][:, i]
                # Typical numbers are order unity.
                assert np.allclose(ddsdmag, jac, atol=1e-6)

        dmu = self.mu_eff * 1e-6
        mu_eff = self.mu_eff + dmu
        ds = np.abs((self.magnification[:, np.newaxis, np.newaxis]
                     * dynamic_field(self.theta, 0., 1.,
                                     self.d_eff, mu_eff, self.f, self.t))
                    .sum(0))**2
        ddsdmu = (ds - self.ds.dynspec).reshape(ds.size) / dmu
        # Typical numbers are of order 0.01 yr/mas.
        assert u.allclose(ddsdmu, jac_mu[:, 0], atol=1e-7*u.yr/u.mas)
