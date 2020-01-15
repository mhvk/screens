import numpy as np
from astropy import units as u, constants as const


def expand2(*arrays):
    """Add two unity axes to the end of all arrays."""
    return [np.reshape(array, np.shape(array)+(1, 1))
            for array in arrays]


def dynamic_field(theta_par, theta_perp, realization, d_eff, mu_eff, f, t,
                  fast=True):
    """Given a set of scattering points, construct the dynamic wave field.

    Parameters
    ----------
    theta_par : ~astropy.units.Quantity
        Angles of the scattering point in the direction parallel to ``mu_eff``
    theta_perp : ~astropy.units.Quantity
        Angles perpendiculat to ``mu_eff``.
    realization : array-like
        Complex amplitudes of the scattering points
    d_eff : ~astropy.units.Quantity
        Effective distance.  Should be constant; if different for
        different points, no screen-to-screen scattering is taken into
        account.
    mu_eff : ~astropy.units.Quantity
        Effective proper motion (``v_eff / d_eff``), parallel to ``theta_par``.
    t : ~astropy.units.Quantity
        Times for which the dynamic wave spectrum should be calculated.
    f : ~astropy.units.frequency
        Frequencies for which the spectrum should be calculated.
    fast : bool
        Calculates the field faster by iteratively applying a phasor for each
        the frequency step along the frequency axis. Assumes the frequencies
        are a linear sequence.  Will lead to inaccuracies at the 1e-9 level,
        which should be negligible for most purposes.

    Returns
    -------
    dynwave : array
        Delayed wave field array, with last axis time, second but last
        frequency, and earlier axes as given by the other parameters.
    """
    theta_par, theta_perp, realization, d_eff, mu_eff = expand2(
        theta_par, theta_perp, realization, d_eff, mu_eff)
    th_par = theta_par + mu_eff * t
    tau_t = (d_eff / (2*const.c)) * (th_par**2 + theta_perp**2)
    phasor = np.empty(np.broadcast(tau_t, f[:, np.newaxis]).shape, complex)
    if fast:
        phase0 = (f[0] * u.cycle * tau_t).to_value(
            u.one, u.dimensionless_angles())
        dphase = ((f[1]-f[0]) * u.cycle * tau_t).to_value(
            u.one, u.dimensionless_angles())
        phasor[..., :1, :] = np.exp(-1j * phase0)
        phasor[..., 1:, :] = np.exp(-1j * dphase)
        phasor = np.cumprod(phasor, out=phasor, axis=-2)
    else:
        phasor.imag = (-f[:, np.newaxis] * u.cycle * tau_t).to_value(
            u.one, u.dimensionless_angles())
        phasor = np.exp(phasor, out=phasor)

    if np.any(realization != 1.):
        phasor *= realization
    return phasor


def theta_theta_indices(theta, lower=-0.25, upper=0.75):
    """Indices to pairs of angles within bounds.

    Select pairs theta0, theta1 for which theta is constraint to lie
    around theta=0 to within (lower*theta0, upper*theta0).

    Here, ``lower=-1, upper=1`` would select all non-duplicate pairs that
    are not on the diagonals with theta1 = +-theta0 (i.e., like
    ``np.triu_indices(theta.size, k=1)``, but also excluding the
    cross-diagonal; ``upper=1+epsilon`` would be identical).  But using
    all off-diagonal points gives too many poor constraints.

    The defaults instead select points nearer the top of the inverted
    arclets, extending further on the outside than on the inside, since
    on the inside all arclets crowd together.
    """
    indgrid = np.indices((len(theta), len(theta)))
    sel = (np.abs(theta[:, np.newaxis] + (upper+lower)/2*theta)
           < (upper-lower)/2 * np.abs(theta))
    return indgrid[1][sel], indgrid[0][sel]


def theta_theta(theta, d_eff, mu_eff, dynspec, f, t):
    """Theta-theta plot for the given theta and dynamic spectrum.

    Uses ``theta_theta_indices`` to determine which pairs to
    include in the theta-theta array, and then brute-force maps those
    by estimating the amplitude at each pair by cross-multiplying their
    expected signature in the dynamic spectrum.
    """
    dynwave = dynamic_field(theta, 0, 1., d_eff, mu_eff, f, t)
    # Get intensities by brute-force mapping:
    # dynspec * dynwave[j] * dynwave[i].conj() / sqrt(2) for all j > i
    # Do first product ahead of time to speed up calculation
    # (remove constant parts of input spectrum to eliminate edge effects)
    ddyn = (dynspec - dynspec.mean()) * dynwave
    # Explicit loop is faster than just broadcasting or using indices
    # for advanced indexing, since it avoids creating a large array.
    result = np.zeros(theta.shape * 2, ddyn.dtype)
    indices = theta_theta_indices(theta)
    for i, j in zip(*indices):
        amplitude = (ddyn[j] * dynwave[i].conj()).mean((-2, -1)) / np.sqrt(2.)
        result[i, j] = amplitude
        result[j, i] = amplitude.conj()

    return result


def theta_grid(d_eff, mu_eff, f, t, tau_max=None, oversample=1.4):
    """Make a grid of theta that sample the parabola roughly uniformly.

    With the constraint that near tau_max, the spacing is
    roughly the spacing allowed by the frequencies.

    Parameters
    ----------
    d_eff : ~astropy.units.Quantity
        Effective distance.  Should be constant; if different for
        different points, no screen-to-screen scattering is taken into
        account.
    mu_eff : ~astropy.units.Quantity
        Effective proper motion (``v_eff / d_eff``), parallel to ``theta_par``.
    t : ~astropy.units.Quantity
        Times for which the dynamic wave spectrum should be calculated.
    f : ~astropy.units.frequency
        Frequencies for which the spectrum should be calculated.
    tau_max : ~astropy.units.Quantity
        Maximum delay to consider.  If not given, taken as the value
        implied by the frequency resolution (i.e., ``1/(f[2]-f[0])``).
    oversample : float
        Factor by which to oversample pixels.  This is a very finicky
        number.  With 1, dynamic spectra seem to be underfit, with
        1.5 fitting takes very long as points are strongly correlated.
    """
    fobs = f.mean()
    tau_factor = d_eff/(2.*const.c)
    fd_factor = d_eff*mu_eff*fobs/const.c
    dtau = (1./oversample/f.ptp()).to(u.us)
    dfd = (1./oversample/t.ptp()).to(u.mHz)
    a_pix = (tau_factor/dtau * (dfd/fd_factor)**2).to_value(
        1, equivalencies=u.dimensionless_angles())
    if tau_max is None:
        tau_max = 1/(f[2]-f[0])
    n_th_by_2 = round((tau_max/dtau).to_value(1))
    s = np.arange(-n_th_by_2, n_th_by_2)
    # Roughly equal spaced along parabola.
    y = (np.sqrt(2.5*np.abs(a_pix*s)+1)-1)**2
    th_r = (np.sqrt(y/y.max()*tau_max/tau_factor) * np.sign(s)).to(
        u.mas, u.dimensionless_angles())
    return th_r


def clean_theta_theta(theta_theta, k=1, clean_cross=True):
    if k > 1:
        theta_theta = np.triu(theta_theta, k=k) + np.tril(theta_theta, k=-k)
    if clean_cross:
        i = np.arange(theta_theta.shape[0]-1)
        theta_theta[theta_theta.shape[0]-1-i, i+1] = 0
    return theta_theta
