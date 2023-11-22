# Licensed under the GPLv3 - see LICENSE
import numpy as np
from astropy import units as u, constants as const


__all__ = ['phasor', 'dynamic_field', 'theta_grid', 'theta_theta_indices']


def expand(*arrays, n=2):
    """Add n unity axes to the end of all arrays."""
    result = [np.reshape(array, np.shape(array)+(1,)*n)
              for array in arrays]
    return result if len(arrays) > 1 else result[0]


def phasor(indep, transform, linear_axis=None):
    """Calculate phase part of a Fourier transform like operation.

    Simply calculates ``exp(j*indep*transform)``, where generally the two
    inputs will be on different dimensions, so that they broadcast against
    each other.  If the independent variable spans a linearly spaced range,
    one can use ``linear_axis`` to speed up the calculation by only calculating
    ``exp(j*indep[0]*transform)`` and ``exp(j*(indep[1]-indep[0])*tranform)``
    and filling the array by cumulative multiplication.

    Parameters
    ----------
    indep : array_like
        Independent variable.
    transform : array_like
        Transformed variable.  If an `~astropy.units.Quantity`, it must have
        the inverse units of ``indep``.  It should *not* include a factor 2pi.
    linear_axis : int, "transform", or None, optional
        Possible axis along which ``indep`` changes by linear steps, and for
        which the calculation can be sped up using cumulative multiplication.
        This will lead to inaccuracies at the 1e-9 level, which should not
        matter for most purposes. If "transform", then take ``transform`` to
        be linearly spaced (along a single axis, e.g., shape ``(n, 1, 1)``).
    """
    if linear_axis is None:
        phasor = 1j * (indep * transform * u.cycle).to_value(u.rad)
        return np.exp(phasor, out=phasor)

    along_transform = (linear_axis == "transform")
    if along_transform:
        linear_axis = transform.shape.index(transform.size) - transform.ndim
    elif linear_axis >= 0:
        linear_axis -= indep.ndim

    extra_slice = (slice(None),) * (-1-linear_axis)
    ph0_index = (Ellipsis, slice(0, 1)) + extra_slice
    ph01_index = (Ellipsis, slice(0, 2)) + extra_slice
    dph0_index = (Ellipsis, slice(1, None)) + extra_slice
    if along_transform:
        ph0 = (indep * transform[ph0_index] * u.cycle).to_value(u.rad)
        dph = (np.diff(transform[ph01_index], axis=linear_axis)
               * indep * u.cycle).to_value(u.rad)
    else:
        ph0 = (indep[ph0_index] * transform * u.cycle).to_value(u.rad)
        dph = (np.diff(indep[ph01_index], axis=linear_axis)
               * transform * u.cycle).to_value(u.rad)
    phasor = np.empty(np.broadcast(indep, transform).shape, complex)
    phasor[ph0_index] = np.exp(1j * ph0)
    phasor[dph0_index] = np.exp(1j * dph)
    return np.cumprod(phasor, out=phasor, axis=linear_axis)


def dynamic_field(theta_par, theta_perp, realization, d_eff, mu_eff, f, t,
                  fast=True):
    """Given a set of scattering points, construct the dynamic wave field.

    Parameters
    ----------
    theta_par : ~astropy.units.Quantity
        Angles of the scattering point in the direction parallel to ``mu_eff``
    theta_perp : ~astropy.units.Quantity
        Angles perpendicular to ``mu_eff``.
    realization : array-like
        Complex amplitudes of the scattering points.  Set to ``1.`` to avoid
        using it.
    d_eff : ~astropy.units.Quantity
        Effective distance.  Should be constant; if different for
        different points, no screen-to-screen scattering is taken into
        account.
    mu_eff : ~astropy.units.Quantity
        Effective proper motion (``v_eff / d_eff``), i.e., parallel to
        ``theta_par``.
    t : ~astropy.units.Quantity
        Times for which the dynamic wave spectrum should be calculated.
        Should broadcast with ``f`` to give the dynamic spectrum shape.
    f : ~astropy.units.frequency
        Frequencies for which the spectrum should be calculated.
        Should broadcast with ``t`` to give the dynamic spectrum shape.
    fast : bool
        Calculates the field faster by iteratively applying a phasor for each
        the frequency step along the frequency axis. Assumes the frequencies
        are a linear sequence.  Will lead to inaccuracies at the 1e-9 level,
        which should be negligible for most purposes.

    Returns
    -------
    dynwave : array
        Delayed wave field array, with time and frequency axes as given by
        ``t`` and ``f``, and earlier axes as given by the other parameters.
    """
    ds_ndim = np.broadcast(f, t).ndim
    theta_par, theta_perp, realization, d_eff, mu_eff = expand(
        theta_par, theta_perp, realization, d_eff, mu_eff, n=ds_ndim)
    th_par = theta_par + mu_eff * t
    tau_t = (((d_eff / (2*const.c)) * (th_par**2 + theta_perp**2))
             .to(u.s, u.dimensionless_angles()))
    result = phasor(f, tau_t, linear_axis=(f.shape.index(f.size)-f.ndim
                                           if fast else None))
    if np.any(realization != 1.):
        result *= realization
    return result


def theta_theta_indices(theta, lower=-0.25, upper=0.75):
    """Indices to pairs of angles within bounds.

    Select pairs theta0, theta1 for which theta1 is constrained to lie
    around theta0 to within (lower*theta0, upper*theta0).

    Here, ``lower=-1, upper=1`` would select all non-duplicate pairs that
    are not on the diagonals with theta1 = ±theta0 (i.e., like
    ``np.triu_indices(theta.size, k=1)``, but also excluding the
    cross-diagonal; ``upper=1+epsilon`` would be identical).  But using
    all off-diagonal points gives too many poor constraints.

    The defaults instead select points nearer the top of the inverted
    arclets, extending further on the outside than on the inside, since
    on the inside all arclets crowd together.

    Parameters
    ----------
    theta : `~astropy.units.Quantity`
        Grid of angles.
    lower, upper : float
        Range of angles for which indices should be returned.
    """
    indgrid = np.indices((len(theta), len(theta)))
    sel = (np.abs(theta[:, np.newaxis] + (upper+lower)/2*theta)
           < (upper-lower)/2 * np.abs(theta))
    return indgrid[1][sel], indgrid[0][sel]


def theta_grid(d_eff, mu_eff, fobs, dtau, tau_max, dfd, fd_max):
    """Make a grid of theta that sample the parabola in a particular way.

    The idea would be to impose the constraint that near tau_max the
    spacing is roughly the spacing allowed by the frequencies, and
    near the origin that allowed by the times.  In practice, one needs
    to oversample in both directions, by about a factor 1.3 in tau (which
    makes some sense from wanting to sample resolution elements with about
    3 points, not just 2 for a FFT), but by about a factor 1.6 in f_d,
    which is less clear.

    Parameters
    ----------
    d_eff : `~astropy.units.Quantity`
        Effective distance.  Should be constant; if different for
        different points, no screen-to-screen scattering is taken into
        account.
    mu_eff : `~astropy.units.Quantity`
        Effective proper motion (``v_eff / d_eff``), parallel to ``theta_par``.
    fobs : `~astropy.units.Quantity`
        Mean frequency for which the doppler factor should be calculated.
    dtau : `~astropy.units.Quantity`
        Requested spacing in delay (typically should somewhat oversample the
        spacing in the secondary spectrum).
    tau_max : `~astropy.units.Quantity`
        Maximum delay to consider.  Can be up to the value implied by the
        the frequency resolution (i.e., ``1/(f[2]-f[0])``).
    dfd : `~astropy.units.Quantity`
        Requested spacing in doppler factor (typically should oversample the
        spacing in the secondary spectrum).
    fd_max : `~astropy.units.Quantity`
        Maximum doppler factor to consider.  Can be up to the value implied
        by the time resolution (i.e., ``1/(t[2]-t[0])``).
    """
    tau_factor = d_eff/(2.*const.c)
    fd_factor = d_eff*mu_eff*fobs/const.c
    # Curvature in physical units.
    a = tau_factor / fd_factor**2
    # Consider relevant maximum.
    fd_max = min(fd_max, np.sqrt(tau_max/a).to(
        fd_max.unit, u.dimensionless_angles()))
    # Curvature in sample units.
    a_pix = (a * dfd**2 / dtau).to_value(
        1, equivalencies=u.dimensionless_angles())
    x_max = (fd_max/dfd).to_value(1)
    x = sample_parabola(x_max, a_pix)
    th_r = (x*dfd/fd_factor).to(u.mas, u.dimensionless_angles())
    return th_r


def sample_parabola(x_max, a=1.):
    """Solve for the x that evenly sample a parabola.

    The points will have spacing of 1 along the parabola (in units of x).

    Parameters
    ----------
    x_max : float
        Maximum x value to extend to.
    """
    s_max = np.round(path_length(x_max, a))
    # Corresponding path length around a parabola
    s = np.arange(1, s_max+1)
    # Initial guesses for x.
    x = np.linspace(1, x_max, s.size)
    d_s = s - path_length(x, a)
    it = 0
    while np.any(np.abs(d_s) > 1e-6) and it < 100:
        dsdx = path_length(x, a, derivative=True)
        x = x + d_s / dsdx
        d_s = s - path_length(x, a)
        it += 1

    return np.hstack([-x[::-1], 0, x[:]])


def path_length(x, a=1, derivative=False):
    r"""Path length along a parabola, measured from the origin.

    For a parabola :math:`y=ax^2`::

    .. math::
        \int ds &= \int \sqrt{dx^2+dy^2} = \int \sqrt{1+(2ax)^2} dx \\
                &= \frac{1}{4a}\left(\asinh(2ax) + x\sqrt{1+(2ax)^2}\right)

    Parameters
    ----------
    x : array-like
        X position to evaluate the path length for.
    a : float, optional
        Curvature (y = a*x**2).  Default: 1.
    derivative : bool
        If true, return ds/dx rather than s.
    """
    x = 2 * a * x
    sq = np.sqrt(1+x**2)
    if derivative:
        return sq
    else:
        return (np.arcsinh(x) + x * sq) / (4*a)
