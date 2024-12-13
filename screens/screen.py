# Licensed under the GPLv3 - see LICENSE
"""
Representations for sources, screens, and telescopes.

.. warning:: experimental, i.e., API likely to change.  For an example,
    see :doc:`tutorials/two_screens`
"""
import operator

import numpy as np
from astropy import constants as const, units as u
from astropy.coordinates import (
    CartesianRepresentation, UnitSphericalRepresentation)
from astropy.utils.shapes import ShapedLikeNDArray
from astropy.utils.decorators import lazyproperty


__all__ = ['Source', 'Screen', 'Telescope', 'Screen1D']


ZERO_POSITION = CartesianRepresentation(0., 0., 0., unit=u.AU)
ZERO_VELOCITY = CartesianRepresentation(0., 0., 0., unit=u.km/u.s)
ZHAT = CartesianRepresentation(0., 0., 1., unit=u.one)


class Source(ShapedLikeNDArray):
    """Source of radiation at a given position and velocity.

    Parameters
    ----------
    pos : `~astropy.coordinates.CartesianRepresentation`
        Position of the source.  Should not include the distance.
    vel : `~astropy.coordinates.CartesianRepresentation`
        Corresponding velocities.
    magnification : array-like
        Brightness of the points.  Can be complex.
    """
    _shaped_attrs = ('pos', 'vel', 'magnification')

    def __init__(self, pos=ZERO_POSITION, vel=ZERO_VELOCITY, magnification=1.):
        self.pos = pos
        self.vel = vel
        self.magnification = np.asanyarray(magnification)

    def _repr_lines(self):
        lines = [f"<{self.__class__.__name__}"]
        for attr in self._shaped_attrs:
            val = getattr(self, attr, None)
            if val is not None:
                lines.append(f"    {attr}={val},")

        lines[-1] = lines[-1][:-1] + '>'
        return lines

    def __repr__(self):
        return '\n'.join(self._repr_lines())

    @lazyproperty
    def shape(self):
        return np.broadcast(*[
            np.empty(getattr(getattr(self, attr), 'shape', ()))
            for attr in self._shaped_attrs]).shape

    def _apply(self, method, *args, **kwargs):
        if not callable(method):
            method = operator.methodcaller(method, *args, **kwargs)

        new = super().__new__(self.__class__)
        for attr in self._shaped_attrs:
            value = getattr(self, attr)
            if getattr(value, 'shape', ()) != ():
                value = method(value)
            setattr(new, attr, value)

        return new

    @property
    def brightness(self):
        """Brightness of each path."""
        return self._paths[0]

    @property
    def tau(self):
        """Delay for each path."""
        return self._paths[1]

    @property
    def taudot(self):
        """Time derivative of the delay for each path."""
        return self._paths[2]

    def _rel_posvel(self, other):
        return self.pos - other.pos, self.vel - other.vel

    @lazyproperty
    def _paths(self):
        return self.magnification, 0*u.us, 0*u.us/u.s


class Screen(Source):
    """Screen passing through a source of radiation.

    Parameters
    ----------
    pos : `~astropy.coordinates.CartesianRepresentation`
        Position of the points, ignoring the distance.
    vel : `~astropy.coordinates.CartesianRepresentation`
        Corresponding velocities.
    magnification : array-like
        Magnification of the points.  Usually complex.
    source : `~screens.screen.Source` or `~screens.screen.Screen`, optional
        Possible source illuminating this screen.  Unless specific broadcasting
        is required, it is recommended to use the `Screen.observe` method.
    distance : `~astropy.units.Quantity`, optional
        Possible distance from the source.  Only useful if ``source`` is given.
    """
    def __init__(self, pos=ZERO_POSITION, vel=ZERO_VELOCITY, magnification=1.,
                 source=None, distance=None):
        super().__init__(pos=pos, vel=vel, magnification=magnification)
        self.source = source
        self.distance = distance

    def _repr_lines(self):
        lines = super()._repr_lines()
        if self.source is not None:
            lines[-1] = lines[-1][:-1] + ','
            source_lines = self.source._repr_lines()
            lines.append(f"    source={source_lines[0]}")
            lines.extend([f"    {ln}" for ln in source_lines[1:]])
            lines[-1] += ','
            lines.append(f"    distance={self.distance}>")
        return lines

    def _apply(self, method, *args, **kwargs):
        new = super()._apply(method, *args, **kwargs)
        new.source = self.source
        new.distance = self.distance
        return new

    @lazyproperty
    def _paths(self):
        if self.source is None:
            raise ValueError('can only calculate paths if ``source`` is set.')
        source = self.source
        distance = self.distance
        rel_pos, rel_vel = source._rel_posvel(self)
        rel_xy = rel_pos.get_xyz(xyz_axis=-1)[..., :2]
        rel_vxy = rel_vel.get_xyz(xyz_axis=-1)[..., :2]
        theta = rel_xy / distance
        tau = source.tau + distance / const.c * 0.5 * (theta**2).sum(-1)
        taudot = source.taudot + (theta * rel_vxy).sum(-1) / const.c
        brightness = source.brightness * self.magnification
        return brightness, tau, taudot

    def observe(self, source, distance):
        new = self[(Ellipsis,)+(np.newaxis,)*source.ndim]
        new.source = source
        new.distance = distance
        return new


class Telescope(Screen):
    """Telescope detecting a source of radiation.

    Parameters
    ----------
    pos : `~astropy.coordinates.CartesianRepresentation`
        Position of the telescope, ignoring the distance.
    vel : `~astropy.coordinates.CartesianRepresentation`
        Corresponding velocity.
    magnification : array-like
        Magnification of the telescope.  Usually unity.
    source : `~screens.screen.Source` or `~screens.screen.Screen`, optional
        Possible source illuminating this screen.  Unless specific broadcasting
        is required, it is recommended to use the `Screen.observe` method.
    distance : `~astropy.units.Quantity`, optional
        Possible distance from the source.  Only useful if ``source`` is given.
    """
    pass


class Screen1D(Screen):
    """One-dimensional screen.

    The assumption is that all scattering points are on lines, which represent
    places where light can be bent perpendicular to the line, such that it
    reaches the observer.  The positions of the scattering ponts along the
    lines then depends on where the source and detector are.

    Parameters
    ----------
    normal : `~astropy.coordinates.CartesianRepresentation`
        Unit vector towards the line.  Should not include a z component, i.e.,
        be perpendicular to both the line and the z axis.
    p : `~astropy.units.Quantity`
        Separations of the lines from the origin, along the normal.
    v : `~astropy.units.Quantity`
        Velocities of the lines along the normal.
    magnification : array-like
        Magnification of scattering points for the lines.  Can be complex.
    source : `~screens.screen.Source` or `~screens.screen.Screen`, optional
        Possible source illuminating this screen.  Unless specific broadcasting
        is required, it is recommended to use the `Screen.observe` method.
    distance : `~astropy.units.Quantity`, optional
        Possible distance from the source.  Only useful if ``source`` is given.

    """
    _shaped_attrs = ('normal', 'p', 'v', 'magnification')

    def __init__(self, normal, p=0*u.AU, v=0*u.km/u.s, magnification=1.,
                 source=None, distance=None):
        super().__init__(pos=None, vel=None,
                         magnification=magnification,
                         source=source, distance=distance)
        self.normal = normal
        self.p = p
        self.v = v

    def _apply(self, method, *args, **kwargs):
        new = super()._apply(method, *args, **kwargs)
        new.pos = None
        new.vel = None
        return new

    @staticmethod
    def _unit_vector(c):
        return c.represent_as(UnitSphericalRepresentation).to_cartesian()

    def _solve_positions(self, other):
        assert not isinstance(other, Screen1D)
        assert self.source is not None
        # Setup arrays.
        sources = [other]
        distances = [0*other.distance, other.distance]
        poss = [other.pos]
        vels = [other.vel]
        rhats = [self._unit_vector(other.pos)]
        source = self
        while isinstance(source, Screen1D):
            sources.append(source)
            distances.append(source.distance)
            poss.append(source.p * source.normal)
            vels.append(source.v * source.normal)
            rhats.append(source.normal)
            source = source.source
        assert isinstance(source, Source)
        sources.append(source)
        poss.append(source.pos)
        vels.append(source.vel)
        rhats.append(self._unit_vector(source.pos))

        distances = u.Quantity(distances).cumsum()
        uhats = [ZHAT.cross(rhat) for rhat in rhats]

        n = len(sources)-1
        A = np.zeros((n*2, n*2))
        for i, (rhat, uhat, distance) in enumerate(
                zip(rhats[:-1], uhats[:-1], distances[:-1])):
            if i == 0:
                A[::2, 0] = uhats[0].x
                A[1::2, 0] = uhats[0].y
            else:
                A[(i-1)*2, i*2] = -uhat.x
                A[(i-1)*2+1, i*2] = -uhat.y

            sij = 1-(distance/distances[i+1:]).to_value(u.one)
            A[i*2::2, i*2+1] = -sij*rhat.x
            A[i*2+1::2, i*2+1] = -sij*rhat.y

        Ainv = np.linalg.inv(A)

        pos_shape = np.broadcast(*[np.empty(x.shape)
                                   for x in poss+rhats]).shape
        Bpos = np.zeros((n*2,) + pos_shape) << (u.AU/u.kpc)
        vel_shape = np.broadcast(*[np.empty(x.shape)
                                   for x in vels+rhats]).shape
        Bvel = np.zeros((n*2,) + vel_shape) << (u.km/u.s/u.kpc)
        for i, (pos, vel, rhat, distance) in enumerate(
                zip(poss[1:], vels[1:], rhats[1:], distances[1:])):
            theta = (pos - other.pos) / distance
            Bpos[i*2] = theta.x
            Bpos[i*2+1] = theta.y
            mu = (vel - other.vel) / distance
            Bvel[i*2] = mu.x
            Bvel[i*2+1] = mu.y

        pos_sol = np.einsum('ij,j...->i...', Ainv, Bpos)
        sigmas = pos_sol[::2]
        alphas = pos_sol[1::2]
        vel_sol = np.einsum('ij,j...->i...', Ainv, Bvel)
        sigma_dots = vel_sol[::2]
        alpha_dots = vel_sol[1::2]

        for (source, rhat, uhat, distance,
             sigma, sigma_dot, alpha, alpha_dot) in zip(
                sources[:-1], rhats[:-1], uhats[:-1], distances[:-1],
                sigmas, sigma_dots, alphas, alpha_dots):
            source.sigma = sigma
            source.sigma_dot = sigma_dot
            source.s = sigma * distance
            source.s_dot = sigma_dot * distance
            source.alpha = alpha
            source.alpha_dot = alpha_dot
            if isinstance(source, Screen1D):
                source.pos = source.p * rhat + source.s * uhat
                source.vel = source.v * rhat + source.s_dot * uhat

    def _rel_posvel(self, other):
        if self.pos is None:
            self._solve_positions(other)
        return super()._rel_posvel(other)
