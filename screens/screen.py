import operator

import numpy as np
from astropy import constants as const, units as u
from astropy.coordinates import CartesianRepresentation
from astropy.utils.shapes import ShapedLikeNDArray
from astropy.utils.decorators import lazyproperty


ZERO_VELOCITY = CartesianRepresentation(0., 0., 0., unit=u.km/u.s)


class Screen(ShapedLikeNDArray):
    """Screen passing through a source of radiation.

    Parameters
    ----------
    pos : `~astropy.coordinates.CartesianRepresentation`
        Position of the points
    vel : `~astropy.coordinates.CartesianRepresentation`
        Corresponding velocities
    magnification : array-like
        Brightness or magnification of the points.  Usually complex.
    """
    _shaped_attrs = ('pos', 'vel', 'magnification')

    def __init__(self, pos, vel=ZERO_VELOCITY, magnification=1.,
                 source=None, distance=None):
        self.pos = pos
        self.vel = vel
        self.magnification = magnification
        self.source = source
        self.distance = distance

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

        new.source = self.source
        new.distance = self.distance
        return new

    @property
    def brightness(self):
        """Brightness of each path."""
        return self.paths[0]

    @property
    def tau(self):
        """Delay for each path."""
        return self.paths[1]

    @property
    def taudot(self):
        """Time derivative of the delay for each path."""
        return self.paths[2]

    @lazyproperty
    def paths(self):
        source = self.source
        distance = self.distance
        if source is None:
            return (np.broadcast_to(self.magnification, self.shape),
                    np.broadcast_to(0*u.us, self.shape),
                    np.broadcast_to(0*u.us/u.s, self.shape))

        rel_xy = (source.pos - self.pos).get_xyz(xyz_axis=-1)[..., :2]
        rel_vxy = (source.vel - self.vel).get_xyz(xyz_axis=-1)[..., :2]
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


class Screen1D(Screen):
    """One-dimensional screen.

    The assumption is that all scattering points are essentially on a line.
    """
    _shaped_attrs = ('pos', 'vel', 'magnification', 'normal')

    def __init__(self, pos, normal, v_normal=0, magnification=1.,
                 source=None, distance=None):
        super().__init__(pos, vel=normal*v_normal, magnification=magnification,
                         source=source, distance=distance)
        self.normal = normal

    @lazyproperty
    def paths(self):
        source = self.source
        distance = self.distance
        if source is None:
            return (np.broadcast_to(self.magnification, self.shape),
                    np.broadcast_to(0*u.us, self.shape),
                    np.broadcast_to(0*u.us/u.s, self.shape))
        # OK for single screen, but not for two...
        rel_pos = (source.pos - self.pos).dot(self.normal)
        rel_vel = (source.vel - self.vel).dot(self.normal)
        theta = rel_pos / distance
        tau = source.tau + distance / const.c * 0.5 * theta**2
        taudot = source.taudot + theta * rel_vel / const.c
        brightness = source.brightness * self.magnification
        return brightness, tau, taudot
