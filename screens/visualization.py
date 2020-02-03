# Licensed under the GPLv3 - see LICENSE
"""Transforms needed to display theta-theta plots linearly with angle.

Sample usage::

  ax = plt.subplot(221, projection=ThetaTheta(my_theta_grid))
  ax.imshow(my_theta_theta_array, ...)

"""
import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import Transform


__all__ = ['ThetaTheta', 'ThetaThetaTransform', 'ThetaThetaAxes']


class ThetaThetaTransform(Transform):
    name = 'theta_theta'
    input_dims = 2
    output_dims = 2
    is_separable = True
    has_inverse = False

    def __init__(self, theta=None, forward=True):
        super().__init__()
        self.theta = getattr(theta, 'value', theta)
        self.forward = forward
        self.lin_theta = np.linspace(self.theta.min(), self.theta.max(),
                                     len(theta))

    def transform_non_affine(self, values):
        if self.forward:
            return np.interp(values, self.lin_theta, self.theta)

        else:
            return np.interp(values, self.theta, self.lin_theta)

    def inverted(self):
        return self.__class__(self.theta, forward=not self.forward)


class ThetaTheta:
    def __init__(self, theta):
        self.theta = theta

    def _as_mpl_axes(self):
        return ThetaThetaAxes, dict(theta=self.theta)


class ThetaThetaAxes(Axes):
    def __init__(self, *args, theta=None, **kwargs):
        """Define transformed axes that are linear in theta.

        This is for showing Theta-Theta images where the grid of angles is
        not uniform (instead, e.g., being uniform along the parabola).

        Parameters
        ----------
        theta : `~astropy.units.Quantity`
            Grid of angles at which the theta-theta image is evaluated.
        """
        if theta is None:
            raise TypeError("Need to pass in theta!")

        self.theta_theta_transformation = ThetaThetaTransform(theta)
        super().__init__(*args, **kwargs)

    def imshow(self, *args, **kwargs):
        """Show image with linear theta scales, defaulting to origin='lower'.

        Wraps :meth:`matplotlib.axes.Axes.imshow`.
        """
        kwargs.setdefault('origin', 'lower')
        im = super().imshow(*args, **kwargs)
        trans_data = self.theta_theta_transformation + self.transData
        im.set_transform(trans_data)
        return im
