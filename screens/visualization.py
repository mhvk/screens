# Licensed under the GPLv3 - see LICENSE
"""Transforms needed to display theta-theta plots linearly with angle.

Usage::

  ax = plt.subplot(221, projection=ThetaTheta(theta))
  ax.imshow(theta_theta, ...)

"""
import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import Transform


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
        if theta is None:
            raise TypeError("Need to pass in theta!")

        self.theta_theta_transformation = ThetaThetaTransform(theta)
        super().__init__(*args, **kwargs)

    def imshow(self, *args, **kwargs):
        """
        Wrapper to Matplotlib's :meth:`~matplotlib.axes.Axes.imshow`.

        Defaults origin to lower, and scales data such that axis scale
        is linear in theta.
        """
        kwargs.setdefault('origin', 'lower')
        im = super().imshow(*args, **kwargs)
        trans_data = self.theta_theta_transformation + self.transData
        im.set_transform(trans_data)
        return im
