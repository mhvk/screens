# Licensed under the GPLv3 - see LICENSE
import astropy.units as u
import numpy as np

__all__ = ['DynamicSpectrum']


class DynamicSpectrum:
    """Dynamic spectrum and its axes.

    While code is meant to be agnostic to which axes are which, some may
    assume a shape of ``(..., time_axis, frequency_axis)``.

    Parameters
    ----------
    dynspec : `~numpy.ndarray`
        Intensities as a function of time and frequency.
    t : `~astropy.units.Quantity`
        Times of the dynamic spectrum.  Should have the proper shape to
        broadcast with ``dynspec``.
    f : `~astropy.units.Quantity`
        Frequencies of the dynamic spectrum.  Should have the proper shape to
        broadcast with ``dynspec``.
    noise : float
        The uncertainty in the intensities in the dynamic spectrum.
    """

    def __init__(self, dynspec, f, t, noise=None):
        self.dynspec = dynspec
        self.f = f
        self.t = t
        self.noise = noise

    @classmethod
    def fromfile(cls, filename, noise=None):
        """Read a dynamic spectrum from an HDF5 file.

        This includes its time and frequency axes.

        Note: this needs the baseband-tasks package for HDF5 file access.
        """
        from baseband.io import hdf5

        with hdf5.open(filename) as fh:
            dynspec = fh.read()
            f = fh.frequency
            t = (np.arange(-fh.shape[0] // 2, fh.shape[0] // 2)
                 / fh.sample_rate).to(u.minute)[:, np.newaxis]
            if noise is None:
                noise = fh.fh_raw.attrs['noise']

        self = cls(dynspec, f, t, noise)
        self.filename = filename
        return self
