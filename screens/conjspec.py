# Licensed under the GPLv3 - see LICENSE
import numpy as np
import astropy.units as u

from .fields import phasor, expand
from .dynspec import DynamicSpectrum


__all__ = ['ConjugateSpectrum']


def power(z):
    return z.real**2 + z.imag**2


class ConjugateSpectrum:
    """Conjugate spectrum and methods to fit it.

    The code is meant to be agnostic to which axes are which, but some may
    assume a shape of ``(..., doppler_axis, delay_axis)``.

    Parameters
    ----------
    conjspec : `~numpy.ndarray`
        Fourier transform of a dynamic spectrum.
    fd : `~astropy.units.Quantity`
        Doppler factors of the conjugate spectrum.  Normally time conjugate
        but can be arbitrary (e.g., conjugate of ``f*t``).  Should have the
        the proper shape to broadcast with ``conjspec``.
    tau : `~astropy.units.Quantity`
        Delays of the conjugate spectrum.  Should have the proper shape to
        broadcast with ``dynspec``.
    noise : float
        The uncertainty in the real and imaginary components of the conjugate
        spectrum.
    """

    def __init__(self, conjspec, tau, fd, noise=None):
        self.conjspec = conjspec
        self.tau = tau
        self.fd = fd
        self.noise = noise

    @property
    def secspec(self):
        """Secondary spectrum, i.e., the power of the conjugate spectrum."""
        return power(self.conjspec)

    @classmethod
    def from_dynamic_spectrum(cls, dynspec, normalization='mean', **kwargs):
        """Create a conjugate spectrum from a dynamic one.

        Easiest if the input is a `~screens.dynspec.DynamicSpectrum`
        instance.

        By passing in an explicit time axis using ``t``, one can get a
        different delay factor conjugate.  Particularly useful with
        ``t=f*t``, which takes into account the frequency dependence of
        the time variation of scintles.

        Note that the dynamic spectrum is assumed to have shape
        ``(..., time_axis, frequency_axis)``.

        Parameters
        ----------
        dynspec : array_like or `~screens.dynspec.DynamicSpectrum`
            Input dynamic spectrum for which the fourier transform will
            be calculated.  If it has attributes ``f``, ``t``, ``d_eff``,
            ``theta``, ``magnification``, and ``noise``, those will be
            used as default inputs.  TODO: ``noise`` is likely wrong!
        normalization : 'mean' or None
            Normalize dynamic spectrum by its mean and subtract 1 before
            transforming and ensure the resulting conjugate spectrum is
            normalized as well, with the 0, 0 element equal unity.
        **kwargs
            Other arguments to initialize the conjugate spectrum.
        """
        for key in ('f', 't', 'noise'):
            val = getattr(dynspec, key, None)
            if val is not None:
                kwargs.setdefault(key, val)

        if isinstance(dynspec, DynamicSpectrum):
            # TODO: give DynamicSpectrum an __array__ method.
            dynspec = dynspec.dynspec

        if normalization == 'mean':
            dynspec = dynspec / dynspec.mean() - 1.  # Not in place!!

        f = kwargs.pop('f')
        t = kwargs.pop('t')
        fd = kwargs.pop('fd', None)
        if t.size in t.shape and fd is None:  # fast FFT possible.
            conj = np.fft.fftshift(np.fft.fft2(dynspec))
            fd = np.fft.fftshift(np.fft.fftfreq(t.size, t[1]-t[0])
                                 .reshape(t.shape))
        else:
            # Time axis has slow FT or explicit fd given.
            # Time is assumed to be along axis -2.
            # TODO: relax this assumption.
            if fd is None:
                t_step = np.abs(np.diff(t, axis=-2)).min()
                n_t = round((np.ptp(t)/t_step).to_value(1).item()) + 1
                fd = np.fft.fftshift(np.fft.fftfreq(n_t, t_step).to(u.mHz))
                fd = expand(fd, n=dynspec.ndim)
                linear_axis = "transform"

            else:
                if fd.ndim == 1:
                    fd = expand(fd, n=dynspec.ndim)

                linear_axis = None
                # Check for linear spacing to speed up the calculation.  Here,
                # the first check is whether fd is a linearly spaced array
                # along a single axis, and the second whether the last axis of
                # time (generally along frequency) is linearly spaced.
                if (fd.size in fd.shape and np.allclose(
                        dfd := np.diff(
                            fd, axis=(axis := fd.shape.index(fd.size))),
                        dfd.take(0, axis=axis))):
                    linear_axis = "transform"
                elif (t.shape[-1] > 1
                      and np.allclose(dt := np.diff(t, axis=-1),
                                      dt[..., :1])):
                    linear_axis = -1

            factor = (phasor(t, fd, linear_axis=linear_axis).conj()
                      * dynspec)
            step1 = factor.sum(-2, keepdims=True).swapaxes(0, -2).squeeze(0)
            conj = np.fft.fftshift(np.fft.fft(step1, axis=-1), axes=-1)
            fd.shape = conj.shape[-2], 1

        if normalization == 'mean':
            conj /= dynspec.size
            conj[conj.shape[-2] // 2, conj.shape[-1] // 2] = 1.

        tau = np.fft.fftshift(np.fft.fftfreq(f.size, f[1]-f[0]))
        tau.shape = f.shape
        self = cls(conj, tau, fd, **kwargs)
        self.f = f
        self.t = t
        self.normalization = normalization
        return self
