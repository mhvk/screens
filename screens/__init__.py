# Licensed under the GPLv3 - see LICENSE
"""Pulsar Scintillation Screen modelling."""

from .dynspec import DynamicSpectrum  # noqa
from .conjspec import ConjugateSpectrum  # noqa

try:
    from .version import version as __version__  # noqa
except ImportError:
    __version__ = ''
