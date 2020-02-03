# Licensed under the GPLv3 - see LICENSE
"""Pulsar Scintillation Screen modelling."""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------

if not _ASTROPY_SETUP_:   # noqa
    # For egg_info test builds to pass, put package imports here.
    from .dynspec import DynamicSpectrum  # noqa
    from .secspec import SecondarySpectrum  # noqa
