"""Compatibility alias for :mod:`bayesian.sequential.export`."""

import warnings

warnings.warn(
    "bayesian.sequential_export is deprecated; use bayesian.sequential.export instead.",
    DeprecationWarning,
    stacklevel=2,
)


import sys
from bayesian.sequential import export as _implementation

sys.modules[__name__] = _implementation
