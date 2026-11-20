"""Compatibility alias for :mod:`bayesian.sequential.config`."""

import warnings

warnings.warn(
    "bayesian.sequential_inference is deprecated; use bayesian.sequential.config instead.",
    DeprecationWarning,
    stacklevel=2,
)


import sys
from bayesian.sequential import config as _implementation

sys.modules[__name__] = _implementation
