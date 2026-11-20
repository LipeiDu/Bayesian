"""Compatibility alias for :mod:`bayesian.sequential.prior`."""

import warnings

warnings.warn(
    "bayesian.sequential_prior is deprecated; use bayesian.sequential.prior instead.",
    DeprecationWarning,
    stacklevel=2,
)


import sys
from bayesian.sequential import prior as _implementation

sys.modules[__name__] = _implementation
