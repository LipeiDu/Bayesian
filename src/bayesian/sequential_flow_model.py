"""Compatibility alias for :mod:`bayesian.sequential.flow`."""

import warnings

warnings.warn(
    "bayesian.sequential_flow_model is deprecated; use bayesian.sequential.flow instead.",
    DeprecationWarning,
    stacklevel=2,
)


import sys
from bayesian.sequential import flow as _implementation

sys.modules[__name__] = _implementation
