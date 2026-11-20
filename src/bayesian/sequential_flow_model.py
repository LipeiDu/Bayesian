"""Compatibility alias for :mod:`bayesian.sequential.flow`."""

import sys
from bayesian.sequential import flow as _implementation

sys.modules[__name__] = _implementation
