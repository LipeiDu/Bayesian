"""Compatibility alias for :mod:`bayesian.sequential.prior`."""

import sys
from bayesian.sequential import prior as _implementation

sys.modules[__name__] = _implementation
