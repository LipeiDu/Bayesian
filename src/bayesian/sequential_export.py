"""Compatibility alias for :mod:`bayesian.sequential.export`."""

import sys
from bayesian.sequential import export as _implementation

sys.modules[__name__] = _implementation
