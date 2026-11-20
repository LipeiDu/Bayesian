"""Compatibility alias for :mod:`bayesian.sequential.config`."""

import sys
from bayesian.sequential import config as _implementation

sys.modules[__name__] = _implementation
