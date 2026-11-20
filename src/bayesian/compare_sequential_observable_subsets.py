"""Compatibility imports and forwarding entry point for a study-owned command.

Numerical helpers remain in bayesian.diagnostics; the CLI recipe belongs to
STAT_soft_hard. See docs/development/ownership.md for installation and ownership.
"""
from importlib import import_module

from bayesian.diagnostics.compare_sequential_observable_subsets import (
    DensitySamples as DensitySamples,
    _decode_names as _decode_names,
    load_posterior as load_posterior,
    load_soft_projection as load_soft_projection,
    _parameter_values as _parameter_values,
    _credible_density_thresholds as _credible_density_thresholds,
    _weighted_limits as _weighted_limits,
    _plot_medium_contours as _plot_medium_contours,
    _plot_marginal as _plot_marginal,
    _summary as _summary,
)


def __getattr__(name):
    return getattr(import_module("bayesian.diagnostics.compare_sequential_observable_subsets"), name)


def main():
    try:
        command = import_module("stat_bayesian_tools.compare_sequential_observable_subsets")
    except ModuleNotFoundError as error:
        if error.name != "stat_bayesian_tools":
            raise
        message = ("This study CLI moved to STAT_soft_hard/supporting_workflows/bayesian_backend_tools. "
                   "Add that directory to PYTHONPATH; see docs/development/ownership.md.")
        raise SystemExit(message) from error
    command.main()


if __name__ == "__main__":
    main()
