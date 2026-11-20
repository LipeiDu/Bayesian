"""Compatibility imports and forwarding entry point for a study-owned command.

Numerical helpers remain in bayesian.diagnostics; the CLI recipe belongs to
STAT_soft_hard. See docs/development/ownership.md for installation and ownership.
"""

import warnings

warnings.warn(
    "bayesian.fixed_medium_alpha_scan is deprecated; use bayesian.diagnostics.fixed_medium_alpha_scan instead.",
    DeprecationWarning,
    stacklevel=2,
)

from importlib import import_module

from bayesian.diagnostics.fixed_medium_alpha_scan import (
    MediumPoint as MediumPoint,
    _decode_names as _decode_names,
    _normalized_weights as _normalized_weights,
    load_posterior_mean as load_posterior_mean,
    load_soft_mean as load_soft_mean,
    build_medium_points as build_medium_points,
    trapezoid_weights as trapezoid_weights,
    normalize_log_curve as normalize_log_curve,
    summarize_density as summarize_density,
    _sampled_parameter_grid as _sampled_parameter_grid,
    evaluate_medium as evaluate_medium,
    _sampled_fixed_summary as _sampled_fixed_summary,
    _write_curve_table as _write_curve_table,
    _plot_scan as _plot_scan,
)


def __getattr__(name):
    return getattr(import_module("bayesian.diagnostics.fixed_medium_alpha_scan"), name)


def main():
    try:
        command = import_module("stat_bayesian_tools.fixed_medium_alpha_scan")
    except ModuleNotFoundError as error:
        if error.name != "stat_bayesian_tools":
            raise
        message = ("This study CLI moved to STAT_soft_hard/supporting_workflows/bayesian_backend_tools. "
                   "Add that directory to PYTHONPATH; see docs/development/ownership.md.")
        raise SystemExit(message) from error
    command.main()


if __name__ == "__main__":
    main()
