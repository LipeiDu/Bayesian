"""Compatibility imports and forwarding entry point for a study-owned command.

Numerical helpers remain in bayesian.diagnostics; the CLI recipe belongs to
STAT_soft_hard. See docs/development/ownership.md for installation and ownership.
"""

import warnings

warnings.warn(
    "bayesian.sector_compatibility_assessment is deprecated; use bayesian.diagnostics.sector_compatibility_assessment instead.",
    DeprecationWarning,
    stacklevel=2,
)

from importlib import import_module

from bayesian.diagnostics.sector_compatibility_assessment import (
    PosteriorCompatibility as PosteriorCompatibility,
    soft_hdr_thresholds as soft_hdr_thresholds,
    weighted_hdr_fractions as weighted_hdr_fractions,
    predictive_moments as predictive_moments,
    _evaluate_posterior as _evaluate_posterior,
    _soft_coordinates as _soft_coordinates,
    _short_observable_label as _short_observable_label,
    _group_for_observable as _group_for_observable,
    _write_sample_table as _write_sample_table,
    _observable_rows as _observable_rows,
    _plot_summary as _plot_summary,
    _plot_observable_changes as _plot_observable_changes,
)


def __getattr__(name):
    return getattr(import_module("bayesian.diagnostics.sector_compatibility_assessment"), name)


def main():
    try:
        command = import_module("stat_bayesian_tools.sector_compatibility_assessment")
    except ModuleNotFoundError as error:
        if error.name != "stat_bayesian_tools":
            raise
        message = ("This study CLI moved to STAT_soft_hard/supporting_workflows/bayesian_backend_tools. "
                   "Add that directory to PYTHONPATH; see docs/development/ownership.md.")
        raise SystemExit(message) from error
    command.main()


if __name__ == "__main__":
    main()
