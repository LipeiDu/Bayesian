"""Compatibility imports and forwarding entry point for a study-owned command.

Numerical helpers remain in bayesian.diagnostics; the CLI recipe belongs to
STAT_soft_hard. See docs/development/ownership.md for installation and ownership.
"""

import warnings

warnings.warn(
    "bayesian.hard_likelihood_diagnostics is deprecated; use bayesian.diagnostics.hard_likelihood_diagnostics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from importlib import import_module

from bayesian.diagnostics.hard_likelihood_diagnostics import (
    PosteriorSamples as PosteriorSamples,
    GroupLikelihoodInputs as GroupLikelihoodInputs,
    _decode_names as _decode_names,
    _normalize_weights as _normalize_weights,
    weighted_summary as weighted_summary,
    _select_indices as _select_indices,
    load_posterior_samples as load_posterior_samples,
    _reorder_and_expand_samples as _reorder_and_expand_samples,
    _load_group_inputs as _load_group_inputs,
    _evaluate_group_batch as _evaluate_group_batch,
    evaluate_group_likelihoods as evaluate_group_likelihoods,
    _effective_sample_size as _effective_sample_size,
    _write_sample_table as _write_sample_table,
    _write_observable_sample_table as _write_observable_sample_table,
    _plot_comparison as _plot_comparison,
    _short_observable_label as _short_observable_label,
    _plot_observable_differences as _plot_observable_differences,
    _build_summary as _build_summary,
)


def __getattr__(name):
    return getattr(import_module("bayesian.diagnostics.hard_likelihood_diagnostics"), name)


def main():
    try:
        command = import_module("stat_bayesian_tools.hard_likelihood_diagnostics")
    except ModuleNotFoundError as error:
        if error.name != "stat_bayesian_tools":
            raise
        message = ("This study CLI moved to STAT_soft_hard/supporting_workflows/bayesian_backend_tools. "
                   "Add that directory to PYTHONPATH; see docs/development/ownership.md.")
        raise SystemExit(message) from error
    command.main()


if __name__ == "__main__":
    main()
