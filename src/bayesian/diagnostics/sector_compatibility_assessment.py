"""Assess soft-support and hard-sector compatibility of sequential inference.

The hard-only and sequential posterior samples are reevaluated with one common
production emulator. Soft compatibility is expressed as projected-soft
highest-density-region membership; hard compatibility uses likelihood,
Mahalanobis discrepancy, and posterior-predictive coverage diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt

from bayesian import log_posterior
from bayesian.posterior_utils import normalized_weights
from bayesian.emulation import base
from bayesian.diagnostics.hard_likelihood_diagnostics import (
    GroupLikelihoodInputs,
    PosteriorSamples,
    _load_group_inputs,
    _reorder_and_expand_samples,
    load_posterior_samples,
    weighted_summary,
)
from bayesian.sequential.config import SequentialInferenceConfig
from bayesian.sequential.prior import load_soft_density_model

logger = logging.getLogger(__name__)


@dataclass
class PosteriorCompatibility:
    group_log_likelihood: dict[str, npt.NDArray[np.float64]]
    group_chi2: dict[str, npt.NDArray[np.float64]]
    observable_log_likelihood: dict[str, npt.NDArray[np.float64]]
    observable_chi2: dict[str, npt.NDArray[np.float64]]
    predictive: dict[str, dict]


def soft_hdr_thresholds(
    reference_log_density: npt.ArrayLike,
    masses: npt.ArrayLike,
) -> dict[float, float]:
    """Return density thresholds enclosing the requested reference masses."""
    values = np.asarray(reference_log_density, dtype=float)
    output = {}
    for mass in np.asarray(masses, dtype=float):
        if not 0.0 < mass < 1.0:
            msg = "HDR masses must lie strictly between zero and one."
            raise ValueError(msg)
        output[float(mass)] = float(np.quantile(values, 1.0 - mass))
    return output


def weighted_hdr_fractions(
    log_density: npt.ArrayLike,
    weights: npt.ArrayLike,
    thresholds: dict[float, float],
) -> dict[float, float]:
    values = np.asarray(log_density, dtype=float)
    normalized = normalized_weights(weights, len(log_density))
    return {
        mass: float(np.sum(normalized[values >= threshold]))
        for mass, threshold in thresholds.items()
    }


def predictive_moments(
    central_values: npt.NDArray[np.float64],
    emulator_covariances: npt.NDArray[np.float64],
    weights: npt.ArrayLike,
    experimental_variances: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Moment-match the posterior predictive mixture, including experiment."""
    normalized = normalized_weights(weights, len(central_values))
    mean = normalized @ central_values
    second_moment = np.einsum("i,ij,ik->jk", normalized, central_values, central_values)
    mean_emulator_covariance = np.einsum("i,ijk->jk", normalized, emulator_covariances)
    covariance = second_moment - np.outer(mean, mean) + mean_emulator_covariance
    covariance += np.diag(np.asarray(experimental_variances, dtype=float))
    covariance = 0.5 * (covariance + covariance.T)
    return mean, covariance


def _evaluate_posterior(
    parameters: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    groups: list[GroupLikelihoodInputs],
    *,
    batch_size: int,
) -> PosteriorCompatibility:
    n_samples = len(parameters)
    group_log_likelihood = {group.name: np.empty(n_samples) for group in groups}
    group_chi2 = {group.name: np.empty(n_samples) for group in groups}
    observable_log_likelihood = {
        name: np.empty(n_samples)
        for group in groups
        for name in group.observable_slices
    }
    observable_chi2 = {
        name: np.empty(n_samples)
        for group in groups
        for name in group.observable_slices
    }
    accumulators = {
        group.name: {
            "mean": np.zeros(len(group.data_y)),
            "second": np.zeros((len(group.data_y), len(group.data_y))),
            "emulator_covariance": np.zeros((len(group.data_y), len(group.data_y))),
        }
        for group in groups
    }

    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        batch_weights = weights[start:stop]
        for group in groups:
            predictions = base.predict_emulation_group(
                parameters[start:stop],
                group.emulator_results,
                group.emulator_config,
                emulator_group_cov_unexplained=group.emulator_cov_unexplained,
            )
            central = np.asarray(predictions["central_value"], dtype=float)
            emulator_covariance = np.asarray(predictions["cov"], dtype=float)
            residual = central - group.data_y
            covariance = emulator_covariance + np.diag(group.data_y_err**2)[np.newaxis, :, :]
            for local_index, (sample_residual, sample_covariance) in enumerate(
                zip(residual, covariance, strict=True)
            ):
                index = start + local_index
                solved = np.linalg.solve(sample_covariance, sample_residual)
                group_chi2[group.name][index] = sample_residual @ solved
                group_log_likelihood[group.name][index] = log_posterior._loglikelihood(
                    sample_residual,
                    sample_covariance,
                )
                for observable_name, observable_slice in group.observable_slices.items():
                    sliced_residual = sample_residual[observable_slice]
                    sliced_covariance = sample_covariance[observable_slice, observable_slice]
                    observable_chi2[observable_name][index] = sliced_residual @ np.linalg.solve(
                        sliced_covariance,
                        sliced_residual,
                    )
                    observable_log_likelihood[observable_name][index] = log_posterior._loglikelihood(
                        sliced_residual,
                        sliced_covariance,
                    )
            accumulator = accumulators[group.name]
            accumulator["mean"] += batch_weights @ central
            accumulator["second"] += np.einsum(
                "i,ij,ik->jk",
                batch_weights,
                central,
                central,
            )
            accumulator["emulator_covariance"] += np.einsum(
                "i,ijk->jk",
                batch_weights,
                emulator_covariance,
            )

    predictive = {}
    for group in groups:
        accumulator = accumulators[group.name]
        mean = accumulator["mean"]
        covariance = accumulator["second"] - np.outer(mean, mean)
        covariance += accumulator["emulator_covariance"]
        covariance += np.diag(group.data_y_err**2)
        covariance = 0.5 * (covariance + covariance.T)
        residual = mean - group.data_y
        sigma = np.sqrt(np.maximum(np.diag(covariance), np.finfo(float).tiny))
        z_score = residual / sigma
        predictive[group.name] = {
            "mean": mean,
            "covariance": covariance,
            "z_score": z_score,
            "mahalanobis_per_bin": float(
                residual @ np.linalg.solve(covariance, residual) / len(residual)
            ),
            "coverage_68": float(np.mean(np.abs(z_score) <= 1.0)),
            "coverage_95": float(np.mean(np.abs(z_score) <= 1.96)),
            "mean_abs_z": float(np.mean(np.abs(z_score))),
            "max_abs_z": float(np.max(np.abs(z_score))),
        }

    group_log_likelihood["total"] = np.sum(
        np.vstack([group_log_likelihood[group.name] for group in groups]),
        axis=0,
    )
    group_chi2["total"] = np.sum(
        np.vstack([group_chi2[group.name] for group in groups]),
        axis=0,
    )
    return PosteriorCompatibility(
        group_log_likelihood=group_log_likelihood,
        group_chi2=group_chi2,
        observable_log_likelihood=observable_log_likelihood,
        observable_chi2=observable_chi2,
        predictive=predictive,
    )


def _soft_coordinates(posterior: PosteriorSamples) -> npt.NDArray[np.float64]:
    indices = [posterior.parameter_names.index(name) for name in ("nucleon_width", "normalization")]
    return posterior.samples[:, indices]


def _short_observable_label(name: str) -> str:
    experiment = next((value.upper() for value in ("alice", "atlas", "cms") if value in name), "")
    centrality_match = re.search(r"__+(\d+-\d+)$", name)
    centrality = centrality_match.group(1) if centrality_match else ""
    if "__inclusive_jet__" in name:
        radius_match = re.search(r"__R(0\.\d+)__", name)
        radius = f"R={radius_match.group(1)}" if radius_match else ""
        return f"{experiment} jet {radius} {centrality}%".strip()
    species_match = re.search(r"__hadron__pt_(ch|pi0|pi)_", name)
    species = species_match.group(1) if species_match else "hadron"
    species_label = {"ch": "charged h.", "pi": "pion", "pi0": "pi0"}.get(species, species)
    return f"{experiment} {species_label} {centrality}%".strip()


def _group_for_observable(name: str, groups: list[GroupLikelihoodInputs]) -> str:
    return next(group.name for group in groups if name in group.observable_slices)


def _write_sample_table(
    path: Path,
    posteriors: list[PosteriorSamples],
    diagnostics: dict[str, PosteriorCompatibility],
    soft_log_density: dict[str, npt.NDArray[np.float64]],
) -> None:
    group_names = list(next(iter(diagnostics.values())).group_log_likelihood)
    fields = [
        "posterior",
        "source_index",
        "weight",
        "log_p_soft",
        *[f"logL_{name}" for name in group_names],
        *[f"chi2_{name}" for name in group_names],
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for posterior in posteriors:
            result = diagnostics[posterior.label]
            for index, source_index in enumerate(posterior.source_indices):
                row = {
                    "posterior": posterior.label,
                    "source_index": int(source_index),
                    "weight": f"{posterior.weights[index]:.17g}",
                    "log_p_soft": f"{soft_log_density[posterior.label][index]:.17g}",
                }
                row.update(
                    {
                        f"logL_{name}": f"{result.group_log_likelihood[name][index]:.17g}"
                        for name in group_names
                    }
                )
                row.update(
                    {
                        f"chi2_{name}": f"{result.group_chi2[name][index]:.17g}"
                        for name in group_names
                    }
                )
                writer.writerow(row)


def _observable_rows(
    posteriors: list[PosteriorSamples],
    diagnostics: dict[str, PosteriorCompatibility],
    groups: list[GroupLikelihoodInputs],
) -> list[dict]:
    rows = []
    reference, candidate = posteriors
    for name in diagnostics[reference.label].observable_log_likelihood:
        group_name = _group_for_observable(name, groups)
        observable_slice = next(group.observable_slices[name] for group in groups if name in group.observable_slices)
        n_bins = observable_slice.stop - observable_slice.start
        reference_logl = weighted_summary(
            diagnostics[reference.label].observable_log_likelihood[name],
            reference.weights,
        )
        candidate_logl = weighted_summary(
            diagnostics[candidate.label].observable_log_likelihood[name],
            candidate.weights,
        )
        reference_chi2 = weighted_summary(
            diagnostics[reference.label].observable_chi2[name] / n_bins,
            reference.weights,
        )
        candidate_chi2 = weighted_summary(
            diagnostics[candidate.label].observable_chi2[name] / n_bins,
            candidate.weights,
        )
        reference_z = diagnostics[reference.label].predictive[group_name]["z_score"][observable_slice]
        candidate_z = diagnostics[candidate.label].predictive[group_name]["z_score"][observable_slice]
        rows.append(
            {
                "observable": name,
                "label": _short_observable_label(name),
                "group": group_name,
                "n_bins": n_bins,
                "hard_only_mean_logL": reference_logl["mean"],
                "sequential_mean_logL": candidate_logl["mean"],
                "delta_mean_logL": candidate_logl["mean"] - reference_logl["mean"],
                "hard_only_mean_chi2_per_bin": reference_chi2["mean"],
                "sequential_mean_chi2_per_bin": candidate_chi2["mean"],
                "delta_mean_chi2_per_bin": candidate_chi2["mean"] - reference_chi2["mean"],
                "hard_only_predictive_coverage_95": float(np.mean(np.abs(reference_z) <= 1.96)),
                "sequential_predictive_coverage_95": float(np.mean(np.abs(candidate_z) <= 1.96)),
                "hard_only_predictive_max_abs_z": float(np.max(np.abs(reference_z))),
                "sequential_predictive_max_abs_z": float(np.max(np.abs(candidate_z))),
            }
        )
    return rows


def _plot_summary(
    path: Path,
    posteriors: list[PosteriorSamples],
    diagnostics: dict[str, PosteriorCompatibility],
    soft_fractions: dict[str, dict[float, float]],
    groups: list[GroupLikelihoodInputs],
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    colors = {
        "Projected soft": "#2a9d3f",
        "Hard obs. only": "#2878a5",
        "Sequential": "#d62828",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))

    ax = axes[0, 0]
    masses = list(next(iter(soft_fractions.values())))
    positions = np.arange(len(masses))
    width = 0.25
    ax.bar(
        positions - width,
        masses,
        width,
        color=colors["Projected soft"],
        label="Projected soft",
    )
    for offset, posterior in zip((0.0, 1.0), posteriors, strict=True):
        ax.bar(
            positions + offset * width,
            [soft_fractions[posterior.label][mass] for mass in masses],
            width,
            color=colors[posterior.label],
            label=posterior.label,
        )
    ax.set_xticks(positions, [f"{100 * mass:g}% HDR" for mass in masses])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Posterior fraction inside soft HDR")
    ax.set_title("Projected-soft support retained")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    total_values = np.concatenate(
        [diagnostics[posterior.label].group_log_likelihood["total"] for posterior in posteriors]
    )
    bins = np.linspace(np.min(total_values), np.max(total_values), 55)
    for posterior in posteriors:
        ax.hist(
            diagnostics[posterior.label].group_log_likelihood["total"],
            bins=bins,
            weights=posterior.weights,
            density=True,
            histtype="step",
            linewidth=2.2,
            color=colors[posterior.label],
            label=posterior.label,
        )
    ax.set_xlabel(r"Common-emulator $\log\mathcal{L}_{\rm H}$")
    ax.set_ylabel("Probability density")
    ax.set_title("Hard-fit quality")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    group_names = [group.name for group in groups]
    positions = np.arange(len(group_names))
    comparison_width = 0.34
    for offset, posterior in zip((-0.5, 0.5), posteriors, strict=True):
        means = []
        lower = []
        upper = []
        for group in groups:
            values = diagnostics[posterior.label].group_chi2[group.name] / len(group.data_y)
            summary = weighted_summary(values, posterior.weights)
            means.append(summary["mean"])
            lower.append(summary["mean"] - summary["q16"])
            upper.append(summary["q84"] - summary["mean"])
        ax.bar(
            positions + offset * comparison_width,
            means,
            comparison_width,
            yerr=np.vstack([lower, upper]),
            capsize=3,
            color=colors[posterior.label],
            label=posterior.label,
        )
    ax.set_xticks(positions, [name.replace("_group", "").title() for name in group_names])
    ax.set_ylabel(r"Posterior mean $\chi^2/n_{\rm bin}$")
    ax.set_title("Covariance-aware hard discrepancy")

    ax = axes[1, 1]
    labels = []
    values = {posterior.label: [] for posterior in posteriors}
    for group in groups:
        for coverage_name, coverage_label in (("coverage_68", "68%"), ("coverage_95", "95%")):
            labels.append(f"{group.name.replace('_group', '').title()}\n{coverage_label}")
            for posterior in posteriors:
                values[posterior.label].append(
                    diagnostics[posterior.label].predictive[group.name][coverage_name]
                )
    positions = np.arange(len(labels))
    for offset, posterior in zip((-0.5, 0.5), posteriors, strict=True):
        ax.bar(
            positions + offset * comparison_width,
            values[posterior.label],
            comparison_width,
            color=colors[posterior.label],
            label=posterior.label,
        )
    ax.set_xticks(positions, labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fraction of data bins covered")
    ax.set_title("Moment-matched posterior prediction")

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat, strict=True):
        ax.text(0.02, 0.97, label, transform=ax.transAxes, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_observable_changes(path: Path, rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: row["delta_mean_logL"])
    positions = np.arange(len(ordered))
    colors = ["#2ca02c" if row["group"] == "jet_group" else "#2878a5" for row in ordered]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.4), sharey=True)
    axes[0].barh(positions, [row["delta_mean_logL"] for row in ordered], color=colors)
    axes[0].axvline(0.0, color="#333333", linewidth=1.0)
    axes[0].set_xlabel(r"$\Delta\langle\log\mathcal{L}\rangle$ (sequential $-$ hard only)")
    axes[1].barh(
        positions,
        [row["delta_mean_chi2_per_bin"] for row in ordered],
        color=colors,
    )
    axes[1].axvline(0.0, color="#333333", linewidth=1.0)
    axes[1].set_xlabel(r"$\Delta\langle\chi^2/n_{\rm bin}\rangle$")
    axes[0].set_yticks(positions, [row["label"] for row in ordered])
    axes[1].tick_params(labelleft=False)
    axes[0].text(0.02, 0.98, "(a)", transform=axes[0].transAxes, va="top")
    axes[1].text(0.02, 0.98, "(b)", transform=axes[1].transAxes, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
