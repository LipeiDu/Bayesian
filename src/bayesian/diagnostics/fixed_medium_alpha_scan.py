"""Scan conditional alpha_s posteriors at representative fixed media.

The script evaluates the common production hard likelihood directly on an
alpha_s grid. With a flat alpha_s prior, each normalized likelihood curve is
the conditional posterior p(alpha_s | N, w, D_H). Representative media are
derived from the projected-soft, sequential, and hard-only posterior samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt

from bayesian.emulation import base
from bayesian.diagnostics.hard_likelihood_diagnostics import (
    GroupLikelihoodInputs,
    _load_group_inputs,
    evaluate_group_likelihoods,
)
from bayesian.parameterization import parameterization_info
from bayesian.posterior_utils import normalized_weights, weighted_quantile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MediumPoint:
    key: str
    label: str
    nucleon_width: float
    normalization: float
    color: str
    marker: str


def _decode_names(values: npt.NDArray) -> tuple[str, ...]:
    return tuple(value.decode() if isinstance(value, bytes) else str(value) for value in values)


def _normalized_weights(weights):
    return normalized_weights(weights, len(weights))


def load_posterior_mean(path: Path) -> dict[str, float]:
    with h5py.File(path, "r") as stream:
        samples = np.asarray(stream["posterior_samples"], dtype=float)
        weights = (
            np.asarray(stream["weights"], dtype=float)
            if "weights" in stream
            else np.ones(len(samples), dtype=float)
        )
        names = _decode_names(np.asarray(stream["parameter_names"][:]))
    weights = normalized_weights(weights, len(samples))
    return {name: float(weights @ samples[:, index]) for index, name in enumerate(names)}


def load_soft_mean(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    return {
        "nucleon_width": float(frame["nucleon_width"].mean()),
        "normalization": float(frame["normalization"].mean()),
    }


def build_medium_points(
    *,
    soft_projection: Path,
    sequential_posterior: Path,
    hard_posterior: Path,
    fixed_width: float,
    fixed_normalization: float,
) -> list[MediumPoint]:
    soft = load_soft_mean(soft_projection)
    sequential = load_posterior_mean(sequential_posterior)
    hard = load_posterior_mean(hard_posterior)
    return [
        MediumPoint(
            key="projected_soft_mean",
            label="Projected-soft mean",
            nucleon_width=soft["nucleon_width"],
            normalization=soft["normalization"],
            color="#2a9d3f",
            marker="o",
        ),
        MediumPoint(
            key="source_fixed_medium",
            label="Source fixed medium",
            nucleon_width=fixed_width,
            normalization=fixed_normalization,
            color="#6c757d",
            marker="s",
        ),
        MediumPoint(
            key="sequential_mean",
            label="Sequential mean",
            nucleon_width=sequential["nucleon_width"],
            normalization=sequential["normalization"],
            color="#d62828",
            marker="D",
        ),
        MediumPoint(
            key="hard_only_mean",
            label="Hard-only mean",
            nucleon_width=hard["nucleon_width"],
            normalization=hard["normalization"],
            color="#277da1",
            marker="^",
        ),
    ]


def trapezoid_weights(grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        message = "Scan grid must contain at least two finite, strictly increasing points"
        raise ValueError(message)
    weights = np.empty_like(grid)
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return weights


def normalize_log_curve(
    grid: npt.NDArray[np.float64],
    log_density: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    grid = np.asarray(grid, dtype=float)
    log_density = np.asarray(log_density, dtype=float)
    if (log_density.shape != grid.shape or np.any(np.isnan(log_density))
            or np.any(np.isposinf(log_density)) or not np.any(np.isfinite(log_density))):
        message = "Scan log density must match the grid and have finite nonzero support"
        raise ValueError(message)
    shifted = np.exp(log_density - np.max(log_density))
    normalization = float(np.sum(trapezoid_weights(grid) * shifted))
    return shifted / normalization


def summarize_density(
    grid: npt.NDArray[np.float64],
    density: npt.NDArray[np.float64],
) -> dict[str, float]:
    grid, density = np.asarray(grid, dtype=float), np.asarray(density, dtype=float)
    trapezoid_weights(grid)  # validates grid before integration
    if density.shape != grid.shape or not np.all(np.isfinite(density)) or np.any(density < 0):
        message = "Scan density must be finite, nonnegative, and match the grid"
        raise ValueError(message)
    widths, left = np.diff(grid), grid[:-1]
    y0, delta = density[:-1], np.diff(density)
    area = widths * (density[:-1] + density[1:]) / 2
    total = area.sum()
    if not np.isfinite(total) or total <= 0:
        message = "Scan density must have finite positive integral"
        raise ValueError(message)
    first = np.sum(widths * (left * (y0 + delta/2) + widths * (y0/2 + delta/3))) / total
    second = np.sum(widths * (left**2 * (y0 + delta/2) + 2*left*widths * (y0/2 + delta/3)
                              + widths**2 * (y0/3 + delta/4))) / total
    cumulative = np.r_[0.0, np.cumsum(area)]
    quantiles = []
    for probability in [0.16, 0.5, 0.84]:
        target = probability * total
        index = min(np.searchsorted(cumulative, target, side="right") - 1, len(widths)-1)
        scaled_area = (target - cumulative[index]) / widths[index]
        # Stable inverse of y0*t + delta*t**2/2 inside the selected cell.
        denominator = y0[index] + np.sqrt(max(y0[index]**2 + 2*delta[index]*scaled_area, 0))
        fraction = 2*scaled_area / denominator if scaled_area > 0 else 0.0
        quantiles.append(float(grid[index] + widths[index]*fraction))
    return {"mean": float(first), "std": float(np.sqrt(max(second-first**2, 0))),
            "q16": quantiles[0], "median": quantiles[1], "q84": quantiles[2],
            "map": float(grid[np.argmax(density)])}


def _sampled_parameter_grid(
    alpha_grid: npt.NDArray[np.float64],
    medium: MediumPoint,
    sampled_names: list[str],
) -> npt.NDArray[np.float64]:
    values = {
        "AlphaS": alpha_grid,
        "nucleon_width": np.full_like(alpha_grid, medium.nucleon_width),
        "normalization": np.full_like(alpha_grid, medium.normalization),
    }
    missing = set(sampled_names) - set(values)
    if missing:
        msg = f"Fixed-medium scan does not define sampled parameters {sorted(missing)}."
        raise ValueError(msg)
    return np.column_stack([values[name] for name in sampled_names])


def evaluate_medium(
    *,
    alpha_grid: npt.NDArray[np.float64],
    medium: MediumPoint,
    groups: list[GroupLikelihoodInputs],
    analysis_config: dict,
    parameterization: str,
    batch_size: int,
) -> dict:
    parameter_info = parameterization_info(analysis_config, parameterization)
    sampled = _sampled_parameter_grid(alpha_grid, medium, parameter_info.sampled_names)
    full = parameter_info.expand_sampled_to_full(sampled, sampled_names=parameter_info.sampled_names)
    components, _, residuals = evaluate_group_likelihoods(full, groups, batch_size=batch_size)
    densities = {
        name: normalize_log_curve(alpha_grid, values)
        for name, values in components.items()
    }
    return {
        "log_likelihood": components,
        "density": densities,
        "summary": {
            name: summarize_density(alpha_grid, density)
            for name, density in densities.items()
        },
        "maximum_observable_sum_residual": residuals,
    }


def _sampled_fixed_summary(path: Path) -> dict[str, float]:
    with h5py.File(path, "r") as stream:
        samples = np.asarray(stream["posterior_samples"], dtype=float)
        weights = (
            np.asarray(stream["weights"], dtype=float)
            if "weights" in stream
            else np.ones(len(samples), dtype=float)
        )
        names = _decode_names(np.asarray(stream["parameter_names"][:]))
    alpha = samples[:, names.index("AlphaS")]
    weights = normalized_weights(weights, len(samples))
    q16, median, q84 = weighted_quantile(alpha, [0.16, 0.5, 0.84], weights)
    mean = float(weights @ alpha)
    variance = float(weights @ np.square(alpha - mean))
    return {
        "sample_count": len(alpha),
        "mean": mean,
        "std": float(np.sqrt(max(variance, 0.0))),
        "q16": float(q16),
        "median": float(median),
        "q84": float(q84),
    }


def _write_curve_table(
    path: Path,
    alpha_grid: npt.NDArray[np.float64],
    media: list[MediumPoint],
    results: dict[str, dict],
) -> None:
    group_names = list(results[media[0].key]["density"])
    fields = ["medium", "nucleon_width", "normalization", "alpha_s"] + [
        f"density_{name}" for name in group_names
    ] + [f"log_likelihood_{name}" for name in group_names]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for medium in media:
            result = results[medium.key]
            for index, alpha in enumerate(alpha_grid):
                row = {
                    "medium": medium.key,
                    "nucleon_width": medium.nucleon_width,
                    "normalization": medium.normalization,
                    "alpha_s": float(alpha),
                }
                row.update(
                    {
                        f"density_{name}": float(values[index])
                        for name, values in result["density"].items()
                    }
                )
                row.update(
                    {
                        f"log_likelihood_{name}": float(values[index])
                        for name, values in result["log_likelihood"].items()
                    }
                )
                writer.writerow(row)


def _plot_scan(
    path: Path,
    alpha_grid: npt.NDArray[np.float64],
    media: list[MediumPoint],
    results: dict[str, dict],
    x_limits: tuple[float, float] | None,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 13,
            "legend.fontsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for medium in media:
        axes[0].plot(
            alpha_grid,
            results[medium.key]["density"]["total"],
            color=medium.color,
            linewidth=2.2,
            label=medium.label,
        )
    axes[0].set_xlabel(r"Coupling $\alpha_s$")
    axes[0].set_ylabel("Probability density")
    axes[0].legend(frameon=False)
    if x_limits is not None:
        axes[0].set_xlim(*x_limits)

    ordered = sorted(media, key=lambda medium: medium.normalization)
    for medium in ordered:
        summary = results[medium.key]["summary"]["total"]
        axes[1].errorbar(
            medium.normalization,
            summary["mean"],
            yerr=[[summary["mean"] - summary["q16"]], [summary["q84"] - summary["mean"]]],
            color=medium.color,
            marker=medium.marker,
            markersize=7,
            capsize=3,
            linestyle="none",
        )
        axes[1].annotate(
            rf"$w={medium.nucleon_width:.3f}$ fm",
            (medium.normalization, summary["mean"]),
            xytext=(5, 7),
            textcoords="offset points",
            color=medium.color,
            fontsize=8,
        )
    axes[1].plot(
        [medium.normalization for medium in ordered],
        [results[medium.key]["summary"]["total"]["mean"] for medium in ordered],
        color="0.65",
        linewidth=1.0,
        zorder=0,
    )
    axes[1].set_xlabel(r"Fixed normalization $N$")
    axes[1].set_ylabel(r"Conditional $\alpha_s$")
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
