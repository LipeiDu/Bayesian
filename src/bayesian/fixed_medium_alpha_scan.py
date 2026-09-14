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
from bayesian.hard_likelihood_diagnostics import (
    GroupLikelihoodInputs,
    _load_group_inputs,
    evaluate_group_likelihoods,
)
from bayesian.parameterization import parameterization_info

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


def _normalized_weights(weights: npt.ArrayLike) -> npt.NDArray[np.float64]:
    output = np.asarray(weights, dtype=float)
    output /= np.sum(output)
    return output


def load_posterior_mean(path: Path) -> dict[str, float]:
    with h5py.File(path, "r") as stream:
        samples = np.asarray(stream["posterior_samples"], dtype=float)
        weights = (
            np.asarray(stream["weights"], dtype=float)
            if "weights" in stream
            else np.ones(len(samples), dtype=float)
        )
        names = _decode_names(np.asarray(stream["parameter_names"][:]))
    weights = _normalized_weights(weights)
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
    weights = np.empty_like(grid)
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return weights


def normalize_log_curve(
    grid: npt.NDArray[np.float64],
    log_density: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    shifted = np.exp(log_density - np.max(log_density))
    normalization = float(np.sum(trapezoid_weights(grid) * shifted))
    return shifted / normalization


def summarize_density(
    grid: npt.NDArray[np.float64],
    density: npt.NDArray[np.float64],
) -> dict[str, float]:
    integration_weights = trapezoid_weights(grid)
    probability_weights = integration_weights * density
    probability_weights /= np.sum(probability_weights)
    cumulative = np.cumsum(probability_weights)
    mean = float(np.sum(probability_weights * grid))
    variance = float(np.sum(probability_weights * np.square(grid - mean)))
    return {
        "mean": mean,
        "std": float(np.sqrt(max(variance, 0.0))),
        "q16": float(np.interp(0.16, cumulative, grid)),
        "median": float(np.interp(0.50, cumulative, grid)),
        "q84": float(np.interp(0.84, cumulative, grid)),
        "map": float(grid[np.argmax(density)]),
    }


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
    weights = _normalized_weights(weights)
    order = np.argsort(alpha)
    cumulative = np.cumsum(weights[order])
    mean = float(weights @ alpha)
    variance = float(weights @ np.square(alpha - mean))
    return {
        "sample_count": len(alpha),
        "mean": mean,
        "std": float(np.sqrt(max(variance, 0.0))),
        "q16": float(np.interp(0.16, cumulative, alpha[order])),
        "median": float(np.interp(0.50, cumulative, alpha[order])),
        "q84": float(np.interp(0.84, cumulative, alpha[order])),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/reduced_soft_hard.yaml"))
    parser.add_argument("--analysis-name", default="reduced_soft_hard")
    parser.add_argument("--parameterization", default="exponential")
    parser.add_argument(
        "--soft-projection",
        type=Path,
        default=Path(
            "/Users/Lipei/Documents/Research/heavy-ion-research/projects/STAT_soft_hard/"
            "supporting_workflows/soft_posterior_projection/projected_soft_posterior_2d.csv"
        ),
    )
    parser.add_argument(
        "--sequential-posterior",
        type=Path,
        default=Path(
            "output/reduced_soft_hard_sequential_pocomc_preprocessed_noPCA/"
            "reduced_soft_hard_exponential/posterior.h5"
        ),
    )
    parser.add_argument(
        "--hard-posterior",
        type=Path,
        default=Path(
            "output/reduced_soft_hard_preprocessed_noPCA/"
            "reduced_soft_hard_exponential/posterior.h5"
        ),
    )
    parser.add_argument(
        "--fixed-posterior",
        type=Path,
        default=Path(
            "output/fixed_soft_comparison_preprocessed_noPCA/"
            "fixed_soft_comparison_exponential/posterior.h5"
        ),
    )
    parser.add_argument("--fixed-width", type=float, default=1.12)
    parser.add_argument("--fixed-normalization", type=float, default=14.2)
    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=0.5)
    parser.add_argument("--alpha-grid-points", type=int, default=1200)
    parser.add_argument("--display-alpha-min", type=float, default=0.28)
    parser.add_argument("--display-alpha-max", type=float, default=0.39)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/fixed_medium_alpha_scan"),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    analysis_config = config["analyses"][args.analysis_name]
    emulation_config = base.EmulatorOrganizationConfig.from_config_file(
        analysis_name=args.analysis_name,
        parameterization=args.parameterization,
        config_file=args.config,
        analysis_config=analysis_config,
    )
    groups = _load_group_inputs(emulation_config)
    media = build_medium_points(
        soft_projection=args.soft_projection,
        sequential_posterior=args.sequential_posterior,
        hard_posterior=args.hard_posterior,
        fixed_width=args.fixed_width,
        fixed_normalization=args.fixed_normalization,
    )
    alpha_grid = np.linspace(args.alpha_min, args.alpha_max, args.alpha_grid_points)
    results = {
        medium.key: evaluate_medium(
            alpha_grid=alpha_grid,
            medium=medium,
            groups=groups,
            analysis_config=analysis_config,
            parameterization=args.parameterization,
            batch_size=args.batch_size,
        )
        for medium in media
    }
    fixed_grid_summary = results["source_fixed_medium"]["summary"]["total"]
    fixed_sample_summary = _sampled_fixed_summary(args.fixed_posterior)
    validation = {
        "grid": fixed_grid_summary,
        "sampled_posterior": fixed_sample_summary,
        "mean_difference_grid_minus_sampled": fixed_grid_summary["mean"]
        - fixed_sample_summary["mean"],
        "median_difference_grid_minus_sampled": fixed_grid_summary["median"]
        - fixed_sample_summary["median"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.output_dir / "fixed_medium_alpha_scan_curves.csv"
    summary_path = args.output_dir / "fixed_medium_alpha_scan.json"
    figure_path = args.output_dir / "fixed_medium_alpha_scan.png"
    _write_curve_table(curve_path, alpha_grid, media, results)
    _plot_scan(
        figure_path,
        alpha_grid,
        media,
        results,
        (args.display_alpha_min, args.display_alpha_max),
    )
    payload = {
        "method": "direct_common_emulator_fixed_medium_alpha_grid",
        "definition": "p(alpha_s | N,w,D_H) proportional to L_H(D_H | N,w,alpha_s) for a flat alpha_s prior",
        "config": str(args.config.resolve()),
        "alpha_grid": {
            "min": args.alpha_min,
            "max": args.alpha_max,
            "points": args.alpha_grid_points,
        },
        "media": {
            medium.key: {
                "label": medium.label,
                "nucleon_width": medium.nucleon_width,
                "normalization": medium.normalization,
                "summary": results[medium.key]["summary"],
            }
            for medium in media
        },
        "fixed_medium_validation": validation,
        "artifacts": {
            "curves": str(curve_path.resolve()),
            "figure": str(figure_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(
        "Conditional posterior summaries:\n%s",
        json.dumps({key: value["summary"]["total"] for key, value in results.items()}, indent=2),
    )
    logger.info("Fixed-medium validation:\n%s", json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
