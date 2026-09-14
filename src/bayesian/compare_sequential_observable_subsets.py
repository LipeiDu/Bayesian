"""Compare soft, hadron-only, jet-only, and full sequential posteriors."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import gaussian_kde

mpl.use("Agg")
import matplotlib.pyplot as plt

from bayesian.hard_likelihood_diagnostics import weighted_summary


@dataclass(frozen=True)
class DensitySamples:
    label: str
    samples: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    parameter_names: tuple[str, ...]


def _decode_names(values: npt.NDArray) -> tuple[str, ...]:
    return tuple(value.decode() if isinstance(value, bytes) else str(value) for value in values)


def load_posterior(path: Path, label: str) -> DensitySamples:
    with h5py.File(path, "r") as stream:
        samples = np.asarray(stream["posterior_samples"], dtype=float)
        weights = (
            np.asarray(stream["weights"], dtype=float)
            if "weights" in stream
            else np.ones(len(samples), dtype=float)
        )
        parameter_names = _decode_names(np.asarray(stream["parameter_names"][:]))
    weights /= np.sum(weights)
    return DensitySamples(label, samples, weights, parameter_names)


def load_soft_projection(path: Path) -> DensitySamples:
    frame = pd.read_csv(path)
    names = ("nucleon_width", "normalization")
    samples = frame.loc[:, names].to_numpy(dtype=float)
    weights = np.full(len(samples), 1.0 / len(samples))
    return DensitySamples("Projected soft", samples, weights, names)


def _parameter_values(samples: DensitySamples, name: str) -> npt.NDArray[np.float64]:
    return samples.samples[:, samples.parameter_names.index(name)]


def _credible_density_thresholds(
    density: npt.NDArray[np.float64],
    probabilities: tuple[float, ...],
) -> list[float]:
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]
    return sorted(float(ordered[np.searchsorted(cumulative, probability)]) for probability in probabilities)


def _plot_medium_contours(
    ax: plt.Axes,
    sample_sets: list[DensitySamples],
    colors: dict[str, str],
) -> None:
    all_width = np.concatenate([_parameter_values(samples, "nucleon_width") for samples in sample_sets])
    all_norm = np.concatenate([_parameter_values(samples, "normalization") for samples in sample_sets])
    width_grid = np.linspace(np.quantile(all_width, 0.002), np.quantile(all_width, 0.998), 180)
    norm_grid = np.linspace(np.quantile(all_norm, 0.002), np.quantile(all_norm, 0.998), 180)
    width_mesh, norm_mesh = np.meshgrid(width_grid, norm_grid)
    positions = np.vstack([width_mesh.ravel(), norm_mesh.ravel()])
    handles = []
    for samples in sample_sets:
        values = np.vstack(
            [
                _parameter_values(samples, "nucleon_width"),
                _parameter_values(samples, "normalization"),
            ]
        )
        density = gaussian_kde(values, weights=samples.weights)(positions).reshape(width_mesh.shape)
        levels = _credible_density_thresholds(density, (0.90, 0.50))
        ax.contour(
            width_mesh,
            norm_mesh,
            density,
            levels=levels,
            colors=colors[samples.label],
            linewidths=(1.5, 2.2),
        )
        handles.append(plt.Line2D([], [], color=colors[samples.label], linewidth=2.2, label=samples.label))
    ax.set_xlabel(r"Nucleon width $w$ [fm]")
    ax.set_ylabel(r"Normalization $N$")
    ax.legend(handles=handles, frameon=False, fontsize=9)


def _plot_marginal(
    ax: plt.Axes,
    sample_sets: list[DensitySamples],
    parameter_name: str,
    colors: dict[str, str],
    xlabel: str,
) -> None:
    available = [samples for samples in sample_sets if parameter_name in samples.parameter_names]
    combined = np.concatenate([_parameter_values(samples, parameter_name) for samples in available])
    grid = np.linspace(np.quantile(combined, 0.001), np.quantile(combined, 0.999), 350)
    for samples in available:
        values = _parameter_values(samples, parameter_name)
        density = gaussian_kde(values, weights=samples.weights)(grid)
        ax.plot(grid, density, color=colors[samples.label], linewidth=2.0, label=samples.label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")


def _summary(sample_sets: list[DensitySamples]) -> dict:
    output = {}
    for samples in sample_sets:
        output[samples.label] = {
            "sample_count": len(samples.samples),
            "effective_sample_size": float(1.0 / np.sum(samples.weights**2)),
            "parameters": {
                name: weighted_summary(_parameter_values(samples, name), samples.weights)
                for name in samples.parameter_names
            },
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--soft-projection",
        type=Path,
        default=Path(
            "/Users/Lipei/Documents/Research/heavy-ion-research/projects/STAT_soft_hard/"
            "supporting_workflows/soft_posterior_projection/projected_soft_posterior_2d.csv"
        ),
    )
    parser.add_argument(
        "--full-posterior",
        type=Path,
        default=Path(
            "output/reduced_soft_hard_sequential_pocomc_preprocessed_noPCA/"
            "reduced_soft_hard_exponential/posterior.h5"
        ),
    )
    parser.add_argument(
        "--hadron-posterior",
        type=Path,
        default=Path(
            "output/reduced_soft_hard_sequential_pocomc_kde_hadron_only/"
            "reduced_soft_hard_exponential/posterior.h5"
        ),
    )
    parser.add_argument(
        "--jet-posterior",
        type=Path,
        default=Path(
            "output/reduced_soft_hard_sequential_pocomc_kde_jet_only/"
            "reduced_soft_hard_exponential/posterior.h5"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/sequential_observable_subset_comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_sets = [
        load_soft_projection(args.soft_projection),
        load_posterior(args.hadron_posterior, "Hadron only"),
        load_posterior(args.jet_posterior, "Jet only"),
        load_posterior(args.full_posterior, "All hard data"),
    ]
    colors = {
        "Projected soft": "#2a9d3f",
        "Hadron only": "#e07a1f",
        "Jet only": "#277da1",
        "All hard data": "#d62828",
    }
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 13,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2))
    _plot_medium_contours(axes[0, 0], sample_sets, colors)
    _plot_marginal(axes[0, 1], sample_sets, "AlphaS", colors, r"Coupling $\alpha_s$")
    _plot_marginal(axes[1, 0], sample_sets, "nucleon_width", colors, r"Nucleon width $w$ [fm]")
    _plot_marginal(axes[1, 1], sample_sets, "normalization", colors, r"Normalization $N$")
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "sequential_observable_subset_comparison.png"
    summary_path = args.output_dir / "sequential_observable_subset_comparison.json"
    fig.savefig(figure_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    summary_path.write_text(json.dumps(_summary(sample_sets), indent=2) + "\n")


if __name__ == "__main__":
    main()
