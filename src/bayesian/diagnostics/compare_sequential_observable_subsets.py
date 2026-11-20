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

from bayesian.diagnostics.hard_likelihood_diagnostics import weighted_summary
from bayesian.posterior_utils import normalized_weights, weighted_quantile


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
    weights = normalized_weights(weights, len(samples))
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


def _weighted_limits(sample_sets, parameter_name, tail):
    limits = np.array([weighted_quantile(_parameter_values(samples, parameter_name), [tail, 1-tail], samples.weights)
                       for samples in sample_sets])
    return float(limits[:, 0].min()), float(limits[:, 1].max())


def _plot_medium_contours(
    ax: plt.Axes,
    sample_sets: list[DensitySamples],
    colors: dict[str, str],
) -> None:
    width_grid = np.linspace(*_weighted_limits(sample_sets, "nucleon_width", 0.002), 180)
    norm_grid = np.linspace(*_weighted_limits(sample_sets, "normalization", 0.002), 180)
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
    grid = np.linspace(*_weighted_limits(available, parameter_name, 0.001), 350)
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
