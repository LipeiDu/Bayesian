"""Compare hard-likelihood components across posterior sample sets.

The diagnostic intentionally reevaluates every sample with one common set of
emulator and observable artifacts. This avoids interpreting offsets between
likelihood values serialized by separate Bayesian runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt

from bayesian import data_IO, log_posterior
from bayesian.emulation import base
from bayesian.parameterization import parameterization_info

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PosteriorSamples:
    label: str
    source: Path
    samples: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    source_indices: npt.NDArray[np.int64]
    parameter_names: tuple[str, ...]
    source_sample_count: int


@dataclass(frozen=True)
class GroupLikelihoodInputs:
    name: str
    emulator_config: base.ConcreteEmulatorConfig
    emulator_results: dict
    emulator_cov_unexplained: npt.NDArray[np.float64]
    data_y: npt.NDArray[np.float64]
    data_y_err: npt.NDArray[np.float64]
    observable_slices: dict[str, slice]


def _decode_names(values: npt.NDArray) -> tuple[str, ...]:
    return tuple(value.decode() if isinstance(value, bytes) else str(value) for value in values)


def _normalize_weights(weights: npt.ArrayLike) -> npt.NDArray[np.float64]:
    normalized = np.asarray(weights, dtype=float)
    if normalized.ndim != 1 or np.any(normalized < 0.0):
        msg = "Posterior weights must be a one-dimensional nonnegative array."
        raise ValueError(msg)
    total = float(np.sum(normalized))
    if not np.isfinite(total) or total <= 0.0:
        msg = "Posterior weights must have a finite positive sum."
        raise ValueError(msg)
    return normalized / total


def weighted_quantile(
    values: npt.ArrayLike,
    probabilities: npt.ArrayLike,
    weights: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    values_array = np.asarray(values, dtype=float)
    probabilities_array = np.asarray(probabilities, dtype=float)
    weights_array = _normalize_weights(weights)
    if values_array.ndim != 1 or values_array.shape != weights_array.shape:
        msg = "Values and weights must be one-dimensional arrays with equal shape."
        raise ValueError(msg)
    if np.any((probabilities_array < 0.0) | (probabilities_array > 1.0)):
        msg = "Quantile probabilities must lie in [0, 1]."
        raise ValueError(msg)
    order = np.argsort(values_array)
    sorted_values = values_array[order]
    sorted_weights = weights_array[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    return np.interp(probabilities_array, cumulative, sorted_values)


def weighted_summary(values: npt.ArrayLike, weights: npt.ArrayLike) -> dict[str, float]:
    values_array = np.asarray(values, dtype=float)
    weights_array = _normalize_weights(weights)
    mean = float(np.sum(weights_array * values_array))
    variance = float(np.sum(weights_array * (values_array - mean) ** 2))
    q16, median, q84 = weighted_quantile(values_array, [0.16, 0.5, 0.84], weights_array)
    return {
        "mean": mean,
        "std": float(np.sqrt(max(variance, 0.0))),
        "q16": float(q16),
        "median": float(median),
        "q84": float(q84),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
    }


def _select_indices(
    sample_count: int,
    max_samples: int | None,
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    if max_samples is None or max_samples <= 0 or max_samples >= sample_count:
        return np.arange(sample_count, dtype=np.int64)
    return np.sort(rng.choice(sample_count, size=max_samples, replace=False)).astype(np.int64)


def load_posterior_samples(
    path: Path,
    *,
    label: str,
    max_samples: int | None,
    seed: int,
) -> PosteriorSamples:
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as stream:
        source_sample_count = int(stream["posterior_samples"].shape[0])
        indices = _select_indices(source_sample_count, max_samples, rng)
        samples = np.asarray(stream["posterior_samples"][indices], dtype=float)
        parameter_names = _decode_names(np.asarray(stream["parameter_names"][:]))
        if "weights" in stream:
            source_weights = np.asarray(stream["weights"][:], dtype=float)
            weights = source_weights[indices]
        else:
            weights = np.ones(len(indices), dtype=float)
    return PosteriorSamples(
        label=label,
        source=path.resolve(),
        samples=samples,
        weights=_normalize_weights(weights),
        source_indices=indices,
        parameter_names=parameter_names,
        source_sample_count=source_sample_count,
    )


def _reorder_and_expand_samples(
    posterior: PosteriorSamples,
    *,
    analysis_config: dict,
    parameterization: str,
) -> npt.NDArray[np.float64]:
    info = parameterization_info(analysis_config, parameterization)
    missing = set(info.sampled_names) - set(posterior.parameter_names)
    if missing:
        msg = f"Posterior {posterior.source} is missing sampled parameters {sorted(missing)}."
        raise ValueError(msg)
    indices = [posterior.parameter_names.index(name) for name in info.sampled_names]
    ordered = posterior.samples[:, indices]
    return info.expand_sampled_to_full(ordered, sampled_names=info.sampled_names)


def _load_group_inputs(
    emulation_config: base.EmulatorOrganizationConfig,
) -> list[GroupLikelihoodInputs]:
    emulation_results = emulation_config.read_all_emulator_groups()
    covariance_unexplained = base.compute_emulator_cov_unexplained(emulation_config, emulation_results)
    observables = data_IO.read_dict_from_h5(
        emulation_config.output_dir,
        emulation_config.observables_filename,
        verbose=False,
    )
    groups = []
    for group_name, group_config in emulation_config.emulation_groups_config.items():
        experimental = data_IO.data_array_from_h5(
            emulation_config.output_dir,
            emulation_config.observables_filename,
            observable_filter=group_config.observable_filter,
        )
        observable_names = data_IO.sorted_observable_list_from_dict(
            observables,
            observable_filter=group_config.observable_filter,
        )
        observable_slices = {}
        start = 0
        for observable_name in observable_names:
            n_bins = int(observables["Data"][observable_name]["y"].shape[0])
            observable_slices[observable_name] = slice(start, start + n_bins)
            start += n_bins
        if start != len(experimental["y"]):
            msg = f"Observable slices for {group_name} cover {start} bins, expected {len(experimental['y'])}."
            raise ValueError(msg)
        groups.append(
            GroupLikelihoodInputs(
                name=group_name,
                emulator_config=group_config,
                emulator_results=emulation_results[group_name],
                emulator_cov_unexplained=covariance_unexplained[group_name],
                data_y=np.asarray(experimental["y"], dtype=float),
                data_y_err=np.asarray(experimental["y_err"], dtype=float),
                observable_slices=observable_slices,
            )
        )
    return groups


def _evaluate_group_batch(
    parameters: npt.NDArray[np.float64],
    group: GroupLikelihoodInputs,
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
    predictions = base.predict_emulation_group(
        parameters,
        group.emulator_results,
        group.emulator_config,
        emulator_group_cov_unexplained=group.emulator_cov_unexplained,
    )
    residual = np.asarray(predictions["central_value"], dtype=float) - group.data_y
    covariance = np.asarray(predictions["cov"], dtype=float)
    covariance += np.diag(group.data_y_err**2)[np.newaxis, :, :]
    group_values = np.asarray(
        [log_posterior._loglikelihood(y, cov) for y, cov in zip(residual, covariance, strict=True)],
        dtype=float,
    )
    observable_values = {}
    for observable_name, observable_slice in group.observable_slices.items():
        observable_values[observable_name] = np.asarray(
            [
                log_posterior._loglikelihood(y[observable_slice], cov[observable_slice, observable_slice])
                for y, cov in zip(residual, covariance, strict=True)
            ],
            dtype=float,
        )
    return group_values, observable_values


def evaluate_group_likelihoods(
    parameters: npt.NDArray[np.float64],
    groups: list[GroupLikelihoodInputs],
    *,
    batch_size: int,
) -> tuple[
    dict[str, npt.NDArray[np.float64]],
    dict[str, npt.NDArray[np.float64]],
    dict[str, float],
]:
    if batch_size <= 0:
        msg = "Batch size must be positive."
        raise ValueError(msg)
    output = {group.name: np.empty(len(parameters), dtype=float) for group in groups}
    observable_output = {
        observable_name: np.empty(len(parameters), dtype=float)
        for group in groups
        for observable_name in group.observable_slices
    }
    for start in range(0, len(parameters), batch_size):
        stop = min(start + batch_size, len(parameters))
        for group in groups:
            group_values, observable_values = _evaluate_group_batch(parameters[start:stop], group)
            output[group.name][start:stop] = group_values
            for observable_name, values in observable_values.items():
                observable_output[observable_name][start:stop] = values
    additivity_residuals = {}
    for group in groups:
        observable_sum = np.sum(
            np.vstack([observable_output[name] for name in group.observable_slices]),
            axis=0,
        )
        residual = output[group.name] - observable_sum
        additivity_residuals[group.name] = float(np.max(np.abs(residual)))
    output["total"] = np.sum(np.vstack(list(output.values())), axis=0)
    return output, observable_output, additivity_residuals


def _effective_sample_size(weights: npt.ArrayLike) -> float:
    normalized = _normalize_weights(weights)
    return float(1.0 / np.sum(normalized**2))


def _write_sample_table(
    output_path: Path,
    posteriors: list[PosteriorSamples],
    components: dict[str, dict[str, npt.NDArray[np.float64]]],
) -> None:
    group_names = [name for name in next(iter(components.values())) if name != "total"]
    fieldnames = ["posterior", "source_index", "weight", *[f"logL_{name}" for name in group_names], "logL_total"]
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for posterior in posteriors:
            values = components[posterior.label]
            for i, source_index in enumerate(posterior.source_indices):
                row = {
                    "posterior": posterior.label,
                    "source_index": int(source_index),
                    "weight": f"{posterior.weights[i]:.17g}",
                    "logL_total": f"{values['total'][i]:.17g}",
                }
                row.update({f"logL_{name}": f"{values[name][i]:.17g}" for name in group_names})
                writer.writerow(row)


def _write_observable_sample_table(
    output_path: Path,
    posteriors: list[PosteriorSamples],
    observable_components: dict[str, dict[str, npt.NDArray[np.float64]]],
) -> None:
    observable_names = list(next(iter(observable_components.values())))
    fieldnames = ["posterior", "source_index", "weight", *observable_names]
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for posterior in posteriors:
            values = observable_components[posterior.label]
            for i, source_index in enumerate(posterior.source_indices):
                row = {
                    "posterior": posterior.label,
                    "source_index": int(source_index),
                    "weight": f"{posterior.weights[i]:.17g}",
                }
                row.update({name: f"{values[name][i]:.17g}" for name in observable_names})
                writer.writerow(row)


def _plot_comparison(
    output_path: Path,
    posteriors: list[PosteriorSamples],
    components: dict[str, dict[str, npt.NDArray[np.float64]]],
    *,
    seed: int,
) -> None:
    colors = {posteriors[0].label: "#277da1", posteriors[1].label: "#d62828"}
    component_names = ["total", *[name for name in components[posteriors[0].label] if name != "total"]]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    histogram_axes = axes.flat[:3]
    for ax, component_name in zip(histogram_axes, component_names, strict=False):
        combined = np.concatenate([components[p.label][component_name] for p in posteriors])
        low, high = np.quantile(combined, [0.002, 0.998])
        bins = np.linspace(low, high, 60)
        for posterior in posteriors:
            ax.hist(
                components[posterior.label][component_name],
                bins=bins,
                weights=posterior.weights,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=colors[posterior.label],
                label=posterior.label,
            )
        short_name = "hard" if component_name == "total" else component_name.replace("_group", "")
        title = "Full hard likelihood" if component_name == "total" else short_name.title()
        ax.set_xlabel(rf"$\log\mathcal{{L}}_{{\rm {short_name}}}$")
        ax.set_ylabel("Probability density")
        ax.set_title(title)
    histogram_axes[0].legend(frameon=False)

    rng = np.random.default_rng(seed)
    scatter_ax = axes.flat[3]
    group_names = [name for name in component_names if name != "total"]
    if len(group_names) >= 2:
        for posterior in posteriors:
            n_draw = min(2000, len(posterior.weights))
            indices = rng.choice(len(posterior.weights), size=n_draw, replace=True, p=posterior.weights)
            scatter_ax.scatter(
                components[posterior.label][group_names[0]][indices],
                components[posterior.label][group_names[1]][indices],
                s=8,
                alpha=0.18,
                color=colors[posterior.label],
                label=posterior.label,
            )
        scatter_ax.set_xlabel(rf"$\log\mathcal{{L}}_{{\rm {group_names[0].replace('_group', '')}}}$")
        scatter_ax.set_ylabel(rf"$\log\mathcal{{L}}_{{\rm {group_names[1].replace('_group', '')}}}$")
        scatter_ax.set_title("Hadron-jet likelihood balance")
    else:
        scatter_ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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


def _plot_observable_differences(
    output_path: Path,
    posteriors: list[PosteriorSamples],
    observable_components: dict[str, dict[str, npt.NDArray[np.float64]]],
) -> None:
    reference, candidate = posteriors
    differences = {}
    for observable_name in observable_components[reference.label]:
        reference_mean = weighted_summary(
            observable_components[reference.label][observable_name],
            reference.weights,
        )["mean"]
        candidate_mean = weighted_summary(
            observable_components[candidate.label][observable_name],
            candidate.weights,
        )["mean"]
        differences[observable_name] = candidate_mean - reference_mean
    ordered = sorted(differences, key=differences.get)
    values = np.asarray([differences[name] for name in ordered])
    colors = np.where(values < 0.0, "#d62828", "#277da1")
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    positions = np.arange(len(ordered))
    ax.barh(positions, values, color=colors, alpha=0.88)
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_yticks(positions, [_short_observable_label(name) for name in ordered])
    ax.set_xlabel(r"$\Delta\langle\log\mathcal{L}\rangle$ (sequential $-$ hard only)")
    ax.set_title("Observable contributions to hard-likelihood change")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_summary(
    *,
    config_path: Path,
    emulation_config: base.EmulatorOrganizationConfig,
    posteriors: list[PosteriorSamples],
    components: dict[str, dict[str, npt.NDArray[np.float64]]],
    observable_components: dict[str, dict[str, npt.NDArray[np.float64]]],
    additivity_residuals: dict[str, dict[str, float]],
    groups: list[GroupLikelihoodInputs],
    artifacts: dict[str, Path],
) -> dict:
    summaries = {}
    for posterior in posteriors:
        summaries[posterior.label] = {
            "source": str(posterior.source),
            "source_sample_count": posterior.source_sample_count,
            "evaluated_sample_count": len(posterior.samples),
            "effective_sample_size": _effective_sample_size(posterior.weights),
            "components": {
                name: weighted_summary(values, posterior.weights)
                for name, values in components[posterior.label].items()
            },
            "observables": {
                name: weighted_summary(values, posterior.weights)
                for name, values in observable_components[posterior.label].items()
            },
            "maximum_observable_sum_residual_by_group": additivity_residuals[posterior.label],
        }
    reference, candidate = posteriors
    mean_differences = {
        name: summaries[candidate.label]["components"][name]["mean"]
        - summaries[reference.label]["components"][name]["mean"]
        for name in components[reference.label]
    }
    observable_mean_differences = {
        name: summaries[candidate.label]["observables"][name]["mean"]
        - summaries[reference.label]["observables"][name]["mean"]
        for name in observable_components[reference.label]
    }
    return {
        "method": "common_emulator_hard_likelihood_component_comparison",
        "config": str(config_path.resolve()),
        "common_emulator_directory": str(emulation_config.output_dir.resolve()),
        "likelihood_convention": "-0.5 residual^T covariance^-1 residual - 0.5 log(det(covariance)); common additive constants omitted",
        "experimental_covariance": "diagonal experimental variances",
        "emulator_covariance": "parameter-dependent predictive covariance",
        "groups": {group.name: {"n_bins": len(group.data_y)} for group in groups},
        "posteriors": summaries,
        "mean_differences_candidate_minus_reference": mean_differences,
        "observable_mean_differences_candidate_minus_reference": observable_mean_differences,
        "artifacts": {name: str(path.resolve()) for name, path in artifacts.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/reduced_soft_hard.yaml"))
    parser.add_argument("--analysis-name", default="reduced_soft_hard")
    parser.add_argument("--parameterization", default="exponential")
    parser.add_argument(
        "--hard-posterior",
        type=Path,
        default=Path("output/reduced_soft_hard_preprocessed_noPCA/reduced_soft_hard_exponential/posterior.h5"),
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
        "--output-dir",
        type=Path,
        default=Path("output/hard_likelihood_posterior_comparison"),
    )
    parser.add_argument("--max-hard-samples", type=int, default=10000)
    parser.add_argument("--max-sequential-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=314159)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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
    posteriors = [
        load_posterior_samples(
            args.hard_posterior,
            label="Hard obs. only",
            max_samples=args.max_hard_samples,
            seed=args.seed,
        ),
        load_posterior_samples(
            args.sequential_posterior,
            label="Sequential",
            max_samples=args.max_sequential_samples,
            seed=args.seed + 1,
        ),
    ]
    components = {}
    observable_components = {}
    additivity_residuals = {}
    for posterior in posteriors:
        parameters = _reorder_and_expand_samples(
            posterior,
            analysis_config=analysis_config,
            parameterization=args.parameterization,
        )
        group_values, observable_values, residuals = evaluate_group_likelihoods(
            parameters,
            groups,
            batch_size=args.batch_size,
        )
        components[posterior.label] = group_values
        observable_components[posterior.label] = observable_values
        additivity_residuals[posterior.label] = residuals

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "hard_likelihood_components.csv"
    observable_csv_path = args.output_dir / "hard_likelihood_observable_components.csv"
    figure_path = args.output_dir / "hard_likelihood_component_comparison.png"
    observable_figure_path = args.output_dir / "hard_likelihood_observable_differences.png"
    summary_path = args.output_dir / "hard_likelihood_component_comparison.json"
    _write_sample_table(csv_path, posteriors, components)
    _write_observable_sample_table(observable_csv_path, posteriors, observable_components)
    _plot_comparison(figure_path, posteriors, components, seed=args.seed)
    _plot_observable_differences(observable_figure_path, posteriors, observable_components)
    summary = _build_summary(
        config_path=args.config,
        emulation_config=emulation_config,
        posteriors=posteriors,
        components=components,
        observable_components=observable_components,
        additivity_residuals=additivity_residuals,
        groups=groups,
        artifacts={
            "sample_table": csv_path,
            "observable_sample_table": observable_csv_path,
            "figure": figure_path,
            "observable_figure": observable_figure_path,
            "summary": summary_path,
        },
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(
        "Mean likelihood differences (sequential minus hard-only):\n%s",
        json.dumps(summary["mean_differences_candidate_minus_reference"], indent=2),
    )


if __name__ == "__main__":
    main()
