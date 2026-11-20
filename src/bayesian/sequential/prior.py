from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from scipy.special import expit
from scipy.stats import truncnorm

from bayesian import prior as prior_module
from bayesian.sequential.flow import load_realnvp_2d_from_payload
from bayesian.sequential.config import (
    SequentialInferenceConfig,
    SoftDensityArtifactMetadata,
    _load_artifact_payload,
    validate_soft_density_artifact_payload,
)

logger = logging.getLogger(__name__)


def build_sequential_log_prior_fn(
    *,
    sequential_config: SequentialInferenceConfig | None,
    combined_prior_config: dict[str, Any] | None,
    sampled_parameter_names: list[str],
    sampled_parameter_min: list[float] | npt.NDArray[np.float64] | None = None,
    sampled_parameter_max: list[float] | npt.NDArray[np.float64] | None = None,
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None:
    if sequential_config is None:
        return None

    soft_parameter_names = list(sequential_config.soft_parameter_names)
    model_file = sequential_config.soft_density_model_file

    missing_names = [name for name in soft_parameter_names if name not in sampled_parameter_names]
    if missing_names:
        raise ValueError(
            "Sequential inference requires the soft-parameter names to be sampled in the Bayesian run. "
            f"Missing names: {missing_names}"
        )

    soft_indices = [sampled_parameter_names.index(name) for name in soft_parameter_names]
    remaining_parameter_names = [name for name in sampled_parameter_names if name not in soft_parameter_names]
    remaining_indices = [sampled_parameter_names.index(name) for name in remaining_parameter_names]

    remainder_prior_config = _subset_manual_prior_config(
        prior_config=combined_prior_config,
        sampled_parameter_names=sampled_parameter_names,
        target_names=remaining_parameter_names,
    )
    remainder_log_prior_fn = prior_module.make_log_prior_fn(remainder_prior_config, remaining_parameter_names)
    soft_log_density_fn = load_soft_density_model(
        sequential_config=sequential_config,
    )

    logger.info(
        "Sequential inference enabled. Replacing prior on %s with soft density model %s",
        soft_parameter_names,
        model_file,
    )

    def _log_prior(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        X = np.array(X, dtype=float, ndmin=2)
        logp = soft_log_density_fn(X[:, soft_indices])
        if remaining_indices:
            logp += remainder_log_prior_fn(X[:, remaining_indices])
        return logp

    sampler_fn = _build_sequential_prior_sampler(
        soft_log_density_fn=soft_log_density_fn,
        soft_indices=soft_indices,
        remaining_indices=remaining_indices,
        sampled_parameter_names=sampled_parameter_names,
        remainder_log_prior_fn=remainder_log_prior_fn,
        remainder_prior_config=remainder_prior_config,
        sampled_parameter_min=sampled_parameter_min,
        sampled_parameter_max=sampled_parameter_max,
    )
    if sampler_fn is not None:
        _log_prior.sample = sampler_fn

    return _log_prior


def _subset_manual_prior_config(
    *,
    prior_config: dict[str, Any] | None,
    sampled_parameter_names: list[str],
    target_names: list[str],
) -> dict[str, Any] | None:
    if not target_names:
        return None
    if prior_config is None:
        return None
    if "prior_source" in prior_config:
        raise ValueError(
            "Sequential inference does not support selectively replacing a correlated prior_source prior. "
            "Please use explicit per-parameter priors for the sampled parameters in sequential mode."
        )

    selected_indices = [sampled_parameter_names.index(name) for name in target_names]
    full_type = _expand_prior_field(
        values=prior_config.get("type", []),
        expected_length=len(sampled_parameter_names),
        default="uniform",
    )
    full_mean = _expand_prior_field(
        values=prior_config.get("mean", []),
        expected_length=len(sampled_parameter_names),
        default=None,
    )
    full_std = _expand_prior_field(
        values=prior_config.get("std", []),
        expected_length=len(sampled_parameter_names),
        default=None,
    )
    subset: dict[str, Any] = {}
    subset["type"] = [full_type[i] for i in selected_indices]
    subset["mean"] = [full_mean[i] for i in selected_indices]
    subset["std"] = [full_std[i] for i in selected_indices]
    return subset if subset else None


def _expand_prior_field(
    *,
    values: list[Any],
    expected_length: int,
    default: Any,
) -> list[Any]:
    expanded = list(values[:expected_length])
    if len(expanded) < expected_length:
        expanded.extend([default] * (expected_length - len(expanded)))
    return expanded


def load_soft_density_model(
    *,
    sequential_config: SequentialInferenceConfig,
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    if sequential_config.soft_density_model_type == "gaussian_kde":
        return _load_kde_soft_density_model(sequential_config=sequential_config)
    if sequential_config.soft_density_model_type == "torch_realnvp_2d":
        return _load_flow_soft_density_model(sequential_config=sequential_config)
    raise NotImplementedError(
        "Sequential inference currently supports only gaussian_kde and torch_realnvp_2d soft models, "
        f"not '{sequential_config.soft_density_model_type}'."
    )


def _load_kde_soft_density_model(
    *,
    sequential_config: SequentialInferenceConfig,
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    model_file = sequential_config.soft_density_model_file

    payload = _load_artifact_payload(model_file=model_file, model_type="gaussian_kde")

    metadata = _validate_loaded_artifact(
        payload=payload,
        sequential_config=sequential_config,
    )
    ranges = np.asarray(metadata.prior_ranges, dtype=float)
    kde = payload["kde"]

    def _log_density(samples_phi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        samples_phi, inside = _soft_density_inputs(samples_phi, ranges)
        values = np.full(len(samples_phi), -np.inf)
        if not np.any(inside):
            return values
        unit = _to_unit_box(samples_phi[inside], ranges)
        z = _logit(unit)
        log_density_z = kde.logpdf(z.T)
        jac = np.sum(-np.log(ranges[:, 1] - ranges[:, 0]) - np.log(unit) - np.log1p(-unit), axis=1)
        values[inside] = log_density_z + jac
        return values

    _log_density.sample = lambda size, random_state=None: _sample_kde_soft_density(
        kde=kde, ranges=ranges, size=size, random_state=random_state
    )
    return _log_density


def _load_flow_soft_density_model(
    *,
    sequential_config: SequentialInferenceConfig,
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    model_file = sequential_config.soft_density_model_file
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Sequential flow prior support requires PyTorch in the Bayesian environment."
        ) from exc

    payload = _load_artifact_payload(model_file=model_file, model_type="torch_realnvp_2d")
    metadata = _validate_loaded_artifact(
        payload=payload,
        sequential_config=sequential_config,
    )
    model = load_realnvp_2d_from_payload(payload)

    ranges = np.asarray(metadata.prior_ranges, dtype=float)
    standardize = bool(payload.get("standardize", False))
    z_mean = np.asarray(payload.get("z_mean", [0.0, 0.0]), dtype=float)
    z_std = np.asarray(payload.get("z_std", [1.0, 1.0]), dtype=float)

    def _log_density(samples_phi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        samples_phi, inside = _soft_density_inputs(samples_phi, ranges)
        values = np.full(len(samples_phi), -np.inf)
        if not np.any(inside):
            return values
        unit = _to_unit_box(samples_phi[inside], ranges)
        z = _logit(unit)
        if standardize:
            x = (z - z_mean) / z_std
            log_det_standardize = -np.sum(np.log(z_std))
        else:
            x = z
            log_det_standardize = 0.0
        tensor_x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            log_prob_x = model.log_prob(tensor_x).cpu().numpy()
        jac = np.sum(-np.log(ranges[:, 1] - ranges[:, 0]) - np.log(unit) - np.log1p(-unit), axis=1)
        values[inside] = log_prob_x + log_det_standardize + jac
        return values

    _log_density.sample = lambda size, random_state=None: _sample_flow_soft_density(
        model=model,
        ranges=ranges,
        standardize=standardize,
        z_mean=z_mean,
        z_std=z_std,
        n_samples=size,
        random_state=random_state,
    )
    return _log_density


def _validate_loaded_artifact(
    *,
    payload: Any,
    sequential_config: SequentialInferenceConfig,
) -> SoftDensityArtifactMetadata:
    metadata = validate_soft_density_artifact_payload(
        payload=payload,
        model_file=sequential_config.soft_density_model_file,
        configured_model_type=sequential_config.soft_density_model_type,
        expected_columns=list(sequential_config.soft_parameter_names),
    )
    logger.info(
        "Loaded sequential soft-density artifact %s [type=%s, columns=%s, schema=%s, source=%s]",
        metadata.model_file,
        metadata.model_type,
        list(metadata.columns),
        metadata.artifact_schema_version or "unspecified",
        metadata.source_project or "unspecified",
    )
    return metadata


def _soft_density_inputs(samples, ranges):
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    if samples.ndim != 2 or samples.shape[1] != len(ranges):
        raise ValueError(f"Soft density expects shape (n, {len(ranges)}), received {samples.shape}.")
    inside = np.all(np.isfinite(samples) & (samples > ranges[:, 0]) & (samples < ranges[:, 1]), axis=1)
    return samples, inside


def _logit(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    clipped = np.clip(x, 1.0e-10, 1.0 - 1.0e-10)
    return np.log(clipped) - np.log1p(-clipped)


def _to_unit_box(
    samples: npt.NDArray[np.float64],
    bounds: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    unit = (samples - lo) / (hi - lo)
    return np.clip(unit, 1.0e-10, 1.0 - 1.0e-10)


def _inverse_logit(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return expit(x)


def _from_unit_box(
    unit: npt.NDArray[np.float64],
    bounds: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    physical = lo + unit * (hi - lo)
    return np.clip(physical, np.nextafter(lo, hi), np.nextafter(hi, lo))


def _sample_kde_soft_density(
    *,
    kde: Any,
    ranges: npt.NDArray[np.float64],
    size: int,
    random_state=None,
) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(random_state)
    z = np.asarray(kde.resample(size=size, seed=rng).T, dtype=float)
    unit = _inverse_logit(z)
    return _from_unit_box(unit, ranges)


def _sample_flow_soft_density(
    *,
    model: Any,
    ranges: npt.NDArray[np.float64],
    standardize: bool,
    z_mean: npt.NDArray[np.float64],
    z_std: npt.NDArray[np.float64],
    n_samples: int,
    random_state=None,
) -> npt.NDArray[np.float64]:
    import torch

    rng = np.random.default_rng(random_state)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(rng.integers(0, np.iinfo(np.int64).max)))
    with torch.no_grad():
        sampled_x = model.sample(n_samples, generator=generator).cpu().numpy()
    if standardize:
        z = sampled_x * z_std + z_mean
    else:
        z = sampled_x
    unit = _inverse_logit(z)
    return _from_unit_box(unit, ranges)


def _sample_manual_prior(prior_config, lower_bounds, upper_bounds, size, rng):
    manual = prior_config or {}
    kinds = manual.get("type", ["uniform"] * len(lower_bounds))
    samples = np.empty((size, len(lower_bounds)))
    for column, (lower, upper, kind) in enumerate(zip(lower_bounds, upper_bounds, kinds, strict=True)):
        if kind == "uniform":
            draws = rng.uniform(lower, upper, size)
        elif kind == "gaussian":
            mean = manual["mean"][column]
            std = manual["std"][column]
            draws = truncnorm((lower - mean) / std, (upper - mean) / std,
                              loc=mean, scale=std).rvs(size, random_state=rng)
        elif kind == "log":
            if lower <= 0.0:
                raise ValueError("Log-prior sampling requires positive bounds.")
            draws = np.exp(rng.uniform(np.log(lower), np.log(upper), size))
        else:
            raise NotImplementedError(f"Cannot sample prior type '{kind}'.")
        samples[:, column] = draws
    return samples


def _build_sequential_prior_sampler(
    *,
    soft_log_density_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    soft_indices: list[int],
    remaining_indices: list[int],
    sampled_parameter_names: list[str],
    remainder_log_prior_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    remainder_prior_config: dict[str, Any] | None,
    sampled_parameter_min: list[float] | npt.NDArray[np.float64] | None,
    sampled_parameter_max: list[float] | npt.NDArray[np.float64] | None,
) -> Callable[[int], npt.NDArray[np.float64]] | None:
    soft_sampler = getattr(soft_log_density_fn, "sample", None)
    remainder_sampler = getattr(remainder_log_prior_fn, "sample", None) if remaining_indices else None

    if soft_sampler is None:
        return None
    if remaining_indices and remainder_sampler is None and (
        sampled_parameter_min is None or sampled_parameter_max is None
    ):
        return None

    def _sample(size: int, random_state=None) -> npt.NDArray[np.float64]:
        rng = np.random.default_rng(random_state)
        X = np.zeros((size, len(sampled_parameter_names)), dtype=float)
        soft_samples = np.asarray(soft_sampler(size, random_state=rng), dtype=float)
        if soft_samples.ndim == 1:
            soft_samples = soft_samples.reshape(size, 1)
        X[:, soft_indices] = soft_samples
        if remaining_indices:
            if remainder_sampler is not None:
                remainder_samples = np.asarray(remainder_sampler(size=size), dtype=float)
                remainder_samples = np.atleast_2d(remainder_samples)
                if remainder_samples.shape[0] != size and remainder_samples.shape[1] == size:
                    remainder_samples = remainder_samples.T
            else:
                bounds_min = np.asarray(sampled_parameter_min, dtype=float)[remaining_indices]
                bounds_max = np.asarray(sampled_parameter_max, dtype=float)[remaining_indices]
                remainder_samples = _sample_manual_prior(remainder_prior_config, bounds_min, bounds_max, size, rng)
            X[:, remaining_indices] = remainder_samples
        return X

    return _sample
