from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from bayesian import prior as prior_module

logger = logging.getLogger(__name__)


def build_sequential_log_prior_fn(
    *,
    sequential_config: dict[str, Any] | None,
    combined_prior_config: dict[str, Any] | None,
    sampled_parameter_names: list[str],
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None:
    if not sequential_config or not sequential_config.get("enabled", False):
        return None

    soft_parameter_names = list(sequential_config.get("soft_parameter_names", ["nucleon_width", "normalization"]))
    model_file = Path(sequential_config["soft_density_model_file"])
    model_type = sequential_config.get("soft_density_model_type", "gaussian_kde")

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
    soft_log_density_fn = _load_soft_density_model(
        model_file=model_file,
        model_type=model_type,
        soft_parameter_names=soft_parameter_names,
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


def _load_soft_density_model(
    *,
    model_file: Path,
    model_type: str,
    soft_parameter_names: list[str],
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    if model_type != "gaussian_kde":
        raise NotImplementedError(
            f"Sequential inference currently supports only gaussian_kde soft models, not '{model_type}'."
        )

    with model_file.open("rb") as stream:
        payload = pickle.load(stream)

    payload_model_type = payload.get("model_type", "gaussian_kde")
    if payload_model_type != "gaussian_kde":
        raise NotImplementedError(
            f"Soft density artifact {model_file} advertises unsupported model_type '{payload_model_type}'."
        )

    payload_columns = list(payload.get("columns", soft_parameter_names))
    if payload_columns != soft_parameter_names:
        raise ValueError(
            f"Soft density model columns {payload_columns} do not match expected sequential soft parameters {soft_parameter_names}."
        )

    ranges = np.asarray(payload["prior_ranges"], dtype=float)
    kde = payload["kde"]

    def _log_density(samples_phi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        samples_phi = np.array(samples_phi, dtype=float, ndmin=2)
        unit = _to_unit_box(samples_phi, ranges)
        z = _logit(unit)
        log_density_z = kde.logpdf(z.T)
        jac = np.sum(-np.log(ranges[:, 1] - ranges[:, 0]) - np.log(unit) - np.log1p(-unit), axis=1)
        return np.asarray(log_density_z + jac, dtype=float)

    return _log_density


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
