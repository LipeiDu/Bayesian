from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from bayesian import prior as prior_module
from bayesian.parameterization import ParameterizationInfo

logger = logging.getLogger(__name__)


def export_hard_likelihood_samples(
    *,
    config,
    parameter_info: ParameterizationInfo,
    sampled_parameter_names: list[str],
    posterior_samples: npt.NDArray[np.float64],
    log_posterior_values: npt.NDArray[np.float64],
    model_prior_config: dict[str, Any] | None,
    combined_prior_config: dict[str, Any] | None,
    sample_weights: npt.NDArray[np.float64] | None = None,
) -> None:
    if not getattr(config, "export_hard_likelihood_samples", False):
        return
    if getattr(config, "sequential_inference_config", None):
        logger.info(
            "Skipping hard-likelihood export because sequential inference mode is enabled for this run."
        )
        return

    if posterior_samples.size == 0:
        logger.warning("Skipping hard-likelihood export because no posterior samples were provided.")
        return

    output_csv = config.mcmc_output_dir / config.hard_likelihood_outputfilename
    output_json = output_csv.with_suffix(".json")

    sampled_parameter_names = list(sampled_parameter_names)
    posterior_samples = np.asarray(posterior_samples, dtype=float)
    log_posterior_values = np.asarray(log_posterior_values, dtype=float).reshape(-1)
    if posterior_samples.shape[0] != log_posterior_values.shape[0]:
        raise ValueError(
            "Posterior sample count does not match log-posterior count: "
            f"{posterior_samples.shape[0]} vs {log_posterior_values.shape[0]}"
        )

    total_log_prior = _evaluate_total_log_prior(
        posterior_samples=posterior_samples,
        combined_prior_config=combined_prior_config,
        sampled_parameter_names=sampled_parameter_names,
    )
    log_likelihood_hard = log_posterior_values - total_log_prior

    model_sampled_names = [name for name in sampled_parameter_names if name in parameter_info.sampled_names]
    model_sampled_indices = [sampled_parameter_names.index(name) for name in model_sampled_names]
    model_sampled_points = posterior_samples[:, model_sampled_indices]
    full_model_points = parameter_info.expand_sampled_to_full(
        model_sampled_points,
        sampled_names=model_sampled_names,
    )

    alpha_name = _resolve_alpha_name(config, parameter_info)
    soft_names = _resolve_soft_parameter_names(config, parameter_info)
    required_names = [alpha_name, *soft_names]
    missing_names = [name for name in required_names if name not in parameter_info.full_names]
    if missing_names:
        raise ValueError(
            "Cannot export sequential hard-likelihood samples because required model parameters "
            f"are missing from the parameterization: {missing_names}"
        )

    name_to_full_index = {name: i for i, name in enumerate(parameter_info.full_names)}
    alpha_values = full_model_points[:, name_to_full_index[alpha_name]]
    nucleon_width_values = full_model_points[:, name_to_full_index["nucleon_width"]]
    normalization_values = full_model_points[:, name_to_full_index["normalization"]]

    log_prior_soft_base = _evaluate_component_log_prior(
        full_model_points=full_model_points,
        full_model_names=parameter_info.full_names,
        model_prior_config=model_prior_config,
        target_names=soft_names,
    )
    log_prior_alpha = _evaluate_component_log_prior(
        full_model_points=full_model_points,
        full_model_names=parameter_info.full_names,
        model_prior_config=model_prior_config,
        target_names=[alpha_name],
    )

    fieldnames = [
        "sample_id",
        alpha_name,
        *soft_names,
        "log_likelihood_hard",
        "log_prior_soft_base",
        "log_prior_alpha",
        "log_posterior_practical",
        "proposal_source",
        "run_id",
        "config_path",
    ]
    if sample_weights is not None:
        fieldnames.append("sample_weight")

    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(posterior_samples.shape[0]):
            row = {
                "sample_id": str(i),
                alpha_name: _format_float(alpha_values[i]),
                soft_names[0]: _format_float(nucleon_width_values[i]),
                soft_names[1]: _format_float(normalization_values[i]),
                "log_likelihood_hard": _format_float(log_likelihood_hard[i]),
                "log_prior_soft_base": _format_float(log_prior_soft_base[i]),
                "log_prior_alpha": _format_float(log_prior_alpha[i]),
                "log_posterior_practical": _format_float(log_posterior_values[i]),
                "proposal_source": config.mcmc_package,
                "run_id": config.mcmc_output_dir.name,
                "config_path": str(config.config_file.resolve()),
            }
            if sample_weights is not None:
                row["sample_weight"] = _format_float(sample_weights[i])
            writer.writerow(row)

    metadata = {
        "run_id": config.mcmc_output_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_name": config.analysis_name,
        "parameterization": config.parameterization,
        "config_path": str(config.config_file.resolve()),
        "proposal_source": config.mcmc_package,
        "sample_count": int(posterior_samples.shape[0]),
        "parameter_names": [alpha_name, *soft_names],
        "source_sampled_parameter_names": sampled_parameter_names,
        "prior_ranges": _build_prior_ranges(parameter_info, alpha_name=alpha_name, soft_names=soft_names),
        "notes": (
            "Rows are sample-based outputs from the Bayesian reduced practical run. "
            "log_likelihood_hard equals log_posterior_practical minus the total prior used during sampling."
        ),
        "artifacts": {
            "hard_likelihood_samples_csv": str(output_csv),
            "mcmc_h5": str(config.mcmc_output_dir / "mcmc.h5"),
            "posterior_h5": str(config.mcmc_output_dir / "posterior.h5"),
        },
    }
    if sample_weights is not None:
        metadata["artifacts"]["sample_weights_in_csv"] = True

    output_json.write_text(json.dumps(metadata, indent=2))
    logger.info("Exported hard-likelihood samples to %s", output_csv)
    logger.info("Exported hard-likelihood metadata to %s", output_json)


def _resolve_alpha_name(config, parameter_info: ParameterizationInfo) -> str:
    configured_name = config.analysis_config["parameterization"][config.parameterization].get(
        "qhat_alpha_s_parameter",
        None,
    )
    if configured_name is not None:
        return configured_name

    fallback_candidates = ["alpha_s", "AlphaS", "alphaS"]
    for name in fallback_candidates:
        if name in parameter_info.full_names:
            return name

    raise ValueError("Could not determine the alpha_s parameter name for hard-likelihood export.")


def _resolve_soft_parameter_names(config, parameter_info: ParameterizationInfo) -> list[str]:
    sequential_config = getattr(config, "sequential_inference_config", None) or {}
    soft_names = list(sequential_config.get("soft_parameter_names", ["nucleon_width", "normalization"]))
    missing = [name for name in soft_names if name not in parameter_info.full_names]
    if missing:
        raise ValueError(
            f"Configured soft parameter names {soft_names} are not all present in the parameterization. Missing: {missing}"
        )
    if len(soft_names) != 2:
        raise ValueError(
            "Hard-likelihood export currently expects exactly two soft parameters in the reduced study. "
            f"Received: {soft_names}"
        )
    return soft_names


def _build_prior_ranges(
    parameter_info: ParameterizationInfo,
    *,
    alpha_name: str,
    soft_names: list[str],
) -> dict[str, list[float]]:
    name_to_bounds = {
        name: [float(lower), float(upper)]
        for name, lower, upper in zip(
            parameter_info.full_names,
            parameter_info.full_min,
            parameter_info.full_max,
            strict=True,
        )
    }
    return {
        alpha_name: name_to_bounds[alpha_name],
        soft_names[0]: name_to_bounds[soft_names[0]],
        soft_names[1]: name_to_bounds[soft_names[1]],
    }


def _subset_manual_prior_config(
    prior_config: dict[str, Any] | None,
    full_model_names: list[str],
    target_names: list[str],
) -> dict[str, Any] | None:
    if prior_config is None:
        return None
    if "prior_source" in prior_config:
        msg = (
            "Sequential hard-likelihood export does not currently support extracting "
            "a soft-only prior term from a correlated prior_source configuration."
        )
        raise ValueError(msg)

    selected_indices = [full_model_names.index(name) for name in target_names]
    full_type = _expand_prior_field(
        values=prior_config.get("type", []),
        expected_length=len(full_model_names),
        default="uniform",
    )
    full_mean = _expand_prior_field(
        values=prior_config.get("mean", []),
        expected_length=len(full_model_names),
        default=None,
    )
    full_std = _expand_prior_field(
        values=prior_config.get("std", []),
        expected_length=len(full_model_names),
        default=None,
    )
    subset = {}
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


def _evaluate_component_log_prior(
    *,
    full_model_points: npt.NDArray[np.float64],
    full_model_names: list[str],
    model_prior_config: dict[str, Any] | None,
    target_names: list[str],
) -> npt.NDArray[np.float64]:
    subset_prior_config = _subset_manual_prior_config(model_prior_config, full_model_names, target_names)
    target_indices = [full_model_names.index(name) for name in target_names]
    target_points = full_model_points[:, target_indices]
    log_prior_fn = prior_module.make_log_prior_fn(subset_prior_config, target_names)
    return np.asarray(log_prior_fn(target_points), dtype=float)


def _evaluate_total_log_prior(
    *,
    posterior_samples: npt.NDArray[np.float64],
    combined_prior_config: dict[str, Any] | None,
    sampled_parameter_names: list[str],
) -> npt.NDArray[np.float64]:
    log_prior_fn = prior_module.make_log_prior_fn(combined_prior_config, sampled_parameter_names)
    return np.asarray(log_prior_fn(posterior_samples), dtype=float)


def _format_float(value: float) -> str:
    return f"{float(value):.16e}"
