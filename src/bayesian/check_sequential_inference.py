from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bayesian import prior as prior_module
from bayesian import sequential_prior
from bayesian.mcmc import MCMCConfig
from bayesian.model_discrepancy import parse_discrepancy_group_settings
from bayesian.parameterization import parameterization_info


def run_self_check(
    *,
    config_file: str,
    analysis_name: str | None,
    parameterization: str,
    n_samples: int,
) -> dict[str, Any]:
    config_path = Path(config_file)
    with config_path.open() as stream:
        root_config = yaml.safe_load(stream)

    analyses = root_config["analyses"]
    selected_analysis_name = analysis_name or next(iter(analyses))
    analysis_config = analyses[selected_analysis_name]

    mcmc_config = MCMCConfig(
        analysis_name=selected_analysis_name,
        parameterization=parameterization,
        analysis_config=analysis_config,
        config_file=str(config_path),
    )

    parameter_info = parameterization_info(analysis_config, parameterization)
    sampled_names = parameter_info.sampled_names
    sampled_min = parameter_info.sampled_min
    sampled_max = parameter_info.sampled_max

    model_prior_config = analysis_config["parameterization"][parameterization].get("prior", None)
    model_prior_config = parameter_info.filter_prior_config(model_prior_config)
    names, parameter_min, parameter_max, combined_prior_config, _, _ = parse_discrepancy_group_settings(
        analysis_config["parameters"]["emulators"],
        sampled_names,
        sampled_min,
        sampled_max,
        model_prior_config,
    )

    base_log_prior_fn = prior_module.make_log_prior_fn(combined_prior_config, names)
    sequential_log_prior_fn = sequential_prior.build_sequential_log_prior_fn(
        sequential_config=mcmc_config.sequential_inference_config,
        combined_prior_config=combined_prior_config,
        sampled_parameter_names=names,
        sampled_parameter_min=parameter_min,
        sampled_parameter_max=parameter_max,
    )
    if sequential_log_prior_fn is None:
        raise ValueError("Sequential inference is not enabled in the provided config.")

    midpoint = np.asarray(parameter_min, dtype=float) + 0.5 * (
        np.asarray(parameter_max, dtype=float) - np.asarray(parameter_min, dtype=float)
    )
    sequential_log_prior_mid = float(sequential_log_prior_fn(midpoint.reshape(1, -1))[0])
    base_log_prior_mid = float(base_log_prior_fn(midpoint.reshape(1, -1))[0])

    sampled_points = None
    sampled_points_shape = None
    sampled_points_summary = None
    if hasattr(sequential_log_prior_fn, "sample"):
        sampled_points = np.asarray(sequential_log_prior_fn.sample(n_samples), dtype=float)
        sampled_points_shape = list(sampled_points.shape)
        sampled_points_summary = {
            "min": sampled_points.min(axis=0).tolist(),
            "max": sampled_points.max(axis=0).tolist(),
            "mean": sampled_points.mean(axis=0).tolist(),
        }

    sequential_config = mcmc_config.sequential_inference_config
    artifact_metadata = sequential_config.artifact_metadata if sequential_config is not None else None

    result = {
        "config_file": str(config_path.resolve()),
        "analysis_name": selected_analysis_name,
        "parameterization": parameterization,
        "sampled_parameter_names": list(names),
        "sampled_parameter_bounds": {
            name: [float(lower), float(upper)]
            for name, lower, upper in zip(names, parameter_min, parameter_max, strict=True)
        },
        "soft_parameter_names": list(sequential_config.soft_parameter_names),
        "alpha_parameter_name": sequential_config.alpha_parameter_name,
        "soft_density_model_file": str(sequential_config.soft_density_model_file),
        "soft_density_model_type": sequential_config.soft_density_model_type,
        "artifact_metadata": artifact_metadata.to_dict() if artifact_metadata is not None else None,
        "checks": {
            "sequential_prior_has_sampler": bool(hasattr(sequential_log_prior_fn, "sample")),
            "sample_shape": sampled_points_shape,
            "base_log_prior_midpoint": base_log_prior_mid,
            "sequential_log_prior_midpoint": sequential_log_prior_mid,
            "sample_summary": sampled_points_summary,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Bayesian sequential-inference configuration and soft-density artifacts.")
    parser.add_argument("-c", "--config", required=True, help="Path to the Bayesian YAML config file.")
    parser.add_argument("-a", "--analysis", default=None, help="Analysis name under 'analyses'. Defaults to the first analysis in the config.")
    parser.add_argument("-p", "--parameterization", default="exponential", help="Parameterization name to validate.")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of prior samples to draw when the sequential prior supports sampling.")
    args = parser.parse_args()

    result = run_self_check(
        config_file=args.config,
        analysis_name=args.analysis,
        parameterization=args.parameterization,
        n_samples=args.n_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
