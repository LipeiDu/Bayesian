from __future__ import annotations

from dataclasses import dataclass, field
import pickle
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_SOFT_DENSITY_MODEL_TYPES = {"gaussian_kde", "torch_realnvp_2d"}


@dataclass(frozen=True)
class SoftDensityArtifactMetadata:
    model_file: Path
    model_type: str
    columns: tuple[str, ...]
    prior_ranges: tuple[tuple[float, float], ...]
    transform: str | None = None
    artifact_schema_version: str | None = None
    source_project: str | None = None
    created_utc: str | None = None
    payload_keys: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "model_file": str(self.model_file),
            "model_type": self.model_type,
            "columns": list(self.columns),
            "prior_ranges": [list(bounds) for bounds in self.prior_ranges],
            "payload_keys": list(self.payload_keys),
        }
        if self.transform is not None:
            result["transform"] = self.transform
        if self.artifact_schema_version is not None:
            result["artifact_schema_version"] = self.artifact_schema_version
        if self.source_project is not None:
            result["source_project"] = self.source_project
        if self.created_utc is not None:
            result["created_utc"] = self.created_utc
        return result


@dataclass(frozen=True)
class SequentialInferenceConfig:
    soft_density_model_file: Path
    soft_density_model_type: str
    soft_parameter_names: tuple[str, ...]
    alpha_parameter_name: str
    artifact_metadata: SoftDensityArtifactMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "enabled": True,
            "soft_density_model_file": str(self.soft_density_model_file),
            "soft_density_model_type": self.soft_density_model_type,
            "soft_parameter_names": list(self.soft_parameter_names),
            "alpha_parameter_name": self.alpha_parameter_name,
        }
        if self.artifact_metadata is not None:
            result["artifact_metadata"] = self.artifact_metadata.to_dict()
        return result


def resolve_sequential_inference_config(
    *,
    mcmc_configuration: dict[str, Any],
    config_file: Path,
    analysis_config: dict[str, Any],
    parameterization: str,
) -> SequentialInferenceConfig | None:
    sequential_config = dict(mcmc_configuration.get("sequential_inference", {}) or {})
    if not sequential_config.get("enabled", False):
        return None

    if "soft_density_model_file" not in sequential_config:
        raise ValueError(
            "Sequential inference is enabled, but no 'soft_density_model_file' was provided in the MCMC config."
        )

    model_path = Path(sequential_config["soft_density_model_file"])
    if not model_path.is_absolute():
        model_path = (config_file.parent / model_path).resolve()

    model_type = sequential_config.get("soft_density_model_type")
    if model_type is None:
        model_type = _infer_soft_density_model_type(model_path)
    if model_type not in SUPPORTED_SOFT_DENSITY_MODEL_TYPES:
        raise NotImplementedError(
            "Sequential inference currently supports only gaussian_kde and torch_realnvp_2d soft models, "
            f"not '{model_type}'."
        )

    soft_parameter_names = tuple(
        sequential_config.get("soft_parameter_names", ["nucleon_width", "normalization"])
    )
    alpha_parameter_name = sequential_config.get(
        "alpha_parameter_name",
        analysis_config["parameterization"][parameterization].get("qhat_alpha_s_parameter", "AlphaS"),
    )

    base_config = SequentialInferenceConfig(
        soft_density_model_file=model_path,
        soft_density_model_type=model_type,
        soft_parameter_names=soft_parameter_names,
        alpha_parameter_name=alpha_parameter_name,
        artifact_metadata=None,
    )
    artifact_metadata = load_soft_density_artifact_metadata(base_config)
    expected_soft_bounds = _resolve_expected_soft_parameter_bounds(
        analysis_config=analysis_config,
        parameterization=parameterization,
        soft_parameter_names=soft_parameter_names,
    )
    _validate_soft_parameter_bounds_match(
        model_path=model_path,
        artifact_metadata=artifact_metadata,
        expected_soft_bounds=expected_soft_bounds,
    )
    return SequentialInferenceConfig(
        soft_density_model_file=base_config.soft_density_model_file,
        soft_density_model_type=base_config.soft_density_model_type,
        soft_parameter_names=base_config.soft_parameter_names,
        alpha_parameter_name=base_config.alpha_parameter_name,
        artifact_metadata=artifact_metadata,
    )


def load_soft_density_artifact_metadata(
    sequential_config: SequentialInferenceConfig,
) -> SoftDensityArtifactMetadata:
    payload = _load_artifact_payload(
        model_file=sequential_config.soft_density_model_file,
        model_type=sequential_config.soft_density_model_type,
    )
    return validate_soft_density_artifact_payload(
        payload=payload,
        model_file=sequential_config.soft_density_model_file,
        configured_model_type=sequential_config.soft_density_model_type,
        expected_columns=list(sequential_config.soft_parameter_names),
    )


def validate_soft_density_artifact_payload(
    *,
    payload: Any,
    model_file: Path,
    configured_model_type: str,
    expected_columns: list[str],
) -> SoftDensityArtifactMetadata:
    if not isinstance(payload, dict):
        raise TypeError(
            f"Soft density artifact {model_file} must deserialize to a dict, not {type(payload).__name__}."
        )

    payload_model_type = payload.get("model_type", configured_model_type)
    if payload_model_type != configured_model_type:
        raise ValueError(
            f"Soft density artifact {model_file} advertises model_type '{payload_model_type}', "
            f"but the config expects '{configured_model_type}'."
        )
    if payload_model_type not in SUPPORTED_SOFT_DENSITY_MODEL_TYPES:
        raise NotImplementedError(
            f"Soft density artifact {model_file} advertises unsupported model_type '{payload_model_type}'."
        )

    payload_columns = list(payload.get("columns", expected_columns))
    if payload_columns != expected_columns:
        raise ValueError(
            f"Soft density model columns {payload_columns} do not match expected sequential soft parameters {expected_columns}."
        )

    required_keys = {"columns", "prior_ranges", "transform"}
    if payload_model_type == "gaussian_kde":
        required_keys.update({"kde"})
    elif payload_model_type == "torch_realnvp_2d":
        required_keys.update({"architecture", "state_dict"})
    missing_keys = sorted(key for key in required_keys if key not in payload)
    if missing_keys:
        raise KeyError(
            f"Soft density artifact {model_file} is missing required keys for model_type '{payload_model_type}': {missing_keys}"
        )

    if "prior_ranges" not in payload:
        raise KeyError(f"Soft density artifact {model_file} is missing required key 'prior_ranges'.")
    ranges = np.asarray(payload["prior_ranges"], dtype=float)
    if ranges.shape != (len(expected_columns), 2):
        raise ValueError(
            f"Soft density artifact {model_file} has prior_ranges shape {ranges.shape}, "
            f"but expected ({len(expected_columns)}, 2)."
        )
    if not np.all(np.isfinite(ranges)):
        raise ValueError(f"Soft density artifact {model_file} contains non-finite prior_ranges values.")
    if np.any(ranges[:, 1] <= ranges[:, 0]):
        raise ValueError(f"Soft density artifact {model_file} has invalid prior_ranges with upper <= lower.")

    return SoftDensityArtifactMetadata(
        model_file=model_file,
        model_type=payload_model_type,
        columns=tuple(payload_columns),
        prior_ranges=tuple((float(lo), float(hi)) for lo, hi in ranges),
        transform=str(payload.get("transform")) if payload.get("transform") is not None else None,
        artifact_schema_version=payload.get("artifact_schema_version"),
        source_project=payload.get("source_project"),
        created_utc=payload.get("created_utc"),
        payload_keys=tuple(sorted(str(key) for key in payload.keys())),
    )


def _load_artifact_payload(*, model_file: Path, model_type: str) -> Any:
    if not model_file.exists():
        raise FileNotFoundError(f"Soft density artifact {model_file} does not exist.")

    if model_type == "gaussian_kde":
        with model_file.open("rb") as stream:
            return pickle.load(stream)

    if model_type == "torch_realnvp_2d":
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Sequential flow prior support requires PyTorch in the Bayesian environment."
            ) from exc
        return torch.load(model_file, map_location="cpu")

    raise NotImplementedError(
        "Sequential inference currently supports only gaussian_kde and torch_realnvp_2d soft models, "
        f"not '{model_type}'."
    )


def _infer_soft_density_model_type(model_file: Path) -> str:
    suffix = model_file.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return "gaussian_kde"
    if suffix in {".pt", ".pth"}:
        return "torch_realnvp_2d"
    raise ValueError(
        "Sequential inference config did not specify 'soft_density_model_type', and the file extension "
        f"for {model_file} is not recognized. Please set the model type explicitly."
    )


def _resolve_expected_soft_parameter_bounds(
    *,
    analysis_config: dict[str, Any],
    parameterization: str,
    soft_parameter_names: tuple[str, ...],
) -> tuple[tuple[float, float], ...]:
    parameterization_config = analysis_config["parameterization"][parameterization]
    names = list(parameterization_config["names"])
    mins = list(parameterization_config["min"])
    maxs = list(parameterization_config["max"])
    bounds_by_name = {
        name: (float(lower), float(upper))
        for name, lower, upper in zip(names, mins, maxs, strict=True)
    }
    return tuple(bounds_by_name[name] for name in soft_parameter_names)


def _validate_soft_parameter_bounds_match(
    *,
    model_path: Path,
    artifact_metadata: SoftDensityArtifactMetadata,
    expected_soft_bounds: tuple[tuple[float, float], ...],
) -> None:
    artifact_bounds = np.asarray(artifact_metadata.prior_ranges, dtype=float)
    configured_bounds = np.asarray(expected_soft_bounds, dtype=float)
    if artifact_bounds.shape != configured_bounds.shape or not np.allclose(artifact_bounds, configured_bounds):
        raise ValueError(
            f"Soft density artifact {model_path} uses prior_ranges {artifact_metadata.prior_ranges}, "
            f"which do not match the Bayesian config soft-parameter bounds {expected_soft_bounds}."
        )
