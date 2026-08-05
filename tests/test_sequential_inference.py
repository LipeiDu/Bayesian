"""Focused tests for the sequential-inference interface."""

from pathlib import Path
import pickle

import numpy as np
import pytest

from bayesian import sequential_prior
from bayesian.posterior_utils import flatten_chain_samples
from bayesian.sequential_inference import resolve_sequential_inference_config


def _analysis_config() -> dict:
    return {
        "parameterization": {
            "exponential": {
                "names": ["AlphaS", "nucleon_width", "normalization"],
                "min": [0.1, 0.5, 10.0],
                "max": [0.5, 1.5, 20.0],
                "qhat_alpha_s_parameter": "AlphaS",
            }
        }
    }


def _write_kde_artifact(path: Path, *, prior_ranges: list[list[float]]) -> None:
    payload = {
        "model_type": "gaussian_kde",
        "columns": ["nucleon_width", "normalization"],
        "prior_ranges": prior_ranges,
        "transform": "bounded_logit",
        "kde": "test-placeholder",
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream)


def test_resolve_sequential_config_validates_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "soft_density.pkl"
    _write_kde_artifact(artifact_path, prior_ranges=[[0.5, 1.5], [10.0, 20.0]])

    config = resolve_sequential_inference_config(
        mcmc_configuration={
            "sequential_inference": {
                "enabled": True,
                "soft_density_model_file": str(artifact_path),
                "soft_parameter_names": ["nucleon_width", "normalization"],
                "alpha_parameter_name": "AlphaS",
            }
        },
        config_file=tmp_path / "analysis.yaml",
        analysis_config=_analysis_config(),
        parameterization="exponential",
    )

    assert config is not None
    assert config.soft_density_model_type == "gaussian_kde"
    assert config.soft_parameter_names == ("nucleon_width", "normalization")
    assert config.artifact_metadata is not None
    assert config.artifact_metadata.prior_ranges == ((0.5, 1.5), (10.0, 20.0))


def test_resolve_sequential_config_rejects_bound_mismatch(tmp_path: Path) -> None:
    artifact_path = tmp_path / "soft_density.pkl"
    _write_kde_artifact(artifact_path, prior_ranges=[[0.4, 1.6], [10.0, 20.0]])

    with pytest.raises(ValueError, match="bounds"):
        resolve_sequential_inference_config(
            mcmc_configuration={
                "sequential_inference": {
                    "enabled": True,
                    "soft_density_model_file": str(artifact_path),
                    "soft_parameter_names": ["nucleon_width", "normalization"],
                }
            },
            config_file=tmp_path / "analysis.yaml",
            analysis_config=_analysis_config(),
            parameterization="exponential",
        )


def test_subset_manual_prior_config_expands_short_fields() -> None:
    subset = sequential_prior._subset_manual_prior_config(
        prior_config={"type": ["uniform"], "mean": [], "std": []},
        sampled_parameter_names=["AlphaS", "nucleon_width", "normalization"],
        target_names=["AlphaS"],
    )

    assert subset == {"type": ["uniform"], "mean": [None], "std": [None]}


def test_flatten_chain_samples_accepts_emcee_and_pocomc() -> None:
    emcee_chain = np.arange(24, dtype=float).reshape(4, 2, 3)
    pocomc_samples = np.arange(15, dtype=float).reshape(5, 3)

    np.testing.assert_array_equal(flatten_chain_samples(emcee_chain), emcee_chain.reshape(8, 3))
    np.testing.assert_array_equal(flatten_chain_samples(pocomc_samples), pocomc_samples)


def test_flatten_chain_samples_rejects_other_shapes() -> None:
    with pytest.raises(ValueError, match="Unsupported chain shape"):
        flatten_chain_samples(np.zeros((2, 2, 2, 2)))
