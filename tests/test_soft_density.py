"""Check real soft-density payloads, support, repeated loading, and prior draws."""

import copy

import numpy as np
import pytest
from scipy.stats import truncnorm

from bayesian import sequential_prior
from bayesian.sequential_inference import validate_soft_density_artifact_payload
from tests.sequential_examples import BOUNDS, artifact_config, artifact_payload, reference_soft_density


@pytest.mark.parametrize("model_type", ["gaussian_kde", "torch_realnvp_2d"])
def test_versioned_density_matches_direct_payload_evaluation(model_type) -> None:
    density = sequential_prior.load_soft_density_model(sequential_config=artifact_config(model_type))
    points = np.array([[0.7, 12.0], [1.0, 14.5], [1.2, 17.0]])
    np.testing.assert_allclose(density(points), reference_soft_density(points, model_type), atol=2e-5, rtol=2e-6)
    again = sequential_prior.load_soft_density_model(sequential_config=artifact_config(model_type))
    np.testing.assert_array_equal(density(points), again(points))


@pytest.mark.parametrize("model_type", ["gaussian_kde", "torch_realnvp_2d"])
def test_soft_density_support_and_shape(model_type) -> None:
    density = sequential_prior.load_soft_density_model(sequential_config=artifact_config(model_type))
    points = [
        [0.4, 15.0],
        [1.6, 15.0],
        [1.0, 9.0],
        [1.0, 21.0],
        [0.5, 15.0],
        [1.5, 15.0],
        [1.0, 10.0],
        [1.0, 20.0],
        [np.nan, 15.0],
        [np.inf, 15.0],
    ]
    assert np.all(np.isneginf(density(points)))
    assert np.isfinite(density([[1.0, 15.0]])[0])
    assert density(np.empty((0, 2))).shape == (0,)
    with pytest.raises(ValueError, match="shape"):
        density([[1.0, 15.0, 0.3]])


@pytest.mark.parametrize("model_type", ["gaussian_kde", "torch_realnvp_2d"])
def test_seeded_soft_draws_stay_inside_bounds(model_type) -> None:
    density = sequential_prior.load_soft_density_model(sequential_config=artifact_config(model_type))
    first = density.sample(100, random_state=314159)
    second = density.sample(100, random_state=314159)
    np.testing.assert_array_equal(first, second)
    assert np.all(first > BOUNDS[:, 0])
    assert np.all(first < BOUNDS[:, 1])
    assert np.all(np.isfinite(density(first)))


@pytest.mark.parametrize("model_type", ["gaussian_kde", "torch_realnvp_2d"])
def test_sequential_prior_uses_parameter_names_and_samples_gaussian_remainder(model_type) -> None:
    names = ["normalization", "AlphaS", "nucleon_width"]
    prior = sequential_prior.build_sequential_log_prior_fn(
        sequential_config=artifact_config(model_type),
        sampled_parameter_names=names,
        sampled_parameter_min=[10.0, 0.1, 0.5],
        sampled_parameter_max=[20.0, 0.5, 1.5],
        combined_prior_config={
            "type": ["uniform", "gaussian", "uniform"],
            "mean": [None, 0.2, None],
            "std": [None, 0.02, None],
        },
    )
    points = np.array([[15.0, 0.2, 1.0], [14.0, 0.23, 0.9]])
    expected = reference_soft_density(points[:, [2, 0]], model_type)
    expected += -0.5 * ((points[:, 1] - 0.2) / 0.02) ** 2 - np.log(0.02 * np.sqrt(2 * np.pi))
    np.testing.assert_allclose(prior(points), expected, atol=2e-5, rtol=2e-6)
    draws = prior.sample(2000, random_state=314159)
    np.testing.assert_array_equal(draws, prior.sample(2000, random_state=314159))
    assert np.all(draws > [10.0, 0.1, 0.5])
    assert np.all(draws < [20.0, 0.5, 1.5])
    distribution = truncnorm(-5.0, 15.0, loc=0.2, scale=0.02)
    assert np.mean(draws[:, 1]) == pytest.approx(distribution.mean(), abs=0.002)
    assert np.std(draws[:, 1]) == pytest.approx(distribution.std(), abs=0.002)


def test_inverse_logit_handles_extreme_latent_draws() -> None:
    unit = sequential_prior._inverse_logit(np.array([[-1000.0, 1000.0]]))
    physical = sequential_prior._from_unit_box(unit, BOUNDS)
    assert np.all(np.isfinite(physical))
    assert np.all(physical > BOUNDS[:, 0])
    assert np.all(physical < BOUNDS[:, 1])


def test_sequential_prior_draws_log_uniform_remainder() -> None:
    prior = sequential_prior.build_sequential_log_prior_fn(
        sequential_config=artifact_config("gaussian_kde"),
        sampled_parameter_names=["AlphaS", "nucleon_width", "normalization"],
        sampled_parameter_min=[0.1, 0.5, 10.0],
        sampled_parameter_max=[0.5, 1.5, 20.0],
        combined_prior_config={"type": ["log", "uniform", "uniform"]},
    )
    draws = prior.sample(3000, random_state=314159)
    expected_mean = 0.5 * (np.log(0.1) + np.log(0.5))
    assert np.mean(np.log(draws[:, 0])) == pytest.approx(expected_mean, abs=0.02)
    reference = reference_soft_density(draws[:3, 1:], "gaussian_kde") - np.log(draws[:3, 0])
    np.testing.assert_allclose(prior(draws[:3]), reference)


def test_artifact_validation_rejects_unsupported_transform() -> None:
    payload = dict(artifact_payload("gaussian_kde"), transform="unrecognized")
    with pytest.raises(ValueError, match="transform"):
        validate_soft_density_artifact_payload(
            payload=payload,
            model_file=artifact_config("gaussian_kde").soft_density_model_file,
            configured_model_type="gaussian_kde",
            expected_columns=["nucleon_width", "normalization"],
        )


@pytest.mark.parametrize("standard_deviation", [[0.0, 1.0], [np.nan, 1.0], [1.0]])
def test_artifact_validation_rejects_invalid_standardization(standard_deviation) -> None:
    payload = copy.copy(artifact_payload("torch_realnvp_2d"))
    payload.update(standardize=True, z_std=standard_deviation)
    with pytest.raises(ValueError, match="standardization"):
        validate_soft_density_artifact_payload(
            payload=payload,
            model_file=artifact_config("torch_realnvp_2d").soft_density_model_file,
            configured_model_type="torch_realnvp_2d",
            expected_columns=["nucleon_width", "normalization"],
        )
