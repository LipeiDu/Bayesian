"""Weighted summaries, named model projection, and shortest contiguous intervals."""

import h5py
import numpy as np
import pytest

from bayesian import mcmc
from bayesian.parameterization import ParameterizationInfo
from bayesian.posterior_utils import (
    model_posterior_samples,
    normalized_weights,
    select_posterior_samples,
    weighted_quantile,
)
from bayesian.prior import sample_prior


def test_weighted_quantiles_ignore_zero_mass_and_duplicate_representation():
    baseline = weighted_quantile([0, 10], [0, 0.16, 0.5, 0.84, 1], [0.9, 0.1])
    split = weighted_quantile([-100, 0, 10, 0, 100], [0, 0.16, 0.5, 0.84, 1], [0, 0.4, 0.1, 0.5, 0])
    np.testing.assert_allclose(split, baseline)
    np.testing.assert_allclose(baseline, [0, 0, 1, 7.8, 10])


@pytest.mark.parametrize("bad", [[0, 0], [1, -1], [1, np.nan], [1, np.inf], [[1, 2]]])
def test_weights_reject_invalid_input(bad):
    with pytest.raises(ValueError, match="weights"):
        normalized_weights(bad, 2)


def test_weights_are_stable_and_do_not_mutate_input():
    weights = np.array([1e308, 1e308])
    before = weights.copy()
    np.testing.assert_allclose(normalized_weights(weights, 2), [0.5, 0.5])
    np.testing.assert_array_equal(weights, before)


@pytest.mark.parametrize(("values", "probs"), [([1, np.nan], [0.5]), ([1, 2], [np.nan]), ([1, 2], [-0.1])])
def test_quantiles_reject_nonfinite_values_and_invalid_probabilities(values, probs):
    with pytest.raises(ValueError, match=r"finite|probabilities"):
        weighted_quantile(values, probs, [1, 1])


def test_intervals_respect_particle_weights_and_high_confidence():
    np.testing.assert_allclose(mcmc.credible_interval([0, 10], confidence=0.8, weights=[0.9, 0.1]), [0, 9])
    np.testing.assert_allclose(mcmc.credible_interval([0, 10], confidence=0.9, interval_type="hpd", weights=[0.95, 0.05]), [0, 0])
    np.testing.assert_allclose(mcmc.credible_interval(np.arange(5), confidence=0.99, interval_type="hpd"), [0, 4])


@pytest.mark.parametrize("confidence", [0, 1.1, np.nan])
def test_intervals_reject_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        mcmc.credible_interval([0, 1], confidence=confidence)


def test_legacy_map_summary_is_finite_for_small_chains():
    np.testing.assert_allclose(mcmc.map_parameters(np.array([[0, 10], [10, 20]])), [5, 15])


@pytest.mark.parametrize("flat", [True, False])
def test_model_projection_preserves_weights_and_fixed_values(flat):
    samples = np.array([[99, 0.8, 0.2], [98, 1.0, 0.4]])
    results = {"chain": samples if flat else samples.reshape(1, 2, 3),
               "parameter_names": np.array(["g__noise", "w", "a"], dtype="S"), "weights": np.array([0.9, 0.1])}
    info = ParameterizationInfo(["a", "fixed", "w"], [0, 0, 0], [1, 20, 2], {"fixed": 14})
    model, weights = model_posterior_samples(results, info)
    np.testing.assert_allclose(model, [[0.2, 14, 0.8], [0.4, 14, 1]])
    np.testing.assert_allclose(weights, [0.9, 0.1])


def test_downsampling_is_weighted_seeded_and_keeps_small_samples_exact():
    samples = np.arange(100)[:, None]
    weights = np.zeros(100)
    weights[-1] = 1
    selected, selected_weights = select_posterior_samples(samples, weights, max_samples=10, random_state=7)
    np.testing.assert_array_equal(selected[:, 0], np.full(10, 99))
    np.testing.assert_allclose(selected_weights, np.full(10, 0.1))
    repeat, _ = select_posterior_samples(samples, weights, max_samples=10, random_state=7)
    np.testing.assert_array_equal(selected, repeat)
    full, exact_weights = select_posterior_samples(samples, weights, max_samples=200, random_state=7)
    np.testing.assert_array_equal(full, samples)
    np.testing.assert_array_equal(exact_weights, weights)


def test_diagnostic_source_prior_draws_preserve_named_subset_and_manual_remainder(tmp_path):
    rng = np.random.default_rng(3)
    samples = rng.normal(loc=[0.8, 0.2], scale=[0.1, 0.05], size=(100, 2))
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as stream:
        stream["posterior_samples"] = samples
        stream["parameter_names"] = np.array(["w", "a"], dtype="S")
    config = {"prior_source": {"posterior_file": path, "parameter_names": ["a", "w"]},
              "type": ["uniform", "uniform", "log"]}
    first = sample_prior(config, ["a", "w", "new"], [0.1, 0.5, 1], [0.4, 1.2, 10], 2000, 7)
    second = sample_prior(config, ["a", "w", "new"], [0.1, 0.5, 1], [0.4, 1.2, 10], 2000, 7)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[:, :2].mean(axis=0), [0.2, 0.8], atol=0.02)
    assert np.mean(np.log(first[:, 2])) == pytest.approx(np.log(10)/2, abs=0.04)
