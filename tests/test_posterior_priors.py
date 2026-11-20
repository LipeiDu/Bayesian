"""Regressions for the original weighted posterior-to-prior interface."""

import h5py
import numpy as np
import pytest
from scipy.stats import gaussian_kde, multivariate_normal, norm

from bayesian.mcmc import _draw_prior_start
from bayesian.model_discrepancy import parse_discrepancy_group_settings
from bayesian.parameterization import ParameterizationInfo
from bayesian.prior import load_posterior_as_prior, make_log_prior_fn, verify_prior_vs_posterior


def write_posterior(path, samples, names, weights=None):
    with h5py.File(path, "w") as stream:
        stream["posterior_samples"] = samples
        stream.create_dataset("parameter_names", data=names, dtype=h5py.string_dtype())
        if weights is not None:
            stream["weights"] = weights
    return path


@pytest.fixture
def source(tmp_path):
    rng = np.random.default_rng(123)
    samples = rng.multivariate_normal([1.2, -0.4, 3.0], [[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1]], 150)
    weights = np.linspace(0.1, 2, len(samples))
    path = write_posterior(tmp_path / "posterior.h5", samples, ["b", "a", "nuisance"], weights)
    return path, samples, weights / weights.sum()


@pytest.mark.parametrize("method", ["kde", "gaussian"])
def test_named_weighted_prior_projection_and_seeded_shapes(source, method):
    path, samples, weights = source
    selected = samples[:, [1, 0]]
    prior = load_posterior_as_prior(path, method, param_names=["a", "b"])
    points = np.array([[0.1, 0.2], [-0.3, 1.7]])
    if method == "kde":
        expected = gaussian_kde(selected.T, weights=weights).logpdf(points.T)
    else:
        expected = multivariate_normal.logpdf(
            points, mean=np.average(selected, axis=0, weights=weights), cov=np.cov(selected.T, aweights=weights)
        )
    np.testing.assert_allclose(prior(points), expected)
    assert prior.sample(1, random_state=5).shape == (1, 2)
    np.testing.assert_array_equal(prior.sample(12, random_state=5), prior.sample(12, random_state=5))


@pytest.mark.parametrize("method", ["kde", "gaussian"])
def test_univariate_prior_and_far_tail_are_finite(tmp_path, method):
    samples = np.linspace(-1, 1, 50)[:, None]
    path = write_posterior(tmp_path / "one.h5", samples, ["a"])
    prior = load_posterior_as_prior(path, method)
    assert np.isfinite(prior([[1000.0]])[0])
    assert prior.sample(1).shape == (1, 1)
    assert prior.sample(3).shape == (3, 1)


@pytest.mark.parametrize("weights", [[1, -1, 1], [0, 0, 0], [1, np.nan, 1], [1, 2]])
def test_invalid_posterior_weights_raise(tmp_path, weights):
    path = write_posterior(tmp_path / "bad.h5", [[0], [1], [2]], ["a"], weights)
    with pytest.raises(ValueError, match="weights"):
        load_posterior_as_prior(path)


@pytest.mark.parametrize("names", [["b", "b", "nuisance"], ["b", "a"]])
def test_invalid_source_names_raise(source, names):
    path, samples, weights = source
    write_posterior(path, samples, names, weights)
    with pytest.raises(ValueError, match="parameter_names"):
        load_posterior_as_prior(path, param_names=["a", "b"])


def test_missing_requested_name_raises(source):
    with pytest.raises(ValueError, match="missing"):
        load_posterior_as_prior(source[0], param_names=["unknown"])


def test_source_replaces_original_priors_and_preserves_independent_remainder(source):
    path, _samples, _weights = source
    config = {
        "prior_source": {"posterior_file": path, "method": "gaussian", "parameter_names": ["a", "b"]},
        "type": ["gaussian", "gaussian", "gaussian"],
        "mean": [500, 500, 0.2],
        "std": [0.01, 0.01, 0.3],
    }
    prior = make_log_prior_fn(config, ["a", "b", "new_discrepancy"])
    points = np.array([[0.1, 0.2, 0.4], [-0.3, 1.7, -0.2]])
    expected = load_posterior_as_prior(path, "gaussian", param_names=["a", "b"])(points[:, :2])
    expected += norm.logpdf(points[:, 2], loc=0.2, scale=0.3)
    np.testing.assert_allclose(prior(points), expected)
    with pytest.raises(ValueError, match="3 columns"):
        prior([[0.1], [0.2], [0.4]])


def test_prior_config_survives_fixed_filter_and_discrepancy_parser(source):
    info = ParameterizationInfo(["a", "fixed", "b"], [-2] * 3, [2] * 3, {"fixed": 1})
    config = {
        "prior_source": {"posterior_file": source[0], "parameter_names": ["a", "b"]},
        "type": ["uniform", "gaussian", "log"],
    }
    filtered = info.filter_prior_config(config)
    assert filtered["type"] == ["uniform", "log"]
    groups = {"g": {"discrepancy": {
        "enabled": True, "kernel_type": "kernel2", "infer_hyperparameters": True,
        "parameter_names": ["c_bar", "length_scale", "r"], "min": [0.1, 0.1, -1], "max": [1, 3, 0],
    }}}
    result = parse_discrepancy_group_settings(groups, info.sampled_names, [-2, -2], [2, 2], filtered)
    assert result[3]["prior_source"] == config["prior_source"]
    assert result[3]["type"] == ["uniform", "log", "uniform", "uniform", "uniform"]
    assert len(result[3]["mean"]) == 5
    assert config["type"] == ["uniform", "gaussian", "log"]


def test_manual_uniform_prior_needs_no_mean_or_std():
    np.testing.assert_array_equal(make_log_prior_fn({"type": ["uniform"]}, ["a"])([[0.2]]), [0])


def test_verification_uses_weighted_marginals(source, tmp_path):
    path, samples, weights = source
    prior = load_posterior_as_prior(path, "kde")
    x = np.linspace(-2, 3, 40)[:, None]
    expected = gaussian_kde(samples[:, 1][None, :], weights=weights, bw_method=prior.density.factor)
    np.testing.assert_allclose(prior.marginal(["a"])(x), expected.logpdf(x.T))
    output = tmp_path / "check.pdf"
    verify_prior_vs_posterior(path, save_plot_to=output, n_points_plot=15)
    assert output.stat().st_size > 1000
    verify_prior_vs_posterior(path, save_plot_to=output, n_points_plot=15, param_names=["a", "b"])
    assert output.stat().st_size > 1000


def test_emcee_posterior_prior_starts_are_seeded_and_inside_bounds(source):
    prior = load_posterior_as_prior(source[0], "gaussian", param_names=["a", "b"])
    starts = _draw_prior_start(prior, 20, [-1, 0], [0.5, 2], np.random.default_rng(71), seeded=True)
    repeat = _draw_prior_start(prior, 20, [-1, 0], [0.5, 2], np.random.default_rng(71), seeded=True)
    np.testing.assert_array_equal(starts, repeat)
    assert np.all(starts > [-1, 0])
    assert np.all(starts < [0.5, 2])


def test_emcee_prior_start_rejection_has_a_limit(source):
    prior = load_posterior_as_prior(source[0], "gaussian", param_names=["a", "b"])
    with pytest.raises(ValueError, match="inside the hard"):
        _draw_prior_start(prior, 20, [1e6, 1e6], [1e6 + 1, 1e6 + 1], np.random.default_rng(71), seeded=True)
