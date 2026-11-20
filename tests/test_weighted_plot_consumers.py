"""Exercise weighted plotting inputs and saved PDFs without study-only metadata."""

from types import SimpleNamespace

import h5py
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pytest

from bayesian import plot_analyses, plot_closure, plot_mcmc, plot_qhat
from bayesian.compare_sequential_observable_subsets import DensitySamples, _weighted_limits
from bayesian.hard_likelihood_diagnostics import load_posterior_samples
from tests.sampler_examples import build_gaussian_example
from tests.sequential_examples import build_sequential_example


def config_for_plots(root):
    return SimpleNamespace(
        analysis_config={"parameterization": {"exponential": {
            "names": ["AlphaS", "fixed", "w"], "min": [0.1, 0, 0], "max": [0.5, 20, 2],
            "fixed_parameters": {"fixed": 14},
        }}, "validation_indices": [0, 1]},
        parameterization="exponential", random_seed=42, output_dir=root,
    )


def test_qhat_mean_intervals_and_map_use_the_correct_measure(tmp_path, monkeypatch):
    config = config_for_plots(tmp_path)
    samples = np.array([[0.2, 14, 0.8], [0.4, 14, 1.2]])
    monkeypatch.setattr(plot_qhat, "qhat_over_T_cubed", lambda points, *_args, **_kwargs: np.atleast_2d(points)[:, 0])
    curves = {}
    original = plt.plot

    def capture(*args, **kwargs):
        curves[kwargs.get("label")] = np.asarray(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(plt, "plot", capture)
    result = plot_qhat.plot_qhat(samples, tmp_path, config, E=100, cred_level=0.8, n_x=3, n_samples=100,
                                 weights=[0.9, 0.1], plot_prior=False, plot_map=True, log_target=[-5, -1],
                                 target_design_point=np.zeros((1, 3)))
    np.testing.assert_allclose(curves["Mean"], 0.22)
    np.testing.assert_allclose(curves["Highest-target saved sample"], 0.4)
    np.testing.assert_allclose(result["qhat_mean"], 0.22)
    assert (tmp_path / "qhat_E100.pdf").stat().st_size > 1000


@pytest.mark.parametrize("flat", [True, False])
def test_closure_restores_fixed_parameters_and_uses_full_weighted_intervals(tmp_path, monkeypatch, flat):
    config = config_for_plots(tmp_path)
    directory = tmp_path / "closure/results/0"
    directory.mkdir(parents=True)
    (directory / "mcmc.h5").touch()
    samples = np.array([[0.8, 0.2], [1.2, 0.4]])
    saved = {"chain": samples if flat else samples.reshape(1, 2, 2), "weights": [0.99, 0.01],
             "parameter_names": np.array(["w", "AlphaS"], dtype="S"), "design_point": np.array([0.2, 14, 0.8])}
    monkeypatch.setattr(plot_closure.data_IO, "read_dict_from_h5", lambda *_args, **_kwargs: saved)
    received, summaries = [], []

    def qhat(points, *_args, **kwargs):
        np.testing.assert_allclose(points[:, 1], 14)
        np.testing.assert_allclose(kwargs["weights"], [0.99, 0.01])
        received.append(points)
        return {"qhat_closure_array": np.ones(50), "qhat_mean": np.ones(50), "x_array": np.arange(50), "cred_level": 0.9}

    monkeypatch.setattr(plot_closure.plot_qhat, "plot_qhat", qhat)
    monkeypatch.setattr(plot_closure.plot_qhat, "qhat_over_T_cubed", lambda *_args, **_kwargs: np.array([1]))
    monkeypatch.setattr(plot_closure, "_plot_closure_summary_qhat", lambda *_args: None)
    monkeypatch.setattr(plot_closure, "_plot_closure_summary_theta", lambda summary, *_args: summaries.append(summary))
    plot_closure.plot(config)
    assert len(received) == 2
    assert all(summary["theta_closure_array"][0] for summary in summaries)


def test_pairplot_and_observable_draws_respect_weights_and_small_posteriors(tmp_path, monkeypatch):
    config = config_for_plots(tmp_path)
    config.input_analysis_dir, config.observables_filename = tmp_path, "observables.h5"
    config.analysis_name, config.config_file = "test", tmp_path / "config.yaml"
    samples = np.array([[90, 0.8, 0.2], [91, 1.2, 0.4]])
    weights = np.array([0.99, 0.01])
    histograms = []
    original = matplotlib.axes.Axes.hist

    def hist(axis, *args, **kwargs):
        histograms.append(kwargs["weights"])
        return original(axis, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "hist", hist)
    plot_mcmc._plot_posterior_pairplot(samples, tmp_path, ["noise", "w", "AlphaS"], weights=weights)
    assert len(histograms) == 3
    np.testing.assert_allclose(histograms[0], weights)
    assert (tmp_path / "pairplot_posterior.pdf").stat().st_size > 1000
    captured = []
    monkeypatch.setattr(plot_mcmc.data_IO, "read_dict_from_h5", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(plot_mcmc.base.EmulatorOrganizationConfig, "from_config_file",
                        lambda **_kwargs: SimpleNamespace(observable_filter=None))

    def predict(points, **_kwargs):
        captured.append(points)
        return {"central_value": np.zeros((len(points), 1))}

    monkeypatch.setattr(plot_mcmc.base, "predict", predict)
    monkeypatch.setattr(plot_mcmc.data_IO, "observable_dict_from_matrix", lambda *_args, **_kwargs: {"central_value": {}})
    monkeypatch.setattr(plot_mcmc.plot_utils, "plot_observable_panels", lambda *_args, **_kwargs: None)
    plot_mcmc._plot_posterior_observables(samples, tmp_path, config, parameter_names=["noise", "w", "AlphaS"], weights=weights)
    np.testing.assert_allclose(captured[0][:, 1], 14)
    assert np.mean(captured[0][:, 0] == 0.2) > 0.95


def test_across_analysis_curves_include_every_weighted_analysis(tmp_path, monkeypatch):
    config = config_for_plots(tmp_path)
    samples = np.array([[0.2, 14, 0.8], [0.4, 14, 1.2]])
    monkeypatch.setattr(plot_qhat, "qhat_over_T_cubed", lambda points, *_args, **_kwargs: np.atleast_2d(points)[:, 0])
    means = []
    original = matplotlib.axes.Axes.plot

    def plot(axis, *args, **kwargs):
        if kwargs.get("label", "").endswith(": mean"):
            means.append(args[1])
        return original(axis, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", plot)
    names = ["first", "second", "third"]
    plot_analyses.plot_qhat_across_analyses(
        {name: {"weights": [0.9, 0.1]} for name in names}, dict.fromkeys(names, samples), tmp_path,
        dict.fromkeys(names, config), E=100, cred_level=0.8, n_x=3, n_samples=100, plot_prior=False,
    )
    assert len(means) == 3
    np.testing.assert_allclose(means, 0.22)


def test_diagnostic_thinning_and_plot_limits_do_not_discard_dominant_particles(tmp_path):
    path = tmp_path / "posterior.h5"
    with h5py.File(path, "w") as stream:
        stream["posterior_samples"] = np.arange(100)[:, None]
        stream["weights"] = np.r_[np.zeros(99), 1]
        stream["parameter_names"] = np.array(["AlphaS"], dtype="S")
    posterior = load_posterior_samples(path, label="test", max_samples=10, seed=1)
    np.testing.assert_allclose(posterior.samples, [[99]])
    np.testing.assert_allclose(posterior.weights, [1])
    assert posterior.source_sample_count == 100
    dataset = DensitySamples("test", np.array([[-100], [0], [10], [100]]), np.array([0, 0.9, 0.1, 0]), ("AlphaS",))
    np.testing.assert_allclose(_weighted_limits([dataset], "AlphaS", 0), [0, 10])


def test_prior_plot_draws_follow_configured_gaussian_and_fixed_values(tmp_path):
    config = build_gaussian_example(tmp_path, "emcee")
    first = plot_qhat._generate_prior_samples(config, n_samples=5000)
    second = plot_qhat._generate_prior_samples(config, n_samples=5000)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[:, 2], 14)
    np.testing.assert_allclose(first[:, :2].mean(axis=0), [0.1, 0.8], atol=0.02)
    assert np.std(first[:, 1]) == pytest.approx(0.4, abs=0.02)


@pytest.mark.parametrize("model_type", ["gaussian_kde", "torch_realnvp_2d"])
def test_prior_plot_draws_use_active_sequential_artifact(tmp_path, model_type):
    config = build_sequential_example(tmp_path, "emcee", model_type, 1.0)
    first = plot_qhat._generate_prior_samples(config, n_samples=128)
    second = plot_qhat._generate_prior_samples(config, n_samples=128)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (128, 3)
    assert np.all(first > [0.1, 0.5, 10])
    assert np.all(first < [0.5, 1.5, 20])
