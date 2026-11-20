"""Exercise actual sampler adapters, serialized inputs, and saved density fields."""

import csv
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from bayesian import log_posterior, mcmc
from bayesian.emulation import base
from bayesian.model_discrepancy import build_observable_xcoords_per_group
from tests.sampler_examples import build_gaussian_example, gaussian_reference


@pytest.fixture(params=["emcee", "pocoMC"])
def gaussian_run(request, tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    config = build_gaussian_example(tmp_path, request.param)
    mcmc.run_mcmc(config)
    return config


@pytest.mark.slow
def test_sampler_outputs_match_analytic_density(gaussian_run) -> None:
    config = gaussian_run
    with h5py.File(config.mcmc_output_dir / "posterior.h5") as stream:
        samples = stream["posterior_samples"][:]
        assert list(stream["parameter_names"][:].astype(str)) == ["AlphaS", "nucleon_width"]
        likelihood, prior, expected_mean, expected_covariance = gaussian_reference(samples)
        np.testing.assert_allclose(stream["log_likelihood_hard"][:], likelihood, atol=1e-11)
        np.testing.assert_allclose(stream["log_prior_bayesian"][:], prior, atol=1e-11)
        np.testing.assert_allclose(stream["log_target"][:], likelihood + prior, atol=1e-11)
        np.testing.assert_allclose(stream["log_likelihood_hard_tempered"][:], likelihood, atol=1e-11)
        weights = stream["weights"][:] if "weights" in stream else np.ones(len(samples))
        assert np.all(np.isfinite(weights))
        assert np.all(weights >= 0.0)
        weights /= weights.sum()
        mean = weights @ samples
        centered = samples - mean
        covariance = np.einsum("i,ij,ik->jk", weights, centered, centered)
        np.testing.assert_allclose(mean, expected_mean, atol=0.035, rtol=0)
        np.testing.assert_allclose(covariance, expected_covariance, atol=0.007, rtol=0)
        if config.mcmc_package == "pocoMC":
            np.testing.assert_allclose(stream["logp"][:], -np.log(16.0))
            np.testing.assert_allclose(stream["logl"][:], likelihood + prior, atol=1e-11)
        else:
            np.testing.assert_allclose(stream["log_prob"][:], likelihood + prior, atol=1e-11)
    with h5py.File(config.mcmc_output_dir / "mcmc.h5") as stream:
        np.testing.assert_allclose(stream["chain"][:].reshape(samples.shape), samples)
        np.testing.assert_allclose(stream["log_target"][:].reshape(-1), likelihood + prior, atol=1e-11)
        np.testing.assert_allclose(stream["log_likelihood_hard"][:].reshape(-1), likelihood, atol=1e-11)
    with (config.mcmc_output_dir / "hard_likelihood_samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    np.testing.assert_allclose([float(row["log_likelihood_hard"]) for row in rows], likelihood, atol=1e-11)
    np.testing.assert_allclose([float(row["log_posterior_practical"]) for row in rows], likelihood + prior, atol=1e-11)
    np.testing.assert_allclose([float(row["normalization"]) for row in rows], 14.0)
    if config.mcmc_package == "pocoMC":
        np.testing.assert_allclose([float(row["sample_weight"]) for row in rows], weights)


@pytest.mark.parametrize("stale", [False, True])
def test_emcee_initializes_parent_before_walker_sampling(tmp_path, monkeypatch, stale) -> None:
    config = build_gaussian_example(tmp_path, "emcee")
    monkeypatch.setattr(
        log_posterior, "g_log_prior_fn", (lambda samples: np.full(len(samples), 999.0)) if stale else None
    )
    monkeypatch.setattr(log_posterior, "g_sequential_log_prior_fn", None)

    class CheckParent:
        def __enter__(self):
            assert log_posterior.g_log_prior_fn is not None
            assert log_posterior.g_param_names == ["AlphaS", "nucleon_width"]
            samples = np.array([[0.1, 0.8]])
            np.testing.assert_allclose(log_posterior.g_log_prior_fn(samples), gaussian_reference(samples)[1])
            raise StopIteration

        def __exit__(self, *_args):
            return False

    class Context:
        def Pool(self, **_kwargs):
            return CheckParent()

    monkeypatch.setattr(mcmc.multiprocessing, "get_context", lambda _method: Context())
    with pytest.raises(StopIteration):
        mcmc.run_mcmc(config)


@pytest.mark.parametrize("power", [0.0, 0.25, 1.0])
def test_raw_likelihood_is_separate_from_prior(tmp_path, monkeypatch, power) -> None:
    config = build_gaussian_example(tmp_path, "emcee")
    emulation = base.EmulatorOrganizationConfig.from_config_file(
        analysis_name=config.analysis_name,
        parameterization=config.parameterization,
        config_file=config.config_file,
        analysis_config=config.analysis_config,
    )
    experimental = base.data_IO.data_array_from_h5(config.input_analysis_dir, "observables.h5")
    log_posterior.initialize_pool_variables(
        [-2.0, -1.0],
        [2.0, 3.0],
        emulation,
        emulation.read_all_emulator_groups(),
        experimental,
        {"linear": np.zeros((2, 2))},
        {"type": ["gaussian", "gaussian"], "mean": [0.1, 0.8], "std": [0.5, 0.4]},
        {},
        build_observable_xcoords_per_group(config, emulation, experimental),
        ["AlphaS", "nucleon_width"],
        [],
        ["AlphaS", "nucleon_width", "normalization"],
        {"normalization": 14.0},
    )
    samples = np.array([[0.1, 0.8], [0.3, 1.0]])
    sequential_config = SimpleNamespace(hard_likelihood_power=power)
    monkeypatch.setattr(log_posterior, "g_sequential_inference_config", sequential_config)
    likelihood, prior, _, _ = gaussian_reference(samples)
    np.testing.assert_allclose(log_posterior.log_likelihood(samples), likelihood)
    target = log_posterior.log_posterior(samples)
    np.testing.assert_allclose(target, power * likelihood + prior)
    components = mcmc._posterior_density_components(samples, target, sequential_config)
    np.testing.assert_allclose(components["log_likelihood_hard"], likelihood)
    np.testing.assert_allclose(components["log_likelihood_hard_tempered"], power * likelihood)
    assert np.isneginf(log_posterior.log_likelihood([[3.0, 0.8]])[0])
    assert log_posterior.log_posterior([[3.0, 0.8]], set_to_infinite_outside_bounds=False)[0] == -1e300


def test_pocomc_receives_discrepancy_metadata(tmp_path, monkeypatch) -> None:
    config = build_gaussian_example(tmp_path, "pocoMC")
    config.analysis_config["parameters"]["emulators"]["linear"]["discrepancy"] = {
        "enabled": True,
        "kernel_type": "kernel2",
        "infer_hyperparameters": True,
        "parameter_names": ["c_bar", "length_scale", "r"],
        "min": [0.01, 0.1, -1.0],
        "max": [1.0, 3.0, 0.0],
    }
    captured = {}

    def record_arguments(*_args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mcmc, "_run_using_pocoMC", record_arguments)
    mcmc.run_mcmc(config)
    assert captured["names"] == ["AlphaS", "nucleon_width", "linear__c_bar", "linear__length_scale", "linear__r"]
    assert captured["discrepancy_enabled_groups"] == ["linear"]
    assert captured["discrepancy_config"]["linear"]["infer_hyperparameters"]
    np.testing.assert_array_equal(captured["observable_xcoords"]["linear"], [1.5, 2.5])
