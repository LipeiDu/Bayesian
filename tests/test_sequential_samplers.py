"""Run both samplers with both versioned soft priors along a tempered path."""

import h5py
import numpy as np
import pytest
from scipy.stats import truncnorm

from bayesian import mcmc
from tests.sequential_examples import UNCERTAINTY, build_sequential_example, hard_reference, reference_soft_density


@pytest.mark.parametrize("model_type", ["gaussian_kde", "torch_realnvp_2d"])
@pytest.mark.parametrize("sampler", ["emcee", "pocoMC"])
@pytest.mark.parametrize("power", [0.0, 0.25, 1.0])
def test_real_sequential_sampler_components(tmp_path, monkeypatch, model_type, sampler, power) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    config = build_sequential_example(tmp_path, sampler, model_type, power)
    mcmc.run_mcmc(config)
    with h5py.File(config.mcmc_output_dir / "posterior.h5") as stream:
        samples = stream["posterior_samples"][:]
        assert list(stream["parameter_names"][:].astype(str)) == ["AlphaS", "nucleon_width", "normalization"]
        assert np.all(samples > [0.1, 0.5, 10.0])
        assert np.all(samples < [0.5, 1.5, 20.0])
        prior = reference_soft_density(samples[:, 1:], model_type)
        likelihood = hard_reference(samples)
        tolerance = 5e-5 if model_type == "torch_realnvp_2d" else 2e-9
        np.testing.assert_allclose(stream["log_prior_bayesian"][:], prior, atol=tolerance, rtol=2e-6)
        np.testing.assert_allclose(stream["log_likelihood_hard"][:], likelihood, atol=tolerance, rtol=2e-6)
        np.testing.assert_allclose(stream["log_target"][:], power * likelihood + prior, atol=tolerance, rtol=2e-6)
        np.testing.assert_allclose(
            stream["log_likelihood_hard_tempered"][:], power * likelihood, atol=tolerance, rtol=2e-6
        )
        assert float(stream["sequential_inference"]["hard_likelihood_power"][()]) == power
        weights = stream["weights"][:] if "weights" in stream else np.ones(len(samples))
        weights /= weights.sum()
        order = np.argsort(samples[:, 0])
        if power == 0.0:
            expected_alpha = 0.3
            reference = (samples[order, 0] - 0.1) / 0.4
        else:
            sigma = UNCERTAINTY[0] / np.sqrt(power)
            distribution = truncnorm((0.1 - 0.32) / sigma, (0.5 - 0.32) / sigma, loc=0.32, scale=sigma)
            expected_alpha = distribution.mean()
            reference = distribution.cdf(samples[order, 0])
        assert weights @ samples[:, 0] == pytest.approx(expected_alpha, abs=0.05)
        cumulative = np.cumsum(weights[order])
        distance = max(np.max(np.abs(cumulative - reference)), np.max(np.abs(cumulative - weights[order] - reference)))
        assert distance < 0.15
    with h5py.File(config.mcmc_output_dir / "mcmc.h5") as stream:
        np.testing.assert_allclose(
            stream["log_target"][:].reshape(-1), power * likelihood + prior, atol=tolerance, rtol=2e-6
        )
