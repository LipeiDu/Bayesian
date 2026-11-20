"""Recover known Gaussian and quadrature posteriors with fixed/inferred discrepancy."""

from functools import lru_cache

import h5py
import numpy as np
import pytest

from bayesian import log_posterior, mcmc, plot_discrepancy
from bayesian.emulation import base
from bayesian.model_discrepancy import (
    build_observable_xcoords_per_group,
    parse_discrepancy_group_settings,
)
from bayesian.parameterization import parameterization_info
from tests.discrepancy_examples import (
    GROUP_INDICES,
    ORDER,
    TRUE_HYPER,
    X,
    build_discrepancy_example,
    density_reference,
    kernel_reference,
    marginal_cdf,
    posterior_reference,
)

cached_reference = lru_cache()(posterior_reference)


@pytest.mark.parametrize("mode", ["fixed", "inferred"])
@pytest.mark.parametrize("sampler", ["emcee", "pocoMC"])
def test_discrepancy_sampler_matches_reference(tmp_path, monkeypatch, mode, sampler):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    config = build_discrepancy_example(tmp_path, sampler, mode)
    mcmc.run_mcmc(config)
    reference = cached_reference(mode)
    names = ["AlphaS", "nucleon_width"] + ([f"a__{name}" for name in ORDER] if mode == "inferred" else [])
    with h5py.File(config.mcmc_output_dir / "posterior.h5") as stream:
        samples = stream["posterior_samples"][:]
        weights = stream["weights"][:] if "weights" in stream else np.ones(len(samples))
        weights /= weights.sum()
        likelihood, prior = density_reference(samples, mode)
        assert list(stream["parameter_names"][:].astype(str)) == names
        np.testing.assert_allclose(stream["log_likelihood_hard"][:], likelihood, atol=2e-10)
        np.testing.assert_allclose(stream["log_prior_bayesian"][:], prior, atol=2e-10)
        np.testing.assert_allclose(stream["log_target"][:], likelihood + prior, atol=2e-10)
        mean = weights @ samples
        centered = samples - mean
        covariance = np.einsum("i,ij,ik->jk", weights, centered, centered)
        np.testing.assert_allclose(mean, reference["mean"], atol=0.09, rtol=0)
        np.testing.assert_allclose(covariance, reference["covariance"], atol=0.025, rtol=0)
        bounds = [(-2, 2), (-1, 3)] + ([(-1.2, 0.3), (0.08, 0.65), (0.2, 1.8)] if mode == "inferred" else [])
        for dimension, (lower, upper) in enumerate(bounds):
            probes = np.linspace(lower, upper, 101)
            observed = (samples[:, dimension, None] <= probes).T @ weights
            expected = marginal_cdf(probes, dimension, reference)
            assert np.max(np.abs(observed - expected)) < 0.14
    with h5py.File(config.mcmc_output_dir / "mcmc.h5") as stream:
        np.testing.assert_allclose(stream["log_target"][:].reshape(-1), likelihood + prior, atol=2e-10)

    # Exercise the actual saved-kernel plotting path for both output shapes.
    plot_discrepancy.plot(config, n_posterior_samples=2)
    with h5py.File(config.output_dir / "discrepancy_kernels.h5") as stream:
        a = GROUP_INDICES["a"]
        np.testing.assert_array_equal(stream["a"]["fixed" if mode == "fixed" else "MAP"]["observable_slices"][:],
                                      [[0, 4], [4, 9]])
        if mode == "fixed":
            expected_kernel = kernel_reference(X[a], **TRUE_HYPER)
            actual = stream["a"]["fixed"]["kernel_matrix"][:]
        else:
            with h5py.File(config.mcmc_output_dir / "posterior.h5") as posterior:
                map_index = np.argmax(posterior["log_target"][:])
                hyper = dict(zip(ORDER, posterior["posterior_samples"][map_index, 2:], strict=True))
                expected_kernel = kernel_reference(X[a], **hyper)
            actual = stream["a"]["MAP"]["kernel_matrix"][:]
        np.testing.assert_allclose(actual, expected_kernel, atol=1e-12)
    assert list(config.output_dir.glob("plot_discrepancy_group_*/*.pdf"))


def test_discrepancy_quadrature_converges():
    coarse, fine = cached_reference("inferred"), cached_reference("inferred", 49)
    np.testing.assert_allclose(coarse["mean"], fine["mean"], atol=0.002, rtol=0)
    np.testing.assert_allclose(coarse["covariance"], fine["covariance"], atol=0.002, rtol=0)


def test_likelihood_keeps_parameter_dependent_log_determinant(tmp_path):
    config = build_discrepancy_example(tmp_path, "emcee", "inferred")
    emulation = base.EmulatorOrganizationConfig.from_config_file(
        analysis_name=config.analysis_name, parameterization=config.parameterization,
        config_file=config.config_file, analysis_config=config.analysis_config,
    )
    info = parameterization_info(config.analysis_config, config.parameterization)
    names, lower, upper, prior, discrepancy, enabled = parse_discrepancy_group_settings(
        config.analysis_config["parameters"]["emulators"], info.sampled_names, info.sampled_min, info.sampled_max,
        info.filter_prior_config(config.analysis_config["parameterization"][config.parameterization]["prior"]),
    )
    data = base.data_IO.data_array_from_h5(config.input_analysis_dir, "observables.h5")
    coords = build_observable_xcoords_per_group(config, emulation, data)
    np.testing.assert_allclose(coords["a"], X[GROUP_INDICES["a"]])
    np.testing.assert_allclose(coords["b"], X[GROUP_INDICES["b"]])
    log_posterior.initialize_pool_variables(
        lower, upper, emulation, emulation.read_all_emulator_groups(), data,
        {"a": np.zeros((9, 9)), "b": np.zeros((3, 3))}, prior, discrepancy, coords, names, enabled,
        info.full_names, info.fixed_parameters,
    )
    points = np.array([[0.25, 0.9, -0.5, 0.12, 0.7], [0.25, 0.9, -0.5, 0.6, 0.7]])
    likelihood, prior_values = density_reference(points, "inferred")
    np.testing.assert_allclose(log_posterior.log_likelihood(points), likelihood, atol=2e-10)
    np.testing.assert_allclose(log_posterior.log_posterior(points), likelihood + prior_values, atol=2e-10)
    permutation = [4, 1, 0, 3, 2]
    reordered = [names[index] for index in permutation]
    log_posterior.initialize_pool_variables(
        np.asarray(lower)[permutation], np.asarray(upper)[permutation], emulation, emulation.read_all_emulator_groups(),
        data, {"a": np.zeros((9, 9)), "b": np.zeros((3, 3))}, None, discrepancy, coords, reordered, enabled,
        info.full_names, info.fixed_parameters,
    )
    np.testing.assert_allclose(log_posterior.log_likelihood(points[:, permutation]), likelihood, atol=2e-10)
