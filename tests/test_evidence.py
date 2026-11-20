"""Independent normalized evidence and checks against misleading legacy estimates."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from bayesian import mcmc
from bayesian.evidence import compute_from_sampler, evidence_record, prior_log_normalization
from tests.sampler_examples import DATA, ERROR, MATRIX, PRIOR_MEAN, PRIOR_STD, build_gaussian_example


def test_native_evidence_corrects_reference_box_and_gaussian_constants():
    # L*=exp(-theta²/(2*sigma²)); prior*=1 over [-2,2].
    sigma = 0.2
    raw_integral = np.sqrt(2*np.pi)*sigma
    native = SimpleNamespace(evidence=lambda: (np.log(raw_integral/4), 0.02))
    result = evidence_record(native, "pocoMC", "native", lower=[-2], upper=[2], prior_config=None,
                             n_observables=1)
    assert result["log_target_integral"] == pytest.approx(np.log(raw_integral))
    assert result["logZ"] == pytest.approx(np.log(sigma/4))
    assert result["logZ_err"] == 0.02
    value, error = compute_from_sampler(native, "pocoMC", "native", lower=[-2], upper=[2], n_observables=1)
    assert value == result["logZ"]
    assert error == 0.02


def test_gaussian_and_log_prior_mass_includes_truncation_and_log_range():
    config = {"type": ["gaussian", "log"], "mean": [0, None], "std": [1, None]}
    expected = np.log(norm.cdf(1)-norm.cdf(-1)) + np.log(np.log(10))
    assert prior_log_normalization(config, [-1, 1], [1, 10]) == pytest.approx(expected)
    assert np.isfinite(prior_log_normalization({"type": ["gaussian"], "mean": [0], "std": [1]}, [10], [11]))


@pytest.mark.parametrize("source", [True, False])
def test_derived_prior_integral_is_not_mislabeled_absolute_evidence(source):
    native = SimpleNamespace(evidence=lambda: (1.0, 0.1))
    prior = {"prior_source": {"posterior_file": "not_read_here"}} if source else None
    sequential = None if source else SimpleNamespace(hard_likelihood_power=0.25)
    result = evidence_record(native, "pocoMC", "native", lower=[0], upper=[2], prior_config=prior,
                             n_observables=2, sequential_config=sequential)
    assert result["status"] == "target_integral_only"
    assert "logZ" not in result
    assert result["log_target_integral"] == pytest.approx(1+np.log(2))


def test_laplace_uses_optimized_mode_and_local_hessian():
    mean, std = 0.3, 0.2
    def target(x):
        return -0.5*((x[:, 0]-mean)/std)**2
    sampler = SimpleNamespace(get_chain=lambda **_kwargs: np.array([[0.1], [0.4], [0.6]]),
                              get_log_prob=lambda **_kwargs: target(np.array([[0.1], [0.4], [0.6]])))
    result = evidence_record(sampler, "emcee", "laplace", lower=[-2], upper=[2], prior_config=None,
                             n_observables=1, log_target_fn=target)
    assert result["log_target_integral"] == pytest.approx(np.log(np.sqrt(2*np.pi)*std), abs=1e-6)
    np.testing.assert_allclose(result["mode"], [mean], atol=1e-6)


@pytest.mark.parametrize("sampler_type", ["emcee", "pocoMC"])
def test_harmonic_mean_is_retired(sampler_type):
    with pytest.raises(ValueError, match="retired"):
        compute_from_sampler(None, sampler_type, "harmonic_mean", lower=[0], upper=[1], n_observables=1)


def test_bare_sampler_evidence_cannot_claim_a_normalized_result():
    with pytest.raises(ValueError, match="normalization"):
        compute_from_sampler(SimpleNamespace(evidence=lambda: (1, 0.1)), "pocoMC")


@pytest.mark.parametrize("sampler_type", ["emcee", "pocoMC"])
def test_real_sampler_evidence_matches_analytic_gaussian(tmp_path, monkeypatch, sampler_type):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    config = build_gaussian_example(tmp_path, sampler_type)
    config.compute_evidence = True
    config.evidence_method = None
    if sampler_type == "pocoMC":
        config.analysis_config["parameters"]["mcmc"]["pocoMC"]["n_importance_samples_for_evidence"] = 5000
    mcmc.run_mcmc(config)
    prior = {"type": ["gaussian", "gaussian"], "mean": PRIOR_MEAN, "std": PRIOR_STD}
    mass = prior_log_normalization(prior, [-2, -1], [2, 3])
    expected = multivariate_normal.logpdf(DATA, mean=MATRIX @ PRIOR_MEAN,
                                        cov=np.diag(ERROR**2) + MATRIX @ np.diag(PRIOR_STD**2) @ MATRIX.T) - mass
    for filename in ["mcmc.h5", "posterior.h5"]:
        with h5py.File(config.mcmc_output_dir / filename) as stream:
            assert stream["evidence"]["status"][()].decode() == "normalized"
            tolerance = 1e-5 if sampler_type == "emcee" else 0.05
            assert stream["logZ"][()] == pytest.approx(expected, abs=tolerance)
            assert stream["evidence"]["logZ"][()] == stream["logZ"][()]
            if sampler_type == "pocoMC":
                assert np.isfinite(stream["logZ_err"][()])
                assert stream["logZ_err"][()] > 0


@pytest.mark.parametrize("mean", [2.5, 1.99])
def test_laplace_refuses_boundary_dominated_estimates(mean):
    def target(x):
        return -0.5*((x[:, 0]-mean)/0.2)**2

    sampler = SimpleNamespace(get_chain=lambda **_kwargs: np.array([[1.8], [1.9]]),
                              get_log_prob=lambda **_kwargs: target(np.array([[1.8], [1.9]])))
    with pytest.raises(ValueError, match=r"boundary|outside"):
        evidence_record(sampler, "emcee", "laplace", lower=[-2], upper=[2], prior_config=None,
                        n_observables=1, log_target_fn=target)


@pytest.mark.parametrize("bounds", [([1], [0]), ([0], [np.inf]), ([0, 1], [2])])
def test_evidence_rejects_invalid_reference_box(bounds):
    with pytest.raises(ValueError, match="bounds"):
        evidence_record(SimpleNamespace(evidence=lambda: (0, 0)), "pocoMC", "native", lower=bounds[0],
                        upper=bounds[1], prior_config=None, n_observables=1)


def test_disabled_evidence_does_not_require_density_context():
    estimate, error = compute_from_sampler(None, "emcee", "none")
    assert np.isnan(estimate)
    assert np.isnan(error)
