"""Route the real two-step workflow and compare both samplers with exact updates."""

import shutil
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from bayesian import inference_workflows
from bayesian.prior import load_posterior_as_prior
from tests.multistep_examples import (
    ERROR,
    MATRIX,
    NAMES,
    SECOND_DATA,
    build_multistep_example,
    joint_reference,
    posterior_reference,
    reference_cdf,
)


@pytest.mark.parametrize("method", ["kde", "gaussian"])
@pytest.mark.parametrize("sampler", ["emcee", "pocoMC"])
def test_two_step_sampler_matches_exact_joint_update(tmp_path, monkeypatch, sampler, method):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    path, raw, source, weights = build_multistep_example(tmp_path, sampler, method)
    original = deepcopy(raw["analyses"])
    inference_workflows.process_inference_workflow(path, tmp_path, raw["inference_workflows"], raw["analyses"])
    assert raw["analyses"] == original
    reference = posterior_reference(source, weights, method)
    analytic_mean, analytic_cov = joint_reference()
    np.testing.assert_allclose(reference[0], analytic_mean, atol=0.025, rtol=0)
    np.testing.assert_allclose(reference[1], analytic_cov, atol=0.003, rtol=0)
    prior = load_posterior_as_prior(tmp_path / "first_exponential" / "posterior.h5", method, param_names=NAMES)
    with h5py.File(tmp_path / "two_step_first_second" / "posterior.h5") as stream:
        samples = stream["posterior_samples"][:]
        result_weights = stream["weights"][:] if "weights" in stream else np.ones(len(samples))
        result_weights /= result_weights.sum()
        likelihood = -0.5 * np.sum(((samples @ MATRIX.T - SECOND_DATA) / ERROR) ** 2, axis=1) - np.log(ERROR).sum()
        np.testing.assert_allclose(stream["log_prior_bayesian"][:], prior(samples), atol=1e-10)
        np.testing.assert_allclose(stream["log_likelihood_hard"][:], likelihood, atol=1e-10)
        np.testing.assert_allclose(stream["log_target"][:], likelihood + prior(samples), atol=1e-10)
        assert list(stream["parameter_names"][:].astype(str)) == NAMES
        mean = result_weights @ samples
        centered = samples - mean
        covariance = np.einsum("i,ij,ik->jk", result_weights, centered, centered)
        np.testing.assert_allclose(mean, reference[0], atol=0.04, rtol=0)
        np.testing.assert_allclose(covariance, reference[1], atol=0.006, rtol=0)
        for dimension in range(2):
            order = np.argsort(samples[:, dimension])
            cumulative = np.cumsum(result_weights[order])
            expected = reference_cdf(samples[order, dimension], dimension, reference)
            distance = max(np.max(np.abs(cumulative - expected)),
                           np.max(np.abs(cumulative - result_weights[order] - expected)))
            assert distance < 0.14


@pytest.mark.parametrize("style", ["plural", "singular", "default"])
def test_workflow_source_paths_closure_and_no_config_mutation(tmp_path, monkeypatch, style):
    path, raw, _samples, _weights = build_multistep_example(tmp_path, "emcee", "gaussian")
    workflow = raw["inference_workflows"]["two_step"]
    if style == "singular":
        workflow["prior_source"] = workflow.pop("prior_sources")["second"]
        workflow["prior_source"]["posterior_file"] = str(tmp_path / "first_exponential" / "posterior.h5")
    elif style == "default":
        workflow.pop("prior_sources")
    workflow.update(run_closure_tests=True, verify_prior_vs_posterior=True)
    raw["analyses"]["second"]["validation_indices"] = [4, 6]
    original = deepcopy(raw["analyses"])
    calls, plots = [], []
    monkeypatch.setattr(inference_workflows.mcmc, "run_mcmc", lambda config, **_kwargs: calls.append(config))
    monkeypatch.setattr(inference_workflows, "verify_prior_vs_posterior", lambda **kwargs: plots.append(kwargs))
    inference_workflows.process_inference_workflow(path, tmp_path, raw["inference_workflows"], raw["analyses"])
    assert raw["analyses"] == original
    assert len(calls) == 3
    for config in calls:
        source = config.analysis_config["parameterization"]["exponential"]["prior"]["prior_source"]
        assert Path(source["posterior_file"]) == tmp_path.resolve() / "first_exponential" / "posterior.h5"
        assert source["parameter_names"] == NAMES
        assert config.input_analysis_dir == tmp_path / "second_exponential"
        assert config.output_dir == tmp_path / "two_step_first_second"
    assert len({config.mcmc_output_dir for config in calls}) == 3
    assert plots[0]["posterior_file"] == source["posterior_file"]
    assert plots[0]["method"] == source["method"]
    assert plots[0]["param_names"] == NAMES


def test_multiple_parameterizations_get_separate_outputs(tmp_path, monkeypatch):
    path, raw, _samples, _weights = build_multistep_example(tmp_path, "emcee", "kde")
    for name, analysis in raw["analyses"].items():
        analysis["parameterizations"].append("alternate")
        analysis["parameterization"]["alternate"] = deepcopy(analysis["parameterization"]["exponential"])
        shutil.copytree(tmp_path / f"{name}_exponential", tmp_path / f"{name}_alternate")
    calls = []
    monkeypatch.setattr(inference_workflows.mcmc, "run_mcmc", calls.append)
    inference_workflows.process_inference_workflow(path, tmp_path, raw["inference_workflows"], raw["analyses"])
    assert len(calls) == 2
    assert calls[0].output_dir != calls[1].output_dir
    assert calls[1].output_dir.name == "alternate"
    source = calls[1].analysis_config["parameterization"]["alternate"]["prior"]["prior_source"]
    assert Path(source["posterior_file"]).parent.name == "first_alternate"


@pytest.mark.parametrize("steps", [["first"], ["first", {}], ["first", "first"], ["first", "unknown"]])
def test_invalid_steps_raise_before_sampling(tmp_path, steps):
    path, raw, _samples, _weights = build_multistep_example(tmp_path, "emcee", "kde")
    raw["inference_workflows"]["two_step"]["steps"] = steps
    with pytest.raises(ValueError, match=r"analysis|names"):
        inference_workflows.process_inference_workflow(path, tmp_path, raw["inference_workflows"], raw["analyses"])


def test_changed_fixed_parameter_requires_conditioning(tmp_path):
    path, raw, _samples, _weights = build_multistep_example(tmp_path, "emcee", "kde")
    raw["analyses"]["second"]["parameterization"]["exponential"]["fixed_parameters"]["normalization"] = 15
    with pytest.raises(ValueError, match="same fixed"):
        inference_workflows.process_inference_workflow(path, tmp_path, raw["inference_workflows"], raw["analyses"])


def test_missing_source_fails_before_sampling(tmp_path, monkeypatch):
    path, raw, _samples, _weights = build_multistep_example(tmp_path, "emcee", "kde")
    (tmp_path / "first_exponential" / "posterior.h5").unlink()
    calls = []
    monkeypatch.setattr(inference_workflows.mcmc, "run_mcmc", calls.append)
    with pytest.raises(FileNotFoundError):
        inference_workflows.process_inference_workflow(path, tmp_path, raw["inference_workflows"], raw["analyses"])
    assert not calls


def test_correlated_source_csv_export_fails_before_sampling(tmp_path):
    path, raw, _samples, _weights = build_multistep_example(tmp_path, "emcee", "kde")
    raw["analyses"]["second"]["parameters"]["mcmc"]["export_hard_likelihood_samples"] = True
    with pytest.raises(ValueError, match="factorized"):
        inference_workflows.process_inference_workflow(path, tmp_path, raw["inference_workflows"], raw["analyses"])
