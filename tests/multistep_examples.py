"""Weighted two-dataset linear inference with independent analytic references."""

import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, multivariate_normal, norm

from bayesian import data_IO
from tests.sampler_examples import (
    DATA,
    ERROR,
    MATRIX,
    PRIOR_MEAN,
    PRIOR_STD,
    build_gaussian_example,
    gaussian_reference,
)
from tests.test_posterior_priors import write_posterior

SECOND_DATA = np.array([-0.15, 0.75])
NAMES = ["AlphaS", "nucleon_width"]


def build_multistep_example(root: Path, sampler: str, method: str):
    config = build_gaussian_example(root, sampler)
    raw = yaml.safe_load(config.config_file.read_text())
    analysis = raw["analyses"].pop("gaussian")
    # The sequential CSV schema requires factorizable coupling and soft priors;
    # a general correlated posterior source has no such factorization.
    analysis["parameters"]["mcmc"]["export_hard_likelihood_samples"] = False
    raw["analyses"] = {"first": deepcopy(analysis), "second": deepcopy(analysis)}
    for name in raw["analyses"]:
        shutil.copytree(root / "gaussian_exponential", root / f"{name}_exponential")
    observation = data_IO.read_dict_from_h5(root / "second_exponential", "observables.h5", verbose=False)
    next(iter(observation["Data"].values()))["y"] = SECOND_DATA
    data_IO.write_dict_to_h5(observation, root / "second_exponential", "observables.h5", verbose=False)
    _, _, mean, covariance = gaussian_reference(np.zeros((1, 2)))
    rng = np.random.default_rng(107)
    proposal_mean = mean + np.array([0.08, -0.05])
    proposal_cov = 1.4 * covariance
    samples = rng.multivariate_normal(proposal_mean, proposal_cov, 500)
    log_weights = multivariate_normal.logpdf(samples, mean=mean, cov=covariance)
    log_weights -= multivariate_normal.logpdf(samples, mean=proposal_mean, cov=proposal_cov)
    weights = np.exp(log_weights - logsumexp(log_weights))
    # Reverse model order and include a nuisance column to force named projection.
    source_samples = np.column_stack([rng.normal(size=len(samples)), samples[:, 1], samples[:, 0]])
    write_posterior(root / "first_exponential" / "posterior.h5", source_samples,
                    ["old_nuisance", "nucleon_width", "AlphaS"], weights)
    workflow = {
        "type": "multistep", "steps": ["first", "second"],
        "prior_sources": {"second": {"posterior_file": "posterior.h5", "method": method}},
    }
    raw["inference_workflows"] = {"two_step": workflow}
    config.config_file.write_text(yaml.safe_dump(raw))
    return config.config_file, raw, samples, weights


def posterior_reference(source_samples, weights, method):
    """Exact Gaussian or common-covariance KDE mixture update with dataset B."""
    if method == "gaussian":
        centers = np.average(source_samples, axis=0, weights=weights)[None, :]
        covariance = np.cov(source_samples.T, aweights=weights)
        prior_weights = np.ones(1)
    else:
        centers = source_samples
        covariance = gaussian_kde(source_samples.T, weights=weights).covariance
        prior_weights = weights
    precision = np.linalg.inv(covariance)
    noise = np.diag(ERROR**2)
    updated_cov = np.linalg.inv(precision + MATRIX.T @ np.linalg.inv(noise) @ MATRIX)
    updated_centers = (centers @ precision + SECOND_DATA / ERROR**2 @ MATRIX) @ updated_cov
    predictive = noise + MATRIX @ covariance @ MATRIX.T
    delta = SECOND_DATA - centers @ MATRIX.T
    log_weights = np.log(prior_weights) + multivariate_normal.logpdf(delta, mean=[0, 0], cov=predictive)
    updated_weights = np.exp(log_weights - logsumexp(log_weights))
    mean = updated_weights @ updated_centers
    centered = updated_centers - mean
    covariance = updated_cov + np.einsum("i,ij,ik->jk", updated_weights, centered, centered)
    return mean, covariance, updated_centers, updated_cov, updated_weights


def joint_reference():
    precision = np.diag(1 / PRIOR_STD**2) + 2 * MATRIX.T @ np.diag(1 / ERROR**2) @ MATRIX
    covariance = np.linalg.inv(precision)
    mean = covariance @ (PRIOR_MEAN / PRIOR_STD**2 + MATRIX.T @ ((DATA + SECOND_DATA) / ERROR**2))
    return mean, covariance


def reference_cdf(x, dimension, reference):
    _mean, _cov, centers, covariance, weights = reference
    return norm.cdf((np.asarray(x)[:, None] - centers[:, dimension]) / np.sqrt(covariance[dimension, dimension])) @ weights
