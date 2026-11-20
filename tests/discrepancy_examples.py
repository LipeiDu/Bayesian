"""A two-group synthetic calibration and independent discrepancy posterior quadrature."""

import pickle
from pathlib import Path

import numpy as np
import yaml
from scipy.special import logsumexp
from scipy.stats import norm

from bayesian import data_IO, mcmc
from tests.sampler_examples import PRIOR_MEAN, PRIOR_STD, build_gaussian_example

X = np.array([1.0, 1.3, 1.6, 1.9, 1.1, 2.0, 3.0, 2.2, 2.5, 2.8, 3.1, 3.4])
MATRIX = np.column_stack([np.sin(1.8 * X), np.cos(0.6 * X)])
GROUP_INDICES = {"a": np.array([0, 1, 2, 3, 7, 8, 9, 10, 11]), "b": np.array([4, 5, 6])}
MEASUREMENT_ERROR = 0.07
EMULATOR_ERROR = 0.03
NOISE = MEASUREMENT_ERROR**2 + EMULATOR_ERROR**2
TRUTH = np.array([0.25, 0.9])
TRUE_HYPER = {"c_bar": 0.3, "length_scale": 0.7, "r": -0.5, "s": 0.07}
BOUNDS = {"r": (-1.2, 0.3), "c_bar": (0.08, 0.65), "length_scale": (0.2, 1.8)}
ORDER = ["r", "c_bar", "length_scale"]


class FeatureEmulator:
    def __init__(self, indices):
        self.indices = indices

    def predict(self, parameters, return_std):
        assert return_std
        assert parameters.shape[1] == 3
        np.testing.assert_array_equal(parameters[:, 2], 14.0)
        return parameters[:, :2] @ MATRIX[self.indices].T, np.full((len(parameters), len(self.indices)), EMULATOR_ERROR)


def kernel_reference(x, c_bar, length_scale, r, s=0.0):
    """Independent scalar-input reference; no production discrepancy code."""
    amplitude = c_bar * x**r
    return np.outer(amplitude, amplitude) * np.exp(-(x[:, None] - x)**2 / (2 * length_scale**2)) + s**2


def total_covariance(mode, hyper=None):
    kernel = np.zeros((len(X), len(X)))
    p = TRUE_HYPER if hyper is None else hyper
    a = GROUP_INDICES["a"]
    kernel[np.ix_(a, a)] = kernel_reference(X[a], p["c_bar"], p["length_scale"], p["r"],
                                          p.get("s", 0) if mode == "fixed" else 0)
    b = GROUP_INDICES["b"]
    kernel[np.ix_(b, b)] = kernel_reference(X[b], 0.08, 0.4, 0)
    return kernel + np.eye(len(X)) * NOISE


def generated_data(mode):
    return MATRIX @ TRUTH + np.linalg.cholesky(total_covariance(mode)) @ np.random.default_rng(418).normal(size=len(X))


def build_discrepancy_example(root: Path, sampler: str, mode: str):
    original = build_gaussian_example(root, sampler, seed=271828)
    raw = yaml.safe_load(original.config_file.read_text())
    analysis = raw["analyses"]["gaussian"]
    analysis["parameters"]["mcmc"].update(n_walkers=32, n_burn_steps=250, n_sampling_steps=600,
                                            export_hard_likelihood_samples=False)
    fixed_a = {"enabled": True, "kernel_type": "kernel1", "infer_hyperparameters": False,
               "parameter_names": ["s", "r", "c_bar", "length_scale"],
               "fixed_params": [TRUE_HYPER[name] for name in ["s", "r", "c_bar", "length_scale"]]}
    inferred_a = {"enabled": True, "kernel_type": "kernel2", "infer_hyperparameters": True,
                  "parameter_names": ORDER, "min": [BOUNDS[name][0] for name in ORDER],
                  "max": [BOUNDS[name][1] for name in ORDER],
                  "prior": {"type": ["uniform", "gaussian", "uniform"],
                            "mean": [None, 0.3, None], "std": [None, 0.18, None]}}
    analysis["parameters"]["emulators"] = {
        "a": {"emulator_name": "sk_learn", "n_pc": None, "observable_list": ["hadron__pt_a", "hadron__pt_z"],
              "discrepancy": fixed_a if mode == "fixed" else inferred_a},
        "b": {"emulator_name": "sk_learn", "n_pc": None, "observable_list": ["hadron__pt_m"],
              "discrepancy": {"enabled": True, "kernel_type": "kernel2", "infer_hyperparameters": False,
                              "fixed_params": [0.08, 0.4, 0.0]}},
    }
    raw["analyses"] = {"discrepancy": analysis}
    original.config_file.write_text(yaml.safe_dump(raw))
    output = root / "discrepancy_exponential"
    data = generated_data(mode)
    observations = {"Data": {}, "Prediction": {}}
    selections = [("5020__PbPb__hadron__pt_a____0-5", np.arange(4)),
                  ("5020__PbPb__hadron__pt_m____0-5", np.arange(4, 7)),
                  ("5020__PbPb__hadron__pt_z____10-20", np.arange(7, 12))]
    # Verify the backend sort keeps group A interleaved with group B.
    for name, indices in selections:
        observations["Data"][name] = {"y": data[indices], "y_err": np.full(len(indices), MEASUREMENT_ERROR),
                                      "xmin": X[indices] - 0.04, "xmax": X[indices] + 0.04}
        observations["Prediction"][name] = {"y": np.zeros((len(indices), 4))}
    expected_order = [name for name, _ in selections]
    assert data_IO.sorted_observable_list_from_dict(observations) == expected_order
    data_IO.write_dict_to_h5(observations, output, "observables.h5", verbose=False)
    for group, chunks in {"a": [np.arange(4), np.arange(7, 12)], "b": [np.arange(4, 7)]}.items():
        with (output / f"emulation_group_{group}.pkl").open("wb") as stream:
            pickle.dump({"emulators": [FeatureEmulator(indices) for indices in chunks]}, stream)
    return mcmc.MCMCConfig(analysis_name="discrepancy", parameterization="exponential",
                           analysis_config=analysis, config_file=original.config_file)


def density_reference(samples, mode):
    prior = norm.logpdf(samples[:, :2], PRIOR_MEAN, PRIOR_STD).sum(axis=1)
    likelihood = []
    for sample in samples:
        hyper = dict(zip(ORDER, sample[2:], strict=True)) if mode == "inferred" else None
        covariance = total_covariance(mode, hyper)
        residual = MATRIX @ sample[:2] - generated_data(mode)
        likelihood.append(-0.5 * (residual @ np.linalg.solve(covariance, residual)
                                   + np.linalg.slogdet(covariance)[1]))
    if mode == "inferred":
        prior += norm.logpdf(samples[:, 3], 0.3, 0.18)
    return np.asarray(likelihood), prior


def posterior_reference(mode, n_grid=35):
    """Integrate hyperparameters, analytically conditioning the Gaussian model.

    Midpoint quadrature is independent of the backend's kernels and likelihood.
    A second resolution is used to check numerical convergence in the tests.
    """
    axes = {name: np.linspace(*BOUNDS[name], n_grid + 1) for name in ORDER}
    axes = {name: 0.5 * (edges[1:] + edges[:-1]) for name, edges in axes.items()}
    grid = np.column_stack([values.ravel() for values in np.meshgrid(*axes.values(), indexing="ij")])
    if mode == "fixed":
        grid = np.zeros((1, 0))
    means, covariances, log_weights = [], [], []
    prior_cov = np.diag(PRIOR_STD**2)
    marginal_residual = generated_data(mode) - MATRIX @ PRIOR_MEAN
    for hyper_values in grid:
        hyper = dict(zip(ORDER, hyper_values, strict=True)) if mode == "inferred" else None
        covariance = total_covariance(mode, hyper)
        predictive_cov = covariance + MATRIX @ prior_cov @ MATRIX.T
        solved = np.linalg.solve(predictive_cov, MATRIX @ prior_cov)
        mean = PRIOR_MEAN + solved.T @ marginal_residual
        conditional_cov = prior_cov - prior_cov @ MATRIX.T @ solved
        weight = -0.5 * (marginal_residual @ np.linalg.solve(predictive_cov, marginal_residual)
                         + np.linalg.slogdet(predictive_cov)[1])
        if mode == "inferred":
            weight += norm.logpdf(hyper["c_bar"], 0.3, 0.18)
        means.append(mean)
        covariances.append(conditional_cov)
        log_weights.append(weight)
    weights = np.exp(log_weights - logsumexp(log_weights))
    means, covariances = np.asarray(means), np.asarray(covariances)
    combined = np.column_stack([means, grid])
    mean = weights @ combined
    centered = combined - mean
    covariance = np.einsum("i,ij,ik->jk", weights, centered, centered)
    covariance[:2, :2] += np.einsum("i,ijk->jk", weights, covariances)
    return {"grid": grid, "weights": weights, "conditional_means": means, "conditional_covariances": covariances,
            "mean": mean, "covariance": covariance, "axes": axes}


def marginal_cdf(points, dimension, reference):
    points = np.asarray(points)
    result = np.empty(len(points))
    for start in range(0, len(points), 128):
        chunk = points[start:start + 128]
        if dimension < 2:
            standardized = (chunk[:, None] - reference["conditional_means"][:, dimension])
            standardized /= np.sqrt(reference["conditional_covariances"][:, dimension, dimension])
            values = norm.cdf(standardized)
        else:
            values = reference["grid"][:, dimension - 2] <= chunk[:, None]
        result[start:start + 128] = values @ reference["weights"]
    return result
