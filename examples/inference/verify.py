"""Check saved example outputs against independent densities and analytic/quadrature posteriors."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np
import yaml
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, multivariate_normal, norm, truncnorm

mpl.use("Agg")
import matplotlib.pyplot as plt

from bayesian.sequential.config import resolve_sequential_inference_config
from bayesian.sequential.prior import load_soft_density_model
from examples.inference.models import (
    DATASETS,
    DISC_MATRIX,
    DISC_NOISE,
    ERROR,
    GROUP_INDICES,
    LINEAR_MATRIX,
    PRIOR_MEAN,
    PRIOR_STD,
    X,
    discrepancy_data,
    kernel,
)


def _read(path):
    with h5py.File(path) as stream:
        samples = stream["posterior_samples"][:]
        weights = stream["weights"][:] if "weights" in stream else np.ones(len(samples))
        return {
            "samples": samples,
            "weights": weights / weights.sum(),
            "names": list(stream["parameter_names"][:].astype(str)),
            "likelihood": stream["log_likelihood_hard"][:],
            "prior": stream["log_prior_bayesian"][:],
            "target": stream["log_target"][:],
        }


def _gaussian_update(covariance, data, matrix, prior_mean=PRIOR_MEAN, prior_cov=None):
    prior_cov = np.diag(PRIOR_STD**2) if prior_cov is None else prior_cov
    predictive = covariance + matrix @ prior_cov @ matrix.T
    gain = np.linalg.solve(predictive, matrix @ prior_cov).T
    return prior_mean + gain @ (data - matrix @ prior_mean), prior_cov - gain @ matrix @ prior_cov


def _likelihood(samples, matrix, data, covariance):
    residuals = samples @ matrix.T - data
    return -0.5 * (
        np.einsum("ij,ji->i", residuals, np.linalg.solve(covariance, residuals.T)) + np.linalg.slogdet(covariance)[1]
    )


def _check_fields(run, likelihood, prior, power=1.0):
    errors = {
        name: float(np.max(abs(actual - expected)))
        for name, actual, expected in [
            ("likelihood", run["likelihood"], likelihood),
            ("prior", run["prior"], prior),
            ("target", run["target"], power * likelihood + prior),
        ]
    }
    tolerance = 1e-4 if run.get("flow") else 2e-9
    if max(errors.values()) > tolerance:
        msg = f"Saved density fields disagree: {errors}"
        raise AssertionError(msg)
    return errors


def _moments(run, reference_mean, reference_cov):
    mean = run["weights"] @ run["samples"]
    centered = run["samples"] - mean
    covariance = np.einsum("i,ij,ik->jk", run["weights"], centered, centered)
    mean_error, cov_error = np.max(abs(mean - reference_mean)), np.max(abs(covariance - reference_cov))
    if mean_error > 0.12 or cov_error > 0.035:
        msg = f"Posterior differs from reference: mean {mean_error}, covariance {cov_error}"
        raise AssertionError(msg)
    return {
        "mean": mean.tolist(),
        "reference_mean": reference_mean.tolist(),
        "max_mean_error": float(mean_error),
        "max_covariance_error": float(cov_error),
        "draws": len(run["samples"]),
    }


def _cdf_plot(ax, run, dimension, grid, reference, label="Posterior"):
    cdf = (run["samples"][:, dimension, None] <= grid).T @ run["weights"]
    ax.plot(grid, reference, color="0.2", label="Independent reference")
    ax.plot(grid, cdf, color="#0072B2", label=label)
    ax.set(ylabel="Cumulative probability", ylim=(-0.02, 1.02))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    return float(np.max(abs(cdf - reference)))


def _fixed_or_inferred(config, analysis, run, output):
    inferred = config["example"]["kind"].endswith("inferred")
    groups = analysis["parameters"]["emulators"]
    mode = "inferred" if inferred else "fixed"
    data = discrepancy_data(mode, config["example"].get("data_seed", 418))

    def covariance(values=None):
        result = np.eye(len(X)) * DISC_NOISE
        for group, settings in groups.items():
            disc, indices = settings["discrepancy"], GROUP_INDICES[group]
            params = (
                dict(zip(disc["parameter_names"], values, strict=True))
                if disc["infer_hyperparameters"]
                else dict(
                    zip(disc.get("parameter_names", ["c_bar", "length_scale", "r"]), disc["fixed_params"], strict=True)
                )
            )
            if disc["kernel_type"] == "kernel2":
                params.pop("s", None)
            result[np.ix_(indices, indices)] += kernel(X[indices], **params)
        return result

    prior = norm.logpdf(run["samples"][:, :2], PRIOR_MEAN, PRIOR_STD).sum(axis=1)
    likelihood = np.array(
        [
            _likelihood(sample[None, :2], DISC_MATRIX[:, :2], data, covariance(sample[2:] if inferred else None))[0]
            for sample in run["samples"]
        ]
    )
    if inferred:
        prior += norm.logpdf(run["samples"][:, 3], 0.3, 0.18)
    fields = _check_fields(run, likelihood, prior)
    grid = np.zeros((1, 0))
    if inferred:
        disc = groups["a"]["discrepancy"]
        axes = [np.linspace(lo, hi, 36) for lo, hi in zip(disc["min"], disc["max"], strict=True)]
        centers = [(axis[1:] + axis[:-1]) / 2 for axis in axes]
        grid = np.column_stack([axis.ravel() for axis in np.meshgrid(*centers, indexing="ij")])
    conditional_means, conditional_covs, log_weights = [], [], []
    for values in grid:
        cov = covariance(values if inferred else None)
        mean, conditional = _gaussian_update(cov, data, DISC_MATRIX[:, :2])
        log_mass = multivariate_normal.logpdf(
            data,
            mean=DISC_MATRIX[:, :2] @ PRIOR_MEAN,
            cov=cov + DISC_MATRIX[:, :2] @ np.diag(PRIOR_STD**2) @ DISC_MATRIX[:, :2].T,
        )
        if inferred:
            log_mass += norm.logpdf(values[1], 0.3, 0.18)
        conditional_means.append(mean)
        conditional_covs.append(conditional)
        log_weights.append(log_mass)
    mass = np.exp(log_weights - logsumexp(log_weights))
    conditional_means, conditional_covs = np.asarray(conditional_means), np.asarray(conditional_covs)
    centers = np.column_stack([conditional_means, grid])
    mean = mass @ centers
    centered = centers - mean
    cov = np.einsum("i,ij,ik->jk", mass, centered, centered)
    cov[:2, :2] += np.einsum("i,ijk->jk", mass, conditional_covs)
    stats = _moments(run, mean, cov)
    n_dim = len(mean)
    fig, axes = plt.subplots(
        2 if inferred else 1,
        3 if inferred else 2,
        figsize=(11, 6.5 if inferred else 3.7),
        squeeze=False,
        layout="constrained",
    )
    errors = []
    for dimension, ax in enumerate(axes.flat):
        if dimension >= n_dim:
            ax.axis("off")
            continue
        points = np.linspace(
            mean[dimension] - 3 * np.sqrt(cov[dimension, dimension]),
            mean[dimension] + 3 * np.sqrt(cov[dimension, dimension]),
            101,
        )
        if dimension < 2:
            expected = (
                norm.cdf(
                    (points[:, None] - conditional_means[:, dimension])
                    / np.sqrt(conditional_covs[:, dimension, dimension])
                )
                @ mass
            )
        else:
            expected = (grid[:, dimension - 2] <= points[:, None]) @ mass
        errors.append(_cdf_plot(ax, run, dimension, points, expected))
        ax.set_xlabel(run["names"][dimension])
    if max(errors) > 0.17:
        msg = f"Discrepancy marginal CDF error {max(errors)}"
        raise AssertionError(msg)
    fig.savefig(output / "validation.png", dpi=150)
    plt.close(fig)
    return {
        **stats,
        "density_errors": fields,
        "marginal_cdf_errors": errors,
        "reference": "analytic Gaussian"
        if not inferred
        else "35^3 midpoint quadrature with analytic model conditioning",
    }


def _sequential(_config, analysis, run, output):
    settings = analysis["parameters"]["mcmc"]
    sequential = resolve_sequential_inference_config(
        mcmc_configuration=settings,
        config_file=output / "resolved.yaml",
        analysis_config=analysis,
        parameterization="exponential",
    )
    run["flow"] = sequential.soft_density_model_type == "torch_realnvp_2d"
    mean = np.column_stack([run["samples"][:, 0], run["samples"][:, 1] + 0.03 * (run["samples"][:, 2] - 15)])
    error = np.sqrt(np.array([0.025, 0.15]) ** 2 + np.array([0.015, 0.03]) ** 2)
    likelihood = -0.5 * np.sum(((mean - [0.32, 1.08]) / error) ** 2, axis=1) - np.log(error).sum()
    soft = load_soft_density_model(sequential_config=sequential)
    prior = soft(run["samples"][:, 1:])
    fields = _check_fields(run, likelihood, prior, sequential.hard_likelihood_power)
    power = sequential.hard_likelihood_power
    sigma = error[0] / np.sqrt(power) if power else None
    points = np.linspace(0.1, 0.5, 201)
    expected = (
        truncnorm.cdf(points, (0.1 - 0.32) / sigma, (0.5 - 0.32) / sigma, loc=0.32, scale=sigma)
        if power
        else (points - 0.1) / 0.4
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    distance = _cdf_plot(axes[0], run, 0, points, expected)
    axes[0].set_xlabel("AlphaS")
    axes[1].hist2d(run["samples"][:, 1], run["samples"][:, 2], weights=run["weights"], bins=30, cmap="Blues")
    axes[1].set(xlabel="Nucleon width", ylabel="Normalization", title="Weighted medium posterior")
    fig.savefig(output / "validation.png", dpi=150)
    plt.close(fig)
    if distance > 0.17:
        msg = f"Sequential coupling CDF error {distance}"
        raise AssertionError(msg)
    return {
        "density_errors": fields,
        "coupling_cdf_error": distance,
        "draws": len(run["samples"]),
        "reference": "analytic truncated-Gaussian coupling marginal; soft-density loader also covered by independent payload regression tests",
    }


def _multistep(config, runs, output):
    matrix = LINEAR_MATRIX[:, :2]
    noise = np.diag(ERROR**2)
    first = runs["first"]
    source = config["inference_workflows"]["two_step"]["prior_sources"]["second"]["method"]
    if source == "gaussian":
        centers = (first["weights"] @ first["samples"])[None, :]
        prior_cov = np.cov(first["samples"].T, aweights=first["weights"])
        mass = np.ones(1)
        density = multivariate_normal(centers[0], prior_cov)
        evaluate = density.logpdf
    else:
        density = gaussian_kde(first["samples"].T, weights=first["weights"])
        centers, prior_cov, mass = first["samples"], density.covariance, first["weights"]

        def evaluate(points):
            return density.logpdf(points.T)

    updated = [_gaussian_update(noise, DATASETS["B"], matrix, center, prior_cov) for center in centers]
    means, covs = np.array([v[0] for v in updated]), np.array([v[1] for v in updated])
    log_weights = np.log(mass) + multivariate_normal.logpdf(
        DATASETS["B"] - centers @ matrix.T, cov=noise + matrix @ prior_cov @ matrix.T
    )
    weights = np.exp(log_weights - logsumexp(log_weights))
    mean = weights @ means
    centered = means - mean
    covariance = np.einsum("i,ijk->jk", weights, covs) + np.einsum("i,ij,ik->jk", weights, centered, centered)
    two_step = runs["two_step"]
    stats = _moments(two_step, mean, covariance)
    fields = _check_fields(
        two_step, _likelihood(two_step["samples"], matrix, DATASETS["B"], noise), evaluate(two_step["samples"])
    )
    joint_mean, joint_cov = _gaussian_update(
        np.kron(np.eye(2), noise), np.r_[DATASETS["A"], DATASETS["B"]], np.vstack([matrix, matrix])
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    errors = []
    for dim, ax in enumerate(axes):
        points = np.linspace(
            joint_mean[dim] - 4 * np.sqrt(joint_cov[dim, dim]), joint_mean[dim] + 4 * np.sqrt(joint_cov[dim, dim]), 160
        )
        expected = norm.cdf((points[:, None] - means[:, dim]) / np.sqrt(covs[:, dim, dim])) @ weights
        errors.append(_cdf_plot(ax, two_step, dim, points, expected, "Two-step samples"))
        ax.plot(
            points,
            norm.cdf(points, joint_mean[dim], np.sqrt(joint_cov[dim, dim])),
            "--",
            color="#D55E00",
            label="Original-prior analytic joint",
        )
        ax.set_xlabel(two_step["names"][dim])
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(output / "validation.png", dpi=150)
    plt.close(fig)
    if max(errors) > 0.17:
        msg = f"Multistep CDF error {max(errors)}"
        raise AssertionError(msg)
    return {
        **stats,
        "density_errors": fields,
        "marginal_cdf_errors": errors,
        "fitted_prior_mean_difference_from_joint": (mean - joint_mean).tolist(),
        "joint_sampler": _moments(runs["joint"], joint_mean, joint_cov),
        "reference": "exact fitted-prior Gaussian/mixture update; original-prior joint shown separately",
    }


def _baseline(run, output):
    matrix = LINEAR_MATRIX[:, :2]
    noise = np.diag(ERROR**2)
    mean, covariance = _gaussian_update(noise, DATASETS["A"], matrix)
    stats = _moments(run, mean, covariance)
    fields = _check_fields(run, _likelihood(run["samples"], matrix, DATASETS["A"], noise),
                           norm.logpdf(run["samples"], PRIOR_MEAN, PRIOR_STD).sum(axis=1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    errors = []
    for dim, ax in enumerate(axes):
        points = np.linspace(mean[dim] - 4*np.sqrt(covariance[dim, dim]),
                             mean[dim] + 4*np.sqrt(covariance[dim, dim]), 160)
        errors.append(_cdf_plot(ax, run, dim, points,
                               norm.cdf(points, mean[dim], np.sqrt(covariance[dim, dim]))))
        ax.set_xlabel(run["names"][dim])
        ax.legend(fontsize=8)
    fig.savefig(output / "validation.png", dpi=150)
    plt.close(fig)
    if max(errors) > 0.17:
        raise AssertionError(f"Baseline CDF error {max(errors)}")
    return {**stats, "density_errors": fields, "marginal_cdf_errors": errors,
            "reference": "analytic Gaussian"}


def verify(output_dir):
    output = Path(output_dir).resolve()
    config = yaml.safe_load((output / "resolved.yaml").read_text())
    runs = {name: _read(output / f"{name}_exponential/posterior.h5") for name in config["analyses"]}
    kind = config["example"]["kind"]
    with plt.rc_context(
        {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9}
    ):
        if kind.startswith("discrepancy"):
            result = _fixed_or_inferred(config, config["analyses"]["demo"], runs["demo"], output)
        elif kind == "baseline":
            result = _baseline(runs["demo"], output)
        elif kind == "sequential":
            result = _sequential(config, config["analyses"]["demo"], runs["demo"], output)
        else:
            runs["two_step"] = _read(output / "two_step_first_second/posterior.h5")
            result = _multistep(config, runs, output)
    result.update(
        status="passed",
        example_kind=kind,
        samplers={
            name: analysis["parameters"]["mcmc"]["mcmc_package"] for name, analysis in config["analyses"].items()
        },
    )
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir), indent=2))  # noqa: T201 - CLI output


if __name__ == "__main__":
    main()
