"""Manual priors and named, weighted posterior density approximations."""

import logging
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import gaussian_kde, multivariate_normal, norm

logger = logging.getLogger(__name__)


def normalize_prior_config(prior_config: dict | None, ndim: int) -> dict:
    """Copy a prior specification, filling omitted per-parameter defaults."""
    config = deepcopy(prior_config) if prior_config is not None else {}
    for key, default in (("type", "uniform"), ("mean", None), ("std", None)):
        values = list(config.get(key, []))
        if len(values) > ndim:
            message = f"Prior {key} has {len(values)} entries for {ndim} parameters"
            raise ValueError(message)
        config[key] = values + [default] * (ndim - len(values))
    return config


def make_log_prior_fn(prior_config: dict | None, param_names: list[str] | None = None) -> Callable:
    """Use the source density once, plus manual priors on independent new parameters.

    ``prior_source.parameter_names`` selects source-covered target parameters.
    By default the source must cover every target parameter. Extra source columns
    are marginalized; source columns are matched by name and reordered.
    """
    if prior_config is None:
        return lambda X: np.zeros(np.atleast_2d(X).shape[0])
    if "prior_source" not in prior_config:
        ndim = len(param_names) if param_names is not None else len(prior_config.get("type", []))
        config = normalize_prior_config(prior_config, ndim)
        return lambda X: log_prior(X, config)

    source = prior_config["prior_source"]
    covered = source.get("parameter_names", param_names)
    loaded = load_posterior_as_prior(source["posterior_file"], source.get("method", "kde"), param_names=covered)
    if param_names is None or loaded.parameter_names == param_names:
        return loaded
    if not set(loaded.parameter_names).issubset(param_names):
        message = "prior_source.parameter_names must be a subset of sampled parameter names"
        raise ValueError(message)
    indices = [param_names.index(name) for name in loaded.parameter_names]
    manual = normalize_prior_config(prior_config, len(param_names))
    for index in indices:
        # The first-stage posterior already contains the original prior.
        manual["type"][index] = "uniform"

    def combined(X):
        X = np.array(X, dtype=float, ndmin=2)
        if X.ndim != 2 or X.shape[1] != len(param_names):
            message = f"Expected samples with {len(param_names)} columns, received {X.shape}"
            raise ValueError(message)
        return loaded(X[:, indices]) + log_prior(X, manual)

    return combined


def log_prior(X: np.ndarray, prior_config: dict | None) -> np.ndarray:
    """Evaluate independent manual priors; hard bounds are applied by the target."""
    X = np.array(X, dtype=float, ndmin=2)
    config = normalize_prior_config(prior_config, X.shape[1])
    logp = np.zeros(X.shape[0])
    for i, kind in enumerate(config["type"]):
        if kind == "uniform":
            continue
        if kind == "log":
            with np.errstate(divide="ignore", invalid="ignore"):
                logp += np.where(X[:, i] > 0, -np.log(X[:, i]), -np.inf)
        elif kind == "gaussian":
            mu, sigma = config["mean"][i], config["std"][i]
            if mu is None or sigma is None or not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
                message = f"Missing or invalid mean/std for gaussian prior on parameter {i}"
                raise ValueError(message)
            logp += norm.logpdf(X[:, i], loc=mu, scale=sigma)
        else:
            message = f"Prior type '{kind}' is not implemented"
            raise NotImplementedError(message)
    return logp


class PosteriorPrior:
    """A fitted density with consistently shaped draws and exact fitted marginals."""

    def __init__(self, samples, weights, parameter_names, method, bandwidth=None):
        self.samples = samples
        self.weights = weights
        self.parameter_names = list(parameter_names)
        self.method = method
        self.ndim = samples.shape[1]
        if method == "kde":
            self.density = gaussian_kde(samples.T, weights=weights, bw_method=bandwidth)
        elif method == "gaussian":
            self.density = multivariate_normal(
                mean=np.average(samples, axis=0, weights=weights),
                cov=np.atleast_2d(np.cov(samples.T, aweights=weights)),
            )
        else:
            message = f"Unknown method for loading posterior: {method}"
            raise ValueError(message)

    def __call__(self, X):
        X = np.array(X, dtype=float, ndmin=2)
        if X.ndim != 2 or X.shape[1] != self.ndim:
            message = f"Expected samples with {self.ndim} columns, received {X.shape}"
            raise ValueError(message)
        finite = np.all(np.isfinite(X), axis=1)
        result = np.full(len(X), -np.inf)
        if finite.any():
            points = X[finite].T if self.method == "kde" else X[finite]
            result[finite] = self.density.logpdf(points)
        return result

    def sample(self, size, random_state=None):
        if self.method == "kde":
            return self.density.resample(size=size, seed=random_state).T
        return np.asarray(self.density.rvs(size=size, random_state=random_state)).reshape(size, self.ndim)

    def marginal(self, parameter_names):
        indices = [self.parameter_names.index(name) for name in parameter_names]
        # Preserve the fitted joint KDE covariance's selected submatrix. Refitting
        # with the default lower-dimensional bandwidth would change the density.
        bandwidth = self.density.factor if self.method == "kde" else None
        return PosteriorPrior(self.samples[:, indices], self.weights, parameter_names, self.method, bandwidth)


def _read_posterior(posterior_file):
    with h5py.File(posterior_file, "r") as stream:
        samples = np.asarray(stream["posterior_samples"][:], dtype=float)
        weights = np.asarray(stream["weights"][:], dtype=float) if "weights" in stream else None
        names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in stream["parameter_names"][:]]
    if samples.ndim != 2 or samples.shape[0] < 2 or not np.all(np.isfinite(samples)):
        message = "posterior_samples must be a finite two-dimensional array with at least two rows"
        raise ValueError(message)
    if len(names) != samples.shape[1] or len(set(names)) != len(names) or any(not name for name in names):
        message = "parameter_names must contain one unique nonempty label per posterior column"
        raise ValueError(message)
    if weights is not None:
        if (weights.shape != (len(samples),) or not np.all(np.isfinite(weights))
                or np.any(weights < 0) or np.count_nonzero(weights > 0) < 2):
            message = "Posterior weights must be finite, nonnegative, match sample rows, and have at least two positive entries"
            raise ValueError(message)
        weights = weights / weights.max()
        weights /= weights.sum()
    return samples, weights, names


def load_posterior_as_prior(posterior_file: Path, method: str = "kde", param_names=None) -> PosteriorPrior:
    """Fit a weighted source marginal in requested target parameter order."""
    samples, weights, names = _read_posterior(posterior_file)
    requested = list(param_names) if param_names is not None else names
    missing = set(requested) - set(names)
    if not requested or len(set(requested)) != len(requested) or missing:
        message = f"Invalid requested parameter_names; missing source names: {sorted(missing)}"
        raise ValueError(message)
    logger.info("Fitting %s prior from %s for %s", method, posterior_file, requested)
    indices = [names.index(name) for name in requested]
    return PosteriorPrior(samples[:, indices], weights, requested, method)


def verify_prior_vs_posterior(posterior_file, method="kde", save_plot_to=None, n_points_plot=80, param_names=None):
    """Compare weighted source histograms with the fitted density's marginals."""
    import matplotlib.pyplot as plt  # noqa: PLC0415 - plotting is optional

    prior = load_posterior_as_prior(posterior_file, method, param_names=param_names)
    samples, weights, names = prior.samples, prior.weights, prior.parameter_names
    ndim = len(names)
    fig, axes = plt.subplots(ndim, ndim, figsize=(2.8 * ndim, 2.8 * ndim), squeeze=False)
    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            xx = np.linspace(samples[:, j].min(), samples[:, j].max(), n_points_plot)
            if i == j:
                ax.hist(samples[:, i], bins=30, weights=weights, density=True, alpha=0.4, label="Source posterior")
                ax.plot(xx, np.exp(prior.marginal([names[i]])(xx[:, None])), "r--", label="Fitted marginal")
                ax.legend(fontsize=6)
            else:
                yy = np.linspace(samples[:, i].min(), samples[:, i].max(), n_points_plot)
                grid_x, grid_y = np.meshgrid(xx, yy)
                points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
                density = np.exp(prior.marginal([names[j], names[i]])(points)).reshape(grid_x.shape)
                ax.hist2d(samples[:, j], samples[:, i], bins=30, weights=weights, cmap="Blues")
                ax.contour(grid_x, grid_y, density, colors="darkred", linewidths=1)
                ax.set_ylabel(names[i], fontsize=8)
            ax.set_xlabel(names[j], fontsize=8)
    fig.tight_layout()
    if save_plot_to:
        fig.savefig(save_plot_to)
    else:
        plt.show()
    plt.close(fig)


def sample_prior(prior_config, param_names, lower, upper, size, random_state=None):
    """Draw the configured bounded prior for diagnostics, including source-covered subsets."""
    from scipy.stats import truncnorm  # noqa: PLC0415

    rng = np.random.default_rng(random_state)
    lower, upper = np.asarray(lower), np.asarray(upper)
    config = normalize_prior_config(prior_config, len(param_names))
    samples = np.empty((size, len(param_names)))
    covered = []
    if "prior_source" in config:
        source = config["prior_source"]
        names = source.get("parameter_names", param_names)
        covered = [param_names.index(name) for name in names]
        fitted = load_posterior_as_prior(source["posterior_file"], source.get("method", "kde"), param_names=names)
        draws = fitted.sample(size, random_state=rng)
        for _ in range(100):
            valid = np.all((draws > lower[covered]) & (draws < upper[covered]) & np.isfinite(draws), axis=1)
            if valid.all():
                break
            draws[~valid] = fitted.sample(np.count_nonzero(~valid), random_state=rng)
        else:
            message = "Could not draw source prior inside diagnostic bounds"
            raise ValueError(message)
        samples[:, covered] = draws
    for index, kind in enumerate(config["type"]):
        if index in covered:
            continue
        lo, hi = lower[index], upper[index]
        if kind == "uniform":
            samples[:, index] = rng.uniform(lo, hi, size)
        elif kind == "log" and lo > 0:
            samples[:, index] = np.exp(rng.uniform(np.log(lo), np.log(hi), size))
        elif kind == "gaussian":
            mean, std = config["mean"][index], config["std"][index]
            samples[:, index] = truncnorm.rvs((lo-mean)/std, (hi-mean)/std, loc=mean, scale=std,
                                              size=size, random_state=rng)
        else:
            message = f"Cannot draw diagnostic prior type {kind} in bounds [{lo}, {hi}]"
            raise ValueError(message)
    return np.clip(samples, np.nextafter(lower.astype(float), upper), np.nextafter(upper.astype(float), lower))
