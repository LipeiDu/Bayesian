import numpy as np
import logging
import h5py
from pathlib import Path
from typing import Callable
from scipy.stats import norm, gaussian_kde
import numpy.typing as npt
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

logger = logging.getLogger(__name__)

########################################################################################################
def log_prior(X: npt.NDArray[np.float64], prior_config: dict | None) -> npt.NDArray[np.float64]:
    """
    Compute the log prior probability for each sample in X.

    Args:
        X (np.ndarray): Samples (n_samples, n_parameters)
        prior_config (dict): Prior configuration from YAML (can be None)

    Returns:
        np.ndarray: Log prior probability for each sample
    """
    X = np.array(X, ndmin=2)
    logp = np.zeros(X.shape[0]) # log(prior) for each sample

    # Default: uniform prior inside bounds
    if prior_config is None:
        return logp

    # retrieve keys from prior config dictionary
    prior_type = prior_config.get("type", ["uniform"] * X.shape[1])
    prior_mean = prior_config.get("mean", [None] * X.shape[1])
    prior_std = prior_config.get("std", [None] * X.shape[1])

    # each parameter can have its own prior type
    for i, kind in enumerate(prior_type):
        xi = X[:, i]
        if kind == "uniform":
            continue  # Uniform log-prob is constant
        elif kind == "log":
            logp += -np.log(xi)
        elif kind == "gaussian":
            mu = prior_mean[i]
            sigma = prior_std[i]
            if mu is None or sigma is None:
                raise ValueError(f"Missing mean/std for gaussian prior on parameter {i}")
            logp += norm(loc=mu, scale=sigma).logpdf(xi)
        else:
            raise NotImplementedError(f"Prior type '{kind}' is not implemented")

    return logp

########################################################################################################
def load_posterior_as_prior(posterior_file: Path, method: str = "kde") -> Callable[[np.ndarray], np.ndarray]:
    """
    Load posterior samples and return a log-prior function.

    Args:
        posterior_file (Path): Path to 'posterior.h5'
        method (str): Estimation method: 'kde' | 'gaussian' (default: kde)

    Returns:
        log_prior_fn (Callable): Function to compute log-prior for input samples (n_samples, n_parameters)
    """
    logger.info(f"Loading posterior samples from: {posterior_file}")

    with h5py.File(posterior_file, "r") as f:
        samples = f["posterior_samples"][:]
        weights = f["weights"][:] if "weights" in f else None
        parameter_names = [x.decode("utf-8") for x in f["parameter_names"][:]]

    if weights is not None:
        weights /= np.sum(weights)

    if method == "kde":
        logger.info("Fitting KDE to posterior samples...")
        kde = gaussian_kde(samples.T, weights=weights)
        def _logp(X):
            X = np.array(X, ndmin=2)
            return np.log(kde(X.T))
        _logp.sample = lambda size: kde.resample(size=size).T
        return _logp

    elif method == "gaussian":
        logger.info("Fitting multivariate Gaussian to posterior samples...")
        mu = np.average(samples, axis=0, weights=weights)
        cov = np.cov(samples.T, aweights=weights)
        cov_inv = np.linalg.inv(cov)
        norm_const = -0.5 * (np.log(np.linalg.det(cov)) + len(mu) * np.log(2 * np.pi))

        def _logp(X):
            X = np.array(X, ndmin=2)
            delta = X - mu
            return norm_const - 0.5 * np.sum(delta @ cov_inv * delta, axis=1)
        mvn = multivariate_normal(mean=mu, cov=cov)
        _logp.sample = lambda size: mvn.rvs(size=size)
        return _logp

    else:
        raise ValueError(f"Unknown method for loading posterior: {method}")

########################################################################################################
def verify_prior_vs_posterior(
    posterior_file, method="kde", save_plot_to=None, n_points_plot=80
):
    """
    Compare posterior samples with loaded prior as corner-style heatmap + contour plot.
    """
    logger.info(f"Verifying prior vs posterior: {posterior_file}")

    # Load posterior samples
    with h5py.File(posterior_file, "r") as f:
        samples = f["posterior_samples"][:]
        parameter_names = [x.decode("utf-8") for x in f["parameter_names"][:]]

    n_dim = samples.shape[1]
    logger.info(f"Posterior samples shape: {samples.shape}, number of parameters: {n_dim}")

    # Load prior
    from bayesian.prior import load_posterior_as_prior
    log_prior_fn = load_posterior_as_prior(posterior_file, method=method)

    fig, axes = plt.subplots(n_dim, n_dim, figsize=(2.5 * n_dim, 2.5 * n_dim), squeeze=False)

    for i, j in itertools.product(range(n_dim), repeat=2):
        ax = axes[i, j]

        if j > i:
            ax.axis('off')
            continue

        if i == j:
            # Diagonal: 1D marginal
            x = samples[:, i]
            sns.kdeplot(x, fill=True, color="steelblue", label="Posterior", ax=ax)

            # Evaluate prior marginal
            xmin, xmax = np.percentile(x, [1, 99])
            xx = np.linspace(xmin, xmax, 200)
            logp_prior = log_prior_fn(np.column_stack([xx if k==i else np.full_like(xx, np.mean(samples[:,k])) for k in range(n_dim)]))
            p_prior = np.exp(logp_prior)
            p_prior /= np.trapz(p_prior, xx)  # normalize for comparison

            ax.plot(xx, p_prior, color="darkred", linestyle="--", label="Prior approx")
            ax.set_xlabel(parameter_names[i], fontsize=8)
            ax.legend(fontsize=6)

        else:
            # Off-diagonal: 2D
            x = samples[:, j]
            y = samples[:, i]

            # 2D histogram for posterior samples
            ax.hist2d(x, y, bins=80, cmap="Blues", cmin=1)

            # Prior contour
            xmin, xmax = np.percentile(x, [1, 99])
            ymin, ymax = np.percentile(y, [1, 99])

            X_grid, Y_grid = np.meshgrid(
                np.linspace(xmin, xmax, n_points_plot),
                np.linspace(ymin, ymax, n_points_plot)
            )
            XY = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)

            # Fill other dimensions by sample means
            full_XY = np.zeros((XY.shape[0], n_dim))
            full_XY[:, j] = XY[:,0]
            full_XY[:, i] = XY[:,1]
            for k in range(n_dim):
                if k != i and k != j:
                    full_XY[:,k] = np.mean(samples[:,k])

            Z_prior = np.exp(log_prior_fn(full_XY))
            Z_prior = Z_prior.reshape(n_points_plot, n_points_plot)

            ax.contour(X_grid, Y_grid, Z_prior, colors="darkred", linewidths=1)

            ax.set_xlabel(parameter_names[j], fontsize=8)
            ax.set_ylabel(parameter_names[i], fontsize=8)

    plt.tight_layout()
    if save_plot_to:
        logger.info(f"Saving prior vs posterior plot to {save_plot_to}")
        plt.savefig(save_plot_to)
    else:
        plt.show()

    plt.close(fig)

