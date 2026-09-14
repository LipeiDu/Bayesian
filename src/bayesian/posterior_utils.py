"""Utilities shared by posterior consumers."""

import numpy as np
import numpy.typing as npt


def flatten_chain_samples(
    chain: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return 2D posterior samples from emcee or flat weighted samplers."""
    if chain.ndim == 3:
        return chain.reshape((chain.shape[0] * chain.shape[1], chain.shape[2]))
    if chain.ndim == 2:
        return chain
    raise ValueError(f"Unsupported chain shape {chain.shape}; expected 2D or 3D samples.")


def decompose_target_density(
    log_target: npt.NDArray[np.float64],
    log_prior_bayesian: npt.NDArray[np.float64],
    *,
    hard_likelihood_power: float = 1.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return explicit hard-likelihood and target components for output files."""
    log_target = np.asarray(log_target, dtype=float)
    log_prior_bayesian = np.asarray(log_prior_bayesian, dtype=float)
    if log_target.shape != log_prior_bayesian.shape:
        raise ValueError(
            "Target and Bayesian-prior arrays must have the same shape; "
            f"received {log_target.shape} and {log_prior_bayesian.shape}."
        )
    if hard_likelihood_power <= 0.0:
        raise ValueError(
            "Cannot reconstruct an untempered hard likelihood when "
            "hard_likelihood_power is zero."
        )
    return (log_target - log_prior_bayesian) / hard_likelihood_power, log_target


def uniform_bounds_to_loc_scale(
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert lower/upper bounds to scipy.stats.uniform loc/scale arrays."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("Lower and upper prior bounds must have the same shape.")
    scale = upper - lower
    if np.any(scale <= 0.0):
        raise ValueError("Every upper prior bound must be greater than its lower bound.")
    return lower, scale
