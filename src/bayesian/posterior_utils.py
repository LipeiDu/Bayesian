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
    message = f"Unsupported chain shape {chain.shape}; expected 2D or 3D samples."
    raise ValueError(message)


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
        message = f"Target and Bayesian-prior arrays must have the same shape; received {log_target.shape} and {log_prior_bayesian.shape}."
        raise ValueError(message)
    if hard_likelihood_power <= 0.0:
        message = "Cannot reconstruct an untempered hard likelihood when hard_likelihood_power is zero."
        raise ValueError(message)
    return (log_target - log_prior_bayesian) / hard_likelihood_power, log_target


def uniform_bounds_to_loc_scale(
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert lower/upper bounds to scipy.stats.uniform loc/scale arrays."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        message = "Lower and upper prior bounds must have the same shape."
        raise ValueError(message)
    scale = upper - lower
    if np.any(scale <= 0.0):
        message = "Every upper prior bound must be greater than its lower bound."
        raise ValueError(message)
    return lower, scale


def normalized_weights(weights, sample_count):
    """Validate and copy particle weights, scaling before summing to avoid overflow."""
    if sample_count <= 0:
        message = "Posterior samples must not be empty"
        raise ValueError(message)
    if weights is None:
        return np.full(sample_count, 1.0 / sample_count)
    values = np.array(weights, dtype=float, copy=True)
    if (values.shape != (sample_count,) or not np.all(np.isfinite(values))
            or np.any(values < 0) or not np.any(values > 0)):
        message = "Posterior weights must be finite, nonnegative, have equal shape to sample rows, and have positive mass"
        raise ValueError(message)
    values /= values.max()
    return values / values.sum()


def _weighted_atoms(values, weights):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        message = "Quantile values must be a finite one-dimensional array"
        raise ValueError(message)
    weights = normalized_weights(weights, len(values))
    positive = weights > 0
    atoms, inverse = np.unique(values[positive], return_inverse=True)
    masses = np.bincount(inverse, weights=weights[positive])
    return atoms, masses


def weighted_quantile(values, probabilities, weights=None):
    """Interpolate at atom CDF midpoints, pooling ties and excluding zero mass.

    This is the convention historically used by hard-likelihood diagnostics.
    Quantiles are invariant to splitting a particle into identical copies.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        message = "Quantile probabilities must be finite and lie in [0, 1]"
        raise ValueError(message)
    atoms, masses = _weighted_atoms(values, weights)
    cumulative = np.cumsum(masses) - 0.5 * masses
    return np.interp(probabilities, cumulative, atoms)


def shortest_interval(values, confidence, weights=None):
    """Smallest contiguous interval containing at least the requested particle mass."""
    atoms, masses = _weighted_atoms(values, weights)
    cumulative = np.cumsum(masses)
    cumulative[-1] = 1.0
    before = np.r_[0.0, cumulative[:-1]]
    targets = before + confidence
    starts = np.flatnonzero(targets <= 1.0 + 1e-14)
    ends = np.searchsorted(cumulative, np.minimum(targets[starts], 1.0))
    best = np.argmin(atoms[ends] - atoms[starts])
    return np.array([atoms[starts[best]], atoms[ends[best]]])


def select_posterior_samples(samples, weights=None, *, max_samples=None, random_state=None):
    """Keep small posteriors exact; approximate larger ones by seeded weighted draws."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or not np.all(np.isfinite(samples)):
        message = "Posterior samples must be a finite two-dimensional array"
        raise ValueError(message)
    mass = normalized_weights(weights, len(samples))
    if max_samples is None or len(samples) <= max_samples:
        return samples, mass
    if max_samples <= 0:
        message = "max_samples must be positive"
        raise ValueError(message)
    indices = np.random.default_rng(random_state).choice(len(samples), size=max_samples, replace=True, p=mass)
    return samples[indices], np.full(max_samples, 1.0 / max_samples)


def model_posterior_samples(results, parameter_info):
    """Project named sampled model columns, excluding nuisance columns and restoring fixed values."""
    samples = flatten_chain_samples(np.asarray(results.get("posterior_samples", results.get("chain")), dtype=float))
    names = [value.decode() if isinstance(value, bytes) else str(value)
             for value in results.get("parameter_names", parameter_info.sampled_names)]
    if len(names) != samples.shape[1] or len(set(names)) != len(names):
        message = "Posterior parameter_names must uniquely label each sampled column"
        raise ValueError(message)
    indices = [names.index(name) for name in parameter_info.sampled_names]
    model = parameter_info.expand_sampled_to_full(samples[:, indices], sampled_names=parameter_info.sampled_names)
    return model, normalized_weights(results.get("weights"), len(samples))


def maximum_posterior_sample(samples, log_target):
    """Highest-density saved sample; particle weights are not posterior densities."""
    samples, log_target = np.asarray(samples), np.asarray(log_target).reshape(-1)
    if log_target.shape != (len(samples),) or not np.all(np.isfinite(log_target)):
        message = "MAP selection requires one finite log_target value per posterior sample"
        raise ValueError(message)
    return samples[np.argmax(log_target)]
