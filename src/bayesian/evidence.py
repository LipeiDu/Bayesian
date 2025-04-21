# bayesian/evidence.py

import numpy as np
import scipy.stats
import logging

logger = logging.getLogger(__name__)


def compute_from_sampler(sampler, sampler_type: str, method: str = "harmonic_mean") -> tuple[float, float]:
    """
    Compute Bayesian evidence (logZ) and error using specified method.

    Args:
        sampler: emcee or pocoMC sampler.
        sampler_type: "emcee" or "pocoMC"
        method: estimation method: "harmonic_mean", "laplace", or "none"

    Returns:
        logZ, logZ_err
    """
    if method == "none":
        return np.nan, np.nan

    if sampler_type == "pocoMC":
        logZ, logZ_err = sampler.evidence()
        logger.info(f"[Evidence] pocoMC native estimate: logZ = {logZ:.3f} ± {logZ_err:.3f}")
        return logZ, logZ_err

    if sampler_type == "emcee":
        if method == "harmonic_mean":
            return _harmonic_mean(sampler)
        elif method == "laplace":
            return _laplace_approximation(sampler)
        else:
            raise ValueError(f"Unsupported evidence estimation method: {method}")

    raise ValueError(f"Unsupported sampler type: {sampler_type}")


def _harmonic_mean(sampler) -> tuple[float, float]:
    """
    Harmonic mean estimator for log evidence.
    Unstable but general-purpose.

    Returns:
        logZ, logZ_err
    """
    logL = sampler.get_log_prob(flat=True)
    max_logL = np.max(logL)

    try:
        weights = np.exp(-(logL - max_logL))
        logZ = max_logL - np.log(np.mean(weights))
        logZ_err = np.std(weights) / (np.mean(weights) * np.sqrt(len(weights)))
    except FloatingPointError as e:
        logger.warning(f"[Evidence] Floating-point error in harmonic mean estimator: {e}")
        logZ, logZ_err = float("-inf"), np.inf

    logger.info(f"[Evidence] Harmonic mean estimate: logZ = {logZ:.3f} ± {logZ_err:.3f}")
    return logZ, logZ_err


def _laplace_approximation(sampler) -> tuple[float, float]:
    """
    Laplace approximation to log evidence using posterior covariance at MAP.

    Returns:
        logZ, logZ_err
    """
    chain = sampler.get_chain(flat=True)
    logL = sampler.get_log_prob(flat=True)

    # Find MAP
    idx_max = np.argmax(logL)
    theta_MAP = chain[idx_max]
    logL_MAP = logL[idx_max]

    cov = np.cov(chain.T)
    ndim = chain.shape[1]

    try:
        log_det_cov = np.linalg.slogdet(cov)[1]
    except np.linalg.LinAlgError:
        logger.warning("[Evidence] Laplace approximation failed: singular covariance.")
        return float("-inf"), np.inf

    logZ = logL_MAP + 0.5 * ndim * np.log(2 * np.pi) + 0.5 * log_det_cov
    logZ_err = np.nan  # Not trivial to estimate from single MAP point

    logger.info(f"[Evidence] Laplace approximation: logZ = {logZ:.3f}")
    return logZ, logZ_err
