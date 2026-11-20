"""Evidence estimates with explicit target measure and normalization bookkeeping."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm

from bayesian.prior import normalize_prior_config


def prior_log_normalization(prior_config, lower, upper):
    """Log integral of the manual Bayesian prior over the hard sampling box.

    Uniform and log priors in this backend omit their normalizers. Gaussian
    priors include their real-line normalization but still require truncation.
    Derived joint priors require a separate box-mass calculation.
    """
    if prior_config and "prior_source" in prior_config:
        return None
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.shape != upper.shape or lower.ndim != 1 or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper)) or np.any(upper <= lower)):
        message = "Evidence requires finite, ordered parameter bounds"
        raise ValueError(message)
    config = normalize_prior_config(prior_config, len(lower))
    total = 0.0
    for i, kind in enumerate(config["type"]):
        if kind == "uniform":
            total += np.log(upper[i] - lower[i])
        elif kind == "log":
            if lower[i] <= 0:
                message = "Log-prior evidence requires strictly positive bounds"
                raise ValueError(message)
            total += np.log(np.log(upper[i]) - np.log(lower[i]))
        elif kind == "gaussian":
            mean, std = config["mean"][i], config["std"][i]
            if mean is None or std is None or not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
                message = "Gaussian prior normalization requires finite mean and positive std"
                raise ValueError(message)
            a, b = (lower[i]-mean)/std, (upper[i]-mean)/std
            # Use survival probabilities on the positive side to avoid CDF cancellation.
            log_large, log_small = (log_ndtr(-a), log_ndtr(-b)) if a >= 0 else (log_ndtr(b), log_ndtr(a))
            total += log_large + np.log(-np.expm1(log_small-log_large))
        else:
            message = f"Evidence normalization is unavailable for prior type {kind}"
            raise ValueError(message)
    if not np.isfinite(total):
        message = "Prior has no numerically resolvable positive mass in the sampling box"
        raise ValueError(message)
    return float(total)


def _laplace_target_integral(sampler, log_target_fn, lower, upper):
    """Optimize the joint target and use its local Hessian in unit-box coordinates."""
    if log_target_fn is None:
        message = "Laplace evidence requires the actual log_target_fn, not only posterior samples"
        raise ValueError(message)
    samples = np.asarray(sampler.get_chain(flat=True), dtype=float)
    values = np.asarray(sampler.get_log_prob(flat=True), dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    widths = upper-lower
    start = (samples[np.argmax(values)]-lower)/widths

    def objective(unit):
        result = np.asarray(log_target_fn((lower + unit*widths)[None, :])).reshape(-1)[0]
        return -float(result)

    mode = minimize(objective, np.clip(start, 1e-8, 1-1e-8), method="L-BFGS-B",
                    bounds=[(1e-8, 1-1e-8)]*len(lower), options={"ftol": 1e-12, "gtol": 1e-6, "maxiter": 1000})
    if not mode.success or not np.isfinite(mode.fun):
        message = f"Laplace target optimization failed: {mode.message}"
        raise ValueError(message)
    steps = np.eye(len(lower))*1e-4
    if np.any(mode.x <= 2e-4) or np.any(mode.x >= 1-2e-4):
        message = "Laplace mode is at a sampling boundary; use native pocoMC evidence"
        raise ValueError(message)
    hessian = np.empty((len(lower), len(lower)))
    for i, si in enumerate(steps):
        hessian[i, i] = (objective(mode.x+si) - 2*mode.fun + objective(mode.x-si))/1e-8
        for j in range(i):
            sj = steps[j]
            hessian[i, j] = hessian[j, i] = (
                objective(mode.x+si+sj)-objective(mode.x+si-sj)
                -objective(mode.x-si+sj)+objective(mode.x-si-sj)
            )/4e-8
    try:
        factor = np.linalg.cholesky(hessian)
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError as error:
        message = "Laplace Hessian must be positive definite"
        raise ValueError(message) from error
    std = np.sqrt(np.diag(covariance))
    outside = np.sum(norm.cdf(-mode.x/std) + norm.sf((1-mode.x)/std))
    if outside > 1e-4:
        message = "Laplace Gaussian has appreciable mass outside the sampling box; use native pocoMC evidence"
        raise ValueError(message)
    log_integral = -mode.fun + 0.5*len(lower)*np.log(2*np.pi) - np.log(np.diag(factor)).sum() + np.log(widths).sum()
    return float(log_integral), np.nan, {"mode": lower+mode.x*widths, "box_tail_upper_bound": float(outside),
                                         "approximation": "local_hessian_laplace"}


def compute_from_sampler(sampler, sampler_type, method=None, *, lower=None, upper=None, log_target_fn=None,
                         prior_config=None, n_observables=None, sequential_config=None):
    """Return normalized (logZ, error) only with complete density context.

    Use evidence_record to retain a separately labeled target integral when a
    derived prior's box normalization has not been established.
    """
    if method == "none":
        return np.nan, np.nan
    if lower is None or upper is None or n_observables is None:
        message = "Evidence requires explicit parameter bounds and density normalization context"
        raise ValueError(message)
    record = evidence_record(sampler, sampler_type, method, lower=lower, upper=upper,
                             log_target_fn=log_target_fn, prior_config=prior_config,
                             n_observables=n_observables, sequential_config=sequential_config)
    if "logZ" not in record:
        message = "Absolute evidence normalization is unavailable; inspect evidence_record instead"
        raise ValueError(message)
    return record["logZ"], record["logZ_err"]


def target_integral(sampler, sampler_type, method=None, *, lower=None, upper=None, log_target_fn=None):
    if lower is None or upper is None:
        message = "Evidence requires explicit parameter bounds and density normalization context"
        raise ValueError(message)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.ndim != 1 or lower.shape != upper.shape or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper)) or np.any(upper <= lower)):
        message = "Evidence requires finite, ordered parameter bounds"
        raise ValueError(message)
    method = method or ("native" if sampler_type == "pocoMC" else "laplace")
    if method == "none":
        return np.nan, np.nan, {"method": "none"}
    if method == "harmonic_mean":
        message = "harmonic_mean evidence is retired: log posterior is not log likelihood and this estimator is unstable"
        raise ValueError(message)
    if sampler_type == "pocoMC" and method == "native":
        log_z, error = sampler.evidence()
        # pocoMC integrates target * normalized uniform reference density.
        volume = np.log(np.asarray(upper)-np.asarray(lower)).sum()
        return float(log_z+volume), float(error) if error is not None else np.nan, {"method": "native_pocoMC"}
    if sampler_type == "emcee" and method == "laplace":
        value, error, metadata = _laplace_target_integral(sampler, log_target_fn, lower, upper)
        return value, error, {"method": "laplace", **metadata}
    message = f"Unsupported evidence method {method} for {sampler_type}"
    raise ValueError(message)


def evidence_record(sampler, sampler_type, method, *, lower, upper, prior_config, n_observables,
                    log_target_fn=None, sequential_config=None):
    """Save an auditable target integral, adding logZ only when normalization is known."""
    if method == "none":
        return {"status": "disabled", "method": "none"}
    integral, error, metadata = target_integral(sampler, sampler_type, method, lower=lower, upper=upper,
                                               log_target_fn=log_target_fn)
    prior_mass = None if sequential_config is not None else prior_log_normalization(prior_config, lower, upper)
    power = sequential_config.hard_likelihood_power if sequential_config is not None else 1.0
    likelihood_constant = -0.5*power*n_observables*np.log(2*np.pi)
    record = {"status": "normalized" if prior_mass is not None else "target_integral_only", **metadata,
              "log_target_integral": integral, "log_target_integral_err": error,
              "target_measure": "physical_parameter_lebesgue", "hard_likelihood_power": power,
              "likelihood_log_constant": likelihood_constant}
    if prior_mass is None:
        record["reason"] = "Derived prior box normalization has not been established; logZ is not reported"
    else:
        record.update(prior_log_normalization=prior_mass, logZ=integral+likelihood_constant-prior_mass, logZ_err=error)
    return record
