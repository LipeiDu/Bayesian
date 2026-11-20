"""Analytic emulator and frozen synthetic inputs used by the public examples."""

import numpy as np

LINEAR_MATRIX = np.array([[1.0, 0.4, 0.0], [-0.3, 1.0, 0.0]])
DATASETS = {"A": np.array([0.35, 1.0]), "B": np.array([-0.15, 0.75])}
ERROR = np.sqrt(np.array([0.2, 0.15]) ** 2 + np.array([0.04, 0.03]) ** 2)
PRIOR_MEAN = np.array([0.1, 0.8])
PRIOR_STD = np.array([0.5, 0.4])
X = np.array([1.0, 1.3, 1.6, 1.9, 1.1, 2.0, 3.0, 2.2, 2.5, 2.8, 3.1, 3.4])
DISC_MATRIX = np.column_stack([np.sin(1.8 * X), np.cos(0.6 * X), np.zeros(len(X))])
GROUP_INDICES = {"a": np.array([0, 1, 2, 3, 7, 8, 9, 10, 11]), "b": np.array([4, 5, 6])}
TRUTH = np.array([0.25, 0.9, 14.0])
TRUE_HYPER = {"c_bar": 0.3, "length_scale": 0.7, "r": -0.5, "s": 0.07}
DISC_ORDER = ["r", "c_bar", "length_scale"]
DISC_NOISE = 0.07**2 + 0.03**2


class LinearEmulator:
    """Implement the backend's emulator prediction protocol exactly."""

    def __init__(self, coefficients, std, offset=0.0, fixed_normalization=None):
        self.coefficients, self.std = np.asarray(coefficients), np.asarray(std)
        self.offset, self.fixed_normalization = np.asarray(offset), fixed_normalization

    def predict(self, parameters, return_std):
        if not return_std or parameters.shape[1] != 3:
            msg = "The example emulator expects three full model columns and return_std=True"
            raise ValueError(msg)
        if self.fixed_normalization is not None and not np.all(parameters[:, 2] == self.fixed_normalization):
            msg = "Fixed normalization did not reach the example emulator"
            raise ValueError(msg)
        return parameters @ self.coefficients.T + self.offset, np.broadcast_to(
            self.std, (len(parameters), len(self.coefficients))
        )


def kernel(x, c_bar, length_scale, r, s=0.0):
    amplitude = c_bar * x**r
    return np.outer(amplitude, amplitude) * np.exp(-((x[:, None] - x) ** 2) / (2 * length_scale**2)) + s**2


def discrepancy_covariance(mode, hyper=None):
    params = TRUE_HYPER if hyper is None else hyper
    a, b = GROUP_INDICES["a"], GROUP_INDICES["b"]
    result = np.eye(len(X)) * DISC_NOISE
    result[np.ix_(a, a)] += kernel(
        X[a], params["c_bar"], params["length_scale"], params["r"], params.get("s", 0) if mode == "fixed" else 0
    )
    result[np.ix_(b, b)] += kernel(X[b], 0.08, 0.4, 0)
    return result


def discrepancy_data(mode, seed=418):
    return DISC_MATRIX @ TRUTH + np.linalg.cholesky(discrepancy_covariance(mode)) @ np.random.default_rng(seed).normal(
        size=len(X)
    )
