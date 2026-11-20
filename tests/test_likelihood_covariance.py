"""Independent references for Gaussian likelihoods and emulator covariance."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

from bayesian import log_posterior
from bayesian.emulation import base


@pytest.mark.parametrize("dimension", [1, 2, 5])
def test_gaussian_likelihood_matches_scipy(dimension) -> None:
    rng = np.random.default_rng(20261119)
    for _ in range(20):
        factor = rng.normal(size=(dimension, dimension))
        covariance = factor @ factor.T + np.eye(dimension) * 0.2
        residual = rng.normal(size=dimension)
        expected = multivariate_normal.logpdf(residual, cov=covariance)
        expected += 0.5 * dimension * np.log(2 * np.pi)
        np.testing.assert_allclose(log_posterior._loglikelihood(residual, covariance), expected, rtol=2e-12, atol=2e-12)


def test_gaussian_likelihood_preserves_small_positive_eigenvalue() -> None:
    covariance = np.array([[1.0, 1.0 - 1e-7], [1.0 - 1e-7, 1.0]])
    residual = np.array([0.3, 0.3])
    expected = multivariate_normal.logpdf(residual, cov=covariance) + np.log(2 * np.pi)
    np.testing.assert_allclose(log_posterior._loglikelihood(residual, covariance), expected, rtol=1e-9)


@pytest.mark.parametrize(
    "covariance",
    [np.zeros((1, 1)), np.array([[-1.0]]), np.ones((2, 2)), np.array([[1.0, 2.0], [2.0, 1.0]])],
    ids=["zero", "negative", "singular", "indefinite"],
)
def test_gaussian_likelihood_rejects_non_positive_definite_covariance(covariance) -> None:
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        log_posterior._loglikelihood(np.ones(len(covariance)), covariance)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_gaussian_likelihood_rejects_nonfinite_covariance(invalid) -> None:
    covariance = np.eye(2)
    covariance[0, 1] = covariance[1, 0] = invalid
    with pytest.raises(ValueError, match="finite"):
        log_posterior._loglikelihood(np.ones(2), covariance)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_gaussian_likelihood_rejects_nonfinite_residual(invalid) -> None:
    with pytest.raises(ValueError, match="finite"):
        log_posterior._loglikelihood(np.array([invalid, 0.0]), np.eye(2))


def test_gaussian_likelihood_rejects_asymmetric_covariance() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        log_posterior._loglikelihood(np.ones(2), np.array([[1.0, 0.5], [0.0, 1.0]]))


@pytest.mark.parametrize(
    ("residual", "covariance"),
    [(np.ones((2, 1)), np.eye(2)), (np.ones(2), np.eye(3)), (np.ones(2), np.ones(2))],
)
def test_gaussian_likelihood_rejects_incompatible_shapes(residual, covariance) -> None:
    with pytest.raises(ValueError, match="shape"):
        log_posterior._loglikelihood(residual, covariance)


def test_gaussian_likelihood_reports_lapack_solve_error(monkeypatch) -> None:
    monkeypatch.setattr(log_posterior.lapack, "dpotrs", lambda *_args: (np.ones(2), -2))
    with pytest.raises(ValueError, match=r"dpotrs.*2-th argument"):
        log_posterior._loglikelihood(np.ones(2), np.eye(2))


@pytest.mark.parametrize("group_order", [("group_a", "group_b"), ("group_b", "group_a")])
@pytest.mark.parametrize("selected", [[0, 1, 2, 3, 4], [4, 2, 0, 3, 1], [1, 2, 4]])
def test_prediction_merge_preserves_covariance_and_likelihood(group_order, selected, monkeypatch) -> None:
    # Group A's first observable has two bins. Its second observable is separated
    # from the first in global order by Group B's observable.
    definitions = {
        "a_first": ("group_a", slice(0, 2), [0, 1]),
        "b_only": ("group_b", slice(0, 1), [2]),
        "a_second": ("group_a", slice(2, 4), [3, 4]),
    }
    mapping = {}
    for output_index, original_index in enumerate(selected):
        name, (group, source, original) = next(
            (name, definition) for name, definition in definitions.items() if original_index in definition[2]
        )
        source_index = source.start + original.index(original_index)
        mapping[f"{name}_{output_index}"] = (
            group,
            slice(output_index, output_index + 1),
            slice(source_index, source_index + 1),
        )
    sorter = base.SortEmulationGroupObservables(mapping, shape=(3, len(selected)))
    covariance_a = np.array([[2.0, 0.2, 0.6, -0.1], [0.2, 1.5, 0.1, 0.4], [0.6, 0.1, 1.7, 0.3], [-0.1, 0.4, 0.3, 1.2]])
    scale = np.arange(1.0, 4.0)
    predictions = {
        "group_a": {"central_value": np.tile([1.0, 2.0, 4.0, 5.0], (3, 1)), "cov": scale[:, None, None] * covariance_a},
        "group_b": {"central_value": np.full((3, 1), 3.0), "cov": scale[:, None, None] * 0.8},
    }
    config = SimpleNamespace(
        emulation_groups_config={name: name for name in group_order}, sort_observables_in_matrix=sorter
    )
    monkeypatch.setattr(base, "predict_emulation_group", lambda _parameters, results, *_args, **_kwargs: results)
    originals = {name: {key: value.copy() for key, value in result.items()} for name, result in predictions.items()}
    merged = base.predict(
        np.zeros((3, 1)),
        config,
        emulation_group_results=predictions,
        emulator_cov_unexplained={name: np.zeros((1, 1)) for name in group_order},
    )

    # Build the global reference directly from labeled bin positions.
    reference = np.zeros((3, 5, 5))
    reference[:, np.array([0, 1, 3, 4])[:, None], [0, 1, 3, 4]] = scale[:, None, None] * covariance_a
    reference[:, 2, 2] = scale * 0.8
    reference = reference[:, selected][:, :, selected]
    np.testing.assert_allclose(merged["central_value"], np.tile(np.arange(1.0, 6.0)[selected], (3, 1)))
    np.testing.assert_allclose(merged["cov"], reference)
    for residual, covariance, expected_covariance in zip(
        merged["central_value"], merged["cov"], reference, strict=True
    ):
        expected = multivariate_normal.logpdf(residual, cov=expected_covariance)
        expected += 0.5 * len(selected) * np.log(2 * np.pi)
        np.testing.assert_allclose(log_posterior._loglikelihood(residual, covariance), expected)
    for name, result in predictions.items():
        for key, value in result.items():
            np.testing.assert_array_equal(value, originals[name][key])


class _PCEmulator:
    def predict(self, parameters, return_std):
        assert return_std
        return parameters[:, 0], np.full(len(parameters), 0.2)


@pytest.mark.parametrize("precomputed", [False, True])
def test_pca_covariance_is_independent_of_prediction_batch_size(precomputed) -> None:
    components = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, np.sqrt(2)]]) / np.sqrt(2)
    scaler = StandardScaler().fit([[-2.0, -3.0, -4.0], [2.0, 3.0, 4.0]])
    results = {
        "emulators": [_PCEmulator()],
        "PCA": {
            "scaler": scaler,
            "pca": SimpleNamespace(components_=components, explained_variance_=np.array([2.0, 0.4, 0.1])),
        },
    }
    config = SimpleNamespace(n_pc=1)
    parameters = np.array([[-0.3], [0.1], [0.8]])
    discarded = components[1:].T @ np.diag([0.4, 0.1]) @ components[1:]
    covariance = discarded if precomputed else None
    batched = base.predict_emulation_group(parameters, results, config, covariance)
    individual = [base.predict_emulation_group(row[None, :], results, config, covariance) for row in parameters]
    retained = 0.2**2 * np.outer(components[0], components[0])
    expected = (retained + discarded) * np.outer([2.0, 3.0, 4.0], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(batched["cov"], np.repeat(expected[None, :, :], 3, axis=0))
    np.testing.assert_allclose(batched["cov"], np.concatenate([result["cov"] for result in individual]))
    np.testing.assert_allclose(
        batched["central_value"], np.concatenate([result["central_value"] for result in individual])
    )


def test_log_posterior_propagates_covariance_failure(monkeypatch) -> None:
    settings = {
        "g_min": np.array([0.0]),
        "g_max": np.array([1.0]),
        "g_model_param_indices": [0],
        "g_param_names": ["theta"],
        "g_full_model_param_names": ["theta"],
        "g_fixed_model_parameters": {},
        "g_discrepancy_enabled": [],
        "g_experimental_results": {"y": np.zeros(2), "y_err": np.full(2, 0.1)},
        "g_sequential_inference_config": None,
        "g_sequential_log_prior_fn": None,
        "g_log_prior_fn": lambda samples: np.zeros(len(samples)),
    }
    for name, value in settings.items():
        monkeypatch.setattr(log_posterior, name, value, raising=False)
    monkeypatch.setattr(
        base,
        "predict",
        lambda *_args, **_kwargs: {"central_value": np.ones((1, 2)), "cov": np.array([[[1.0, 2.0], [2.0, 1.0]]])},
    )
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        log_posterior.log_posterior([[0.5]])
