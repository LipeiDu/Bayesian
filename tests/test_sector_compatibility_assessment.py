from __future__ import annotations

import numpy as np
import pytest

from bayesian.sector_compatibility_assessment import (
    predictive_moments,
    soft_hdr_thresholds,
    weighted_hdr_fractions,
)


def test_soft_hdr_thresholds_and_weighted_fractions() -> None:
    reference = np.arange(100, dtype=float)
    thresholds = soft_hdr_thresholds(reference, [0.68, 0.95])
    fractions = weighted_hdr_fractions(reference, np.ones(100), thresholds)

    assert fractions[0.68] == pytest.approx(0.68, abs=0.01)
    assert fractions[0.95] == pytest.approx(0.95, abs=0.01)


def test_predictive_moments_include_parameter_emulator_and_experiment_variance() -> None:
    central = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    emulator_covariance = np.repeat(np.eye(2)[np.newaxis, :, :] * 0.5, 2, axis=0)
    mean, covariance = predictive_moments(
        central,
        emulator_covariance,
        weights=[0.25, 0.75],
        experimental_variances=[0.2, 0.3],
    )

    np.testing.assert_allclose(mean, [1.5, 2.5])
    np.testing.assert_allclose(covariance, [[1.45, 0.75], [0.75, 1.55]])


def test_soft_hdr_thresholds_reject_invalid_mass() -> None:
    with pytest.raises(ValueError, match="strictly"):
        soft_hdr_thresholds([0.0, 1.0], [1.0])
