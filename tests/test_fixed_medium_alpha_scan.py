from __future__ import annotations

import numpy as np
import pytest

from bayesian.fixed_medium_alpha_scan import normalize_log_curve, summarize_density


def test_normalize_log_curve_integrates_to_one() -> None:
    grid = np.linspace(-3.0, 3.0, 1001)
    density = normalize_log_curve(grid, -0.5 * np.square(grid))

    integral = np.sum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))
    assert integral == pytest.approx(1.0, rel=1.0e-8)


def test_summarize_density_recovers_symmetric_gaussian() -> None:
    grid = np.linspace(-5.0, 5.0, 4001)
    density = normalize_log_curve(grid, -0.5 * np.square(grid))

    summary = summarize_density(grid, density)

    assert summary["mean"] == pytest.approx(0.0, abs=1.0e-8)
    assert summary["median"] == pytest.approx(0.0, abs=3.0e-3)
    assert summary["std"] == pytest.approx(1.0, rel=1.0e-4)
