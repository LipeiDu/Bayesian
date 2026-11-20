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


def test_uniform_density_quantiles_include_both_grid_endpoints():
    summary = summarize_density(np.array([0.0, 0.2, 0.9, 1.0]), np.ones(4))
    np.testing.assert_allclose([summary["q16"], summary["median"], summary["q84"]], [0.16, 0.5, 0.84], atol=1e-12)


def test_piecewise_linear_density_quantiles_are_inverted_exactly():
    # p(x)=2*x, CDF=x**2 on a nonuniform grid.
    grid = np.array([0, 0.1, 0.6, 1.0])
    summary = summarize_density(grid, 2 * grid)
    np.testing.assert_allclose([summary["q16"], summary["median"], summary["q84"]], np.sqrt([0.16, 0.5, 0.84]), atol=1e-12)
    assert summary["mean"] == pytest.approx(2 / 3, abs=1e-12)
    assert summary["std"] == pytest.approx(np.sqrt(1 / 18), abs=1e-12)


@pytest.mark.parametrize("grid,density", [([0], [1]), ([1, 0], [1, 1]), ([0, 1], [0, 0]),
                                         ([0, 1], [1, -1]), ([0, np.nan], [1, 1])])
def test_invalid_scan_density_raises(grid, density):
    with pytest.raises(ValueError):
        summarize_density(np.asarray(grid, dtype=float), np.asarray(density, dtype=float))
