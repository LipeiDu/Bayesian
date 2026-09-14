from __future__ import annotations

import numpy as np
import pytest

from bayesian.hard_likelihood_diagnostics import (
    _short_observable_label,
    weighted_quantile,
    weighted_summary,
)


def test_weighted_quantile_reproduces_symmetric_sample() -> None:
    values = np.asarray([0.0, 1.0, 2.0])
    weights = np.ones(3)

    result = weighted_quantile(values, [0.5], weights)

    assert result[0] == pytest.approx(1.0)


def test_weighted_summary_respects_dominant_weight() -> None:
    summary = weighted_summary([0.0, 10.0], [0.9, 0.1])

    assert summary["mean"] == pytest.approx(1.0)
    assert summary["std"] == pytest.approx(3.0)
    assert summary["min"] == 0.0
    assert summary["max"] == 10.0


def test_weighted_quantile_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        weighted_quantile([0.0, 1.0], [0.5], [1.0])


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("2760__PbPb__hadron__pt_ch_atlas____0-5", "ATLAS charged h. 0-5%"),
        ("2760__PbPb__hadron__pt_pi0_alice____0-10", "ALICE pi0 0-10%"),
        ("2760__PbPb__inclusive_jet__pt_cms__R0.3__5-10", "CMS jet R=0.3 5-10%"),
    ],
)
def test_short_observable_label(name: str, expected: str) -> None:
    assert _short_observable_label(name) == expected
