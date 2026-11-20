"""Tests for standalone smoothing functions.

"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest  # noqa: F401

from bayesian import data_IO, outliers_smoothing

logger = logging.getLogger(__name__)

_data_dir = Path(__file__).parent / "test_data"


def test_smoothing() -> None:
    # Setup: Load data
    observables = data_IO.read_dict_from_h5(_data_dir, "observables.h5", verbose=False)
    measured_data = observables["Data"]["5020__PbPb__hadron__pt_ch_cms____0-5"]
    # Calculate bin centers from data
    x_min = measured_data["xmin"]
    x_max = measured_data["xmax"]
    bin_centers = x_min + (x_max - x_min) / 2.
    # And load values and errors
    values = np.loadtxt(_data_dir / "tables" / "Prediction" / "Prediction__exponential__5020__PbPb__hadron__pt_ch_cms____0-5__values.dat", ndmin=2)
    y_err = np.loadtxt(_data_dir / "tables" / "Prediction" / "Prediction__exponential__5020__PbPb__hadron__pt_ch_cms____0-5__errors.dat", ndmin=2)

    # Identify outliers and smooth them
    output_values, output_y_err, outliers_that_cannot_be_removed = outliers_smoothing.find_and_smooth_outliers_standalone(
        observable_key="hadron__pt_ch_cms",
        bin_centers=bin_centers,
        values=values,
        y_err=y_err,
        # Default values as of September 2024
        outliers_identification_methods={
            "large_statistical_errors": outliers_smoothing.OutliersConfig(n_RMS=2),
            "large_central_value_difference": outliers_smoothing.OutliersConfig(n_RMS=2),
        },
        smoothing_interpolation_method="linear",
        max_n_points_to_interpolate=2,
    )

    assert not np.allclose(output_values, values)
    assert not np.allclose(output_y_err, y_err)
