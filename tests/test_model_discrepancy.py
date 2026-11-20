"""Independent discrepancy kernels, configuration checks, and covariance placement."""

from types import SimpleNamespace

import numpy as np
import pytest

from bayesian.emulation.base import SortEmulationGroupObservables
from bayesian.model_discrepancy import (
    DiscrepancyKernelParams,
    add_discrepancy_covariance_all_groups,
    build_discrepancy_covariance_matrix,
    parse_discrepancy_config,
    parse_discrepancy_group_settings,
)


def loop_kernel(x, kernel, params):
    result = np.empty((len(x), len(x)))
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            result[i, j] = params.c_bar**2 * (xi * xj)**params.r * np.exp(
                -(xi - xj)**2 / (2 * params.length_scale**2)
            )
            if kernel == "kernel1":
                result[i, j] += params.s**2
    return result


@pytest.mark.parametrize("kernel", ["kernel1", "kernel2"])
@pytest.mark.parametrize("amplitude", [0.0, 0.35])
def test_kernel_matches_elementwise_reference_and_is_psd(kernel, amplitude):
    x = np.array([3.2, 0.5, 1.1, 1.1])
    params = DiscrepancyKernelParams(amplitude, 0.8, -0.6, 0.1)
    actual = build_discrepancy_covariance_matrix(x, kernel, params)
    np.testing.assert_allclose(actual, loop_kernel(x, kernel, params), atol=1e-15)
    assert np.linalg.eigvalsh(actual).min() > -1e-12
    order = [3, 0, 2, 1]
    np.testing.assert_allclose(build_discrepancy_covariance_matrix(x[order], kernel, params), actual[np.ix_(order, order)])


def block_example():
    sorter = SortEmulationGroupObservables({
        "a_first": ("a", slice(0, 2), slice(0, 2)),
        "b_only": ("b", slice(2, 3), slice(0, 1)),
        "a_last": ("a", slice(3, 5), slice(2, 4)),
    }, (1, 5))
    config = SimpleNamespace(sort_observables_in_matrix=sorter, emulation_groups_config={
        "a": SimpleNamespace(observable_slices=[slice(0, 2), slice(2, 4)]),
        "b": SimpleNamespace(observable_slices=[slice(0, 1)]),
    })
    params = {"a": DiscrepancyKernelParams(0.3, 1.2, -0.5, 0.07),
              "b": DiscrepancyKernelParams(0.5, 0.8, 0)}
    settings = {name: {"infer_hyperparameters": False, "fixed_params": value,
                       "kernel_type": "kernel1", "enabled": True, "param_names": []}
                for name, value in params.items()}
    coords = {"a": np.array([1, 2, 3, 4]), "b": np.array([1.5])}
    return config, settings, coords, params


@pytest.mark.parametrize("groups", [["a", "b"], ["b", "a"], ["b"], []])
def test_group_blocks_use_global_indices_and_preserve_within_group_correlations(groups):
    config, settings, coords, params = block_example()
    actual = add_discrepancy_covariance_all_groups(np.zeros((2, 1)), settings, coords, config, ["theta"], 5, {},
                                                 discrepancy_enabled_groups=groups)
    expected = np.zeros((5, 5))
    indices = {"a": [0, 1, 3, 4], "b": [2]}
    for name in groups:
        expected[np.ix_(indices[name], indices[name])] = loop_kernel(coords[name], "kernel1", params[name])
    np.testing.assert_allclose(actual[0], expected)
    np.testing.assert_allclose(actual[1], expected)


@pytest.mark.parametrize("order", [["r", "s", "length_scale", "c_bar"], ["length_scale", "r", "c_bar", "s"]])
def test_inferred_hyperparameters_are_matched_by_name(order):
    config, settings, coords, params = block_example()
    settings["a"].update(infer_hyperparameters=True, fixed_params=None, param_names=order)
    names = ["theta", *[f"a__{name}" for name in order]]
    theta = [[0.2, *[getattr(params["a"], name) for name in order]]]
    actual = add_discrepancy_covariance_all_groups(np.array(theta), settings, coords, config, names, 5,
                                                 {"a": list(range(1, 5))}, discrepancy_enabled_groups=["a"])
    expected = np.zeros((5, 5))
    expected[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])] = loop_kernel(coords["a"], "kernel1", params["a"])
    np.testing.assert_allclose(actual[0], expected)


@pytest.mark.parametrize("values", [[-0.1, 1, 0], [0.1, 0, 0], [0.1, -1, 0], [0.1, 1, np.nan], [0.1, 1, 0, -0.1]])
def test_invalid_fixed_parameters_raise(values):
    with pytest.raises(ValueError, match=r"discrepancy|Discrepancy"):
        parse_discrepancy_config({"kernel_type": "kernel1", "fixed_params": values}, "g")


@pytest.mark.parametrize("names", [["c_bar", "r"], ["c_bar", "length_scale", "oops"],
                                   ["c_bar", "length_scale", "r", "r"]])
def test_invalid_inferred_names_raise(names):
    with pytest.raises(ValueError, match="parameter_names"):
        parse_discrepancy_config({"infer_hyperparameters": True, "parameter_names": names}, "g")


@pytest.mark.parametrize(("lower", "upper"), [([0.1, 0.2], [1, 2, 0]),
                                        ([0.1, 0.2, -1], [0.05, 2, 0]),
                                        ([0.1, 0, -1], [1, 2, 0]),
                                        ([-1, 0.2, -1], [1, 2, 0]),
                                        ([0.1, 0.2, -1], [1, np.inf, 0])])
def test_invalid_inferred_bounds_raise(lower, upper):
    groups = {"g": {"discrepancy": {"enabled": True, "infer_hyperparameters": True,
                                    "parameter_names": ["c_bar", "length_scale", "r"],
                                    "min": lower, "max": upper}}}
    with pytest.raises(ValueError, match="bounds"):
        parse_discrepancy_group_settings(groups, ["theta"], [0], [1], None)


def test_kernel_failure_does_not_silently_remove_discrepancy():
    config, settings, coords, _params = block_example()
    coords["a"] = np.array([1, 2, np.nan, 4])
    with pytest.raises(ValueError, match="finite"):
        add_discrepancy_covariance_all_groups(np.zeros((1, 1)), settings, coords, config, ["theta"], 5, {},
                                             discrepancy_enabled_groups=["a"])


def test_coordinate_length_mismatch_raises():
    config, settings, coords, _params = block_example()
    coords["a"] = np.array([1, 2])
    with pytest.raises(ValueError, match=r"coordinate|Coordinate"):
        add_discrepancy_covariance_all_groups(np.zeros((1, 1)), settings, coords, config, ["theta"], 5, {},
                                             discrepancy_enabled_groups=["a"])


@pytest.mark.parametrize("kind", ["fixed", "inferred"])
def test_unknown_kernel_is_rejected_during_configuration(kind):
    config = {"kernel_type": "typo", "infer_hyperparameters": kind == "inferred",
              "fixed_params": [0.3, 1, 0], "parameter_names": ["c_bar", "length_scale", "r"]}
    with pytest.raises(ValueError, match="kernel"):
        parse_discrepancy_config(config, "a")


@pytest.mark.parametrize("coords", [[0, 1], [-1, 1], [[1, 2]]])
def test_kernel_rejects_coordinates_outside_its_domain(coords):
    with pytest.raises(ValueError, match="coordinates"):
        build_discrepancy_covariance_matrix(coords, "kernel2", DiscrepancyKernelParams(0.3, 1, -0.5))


def test_fixed_values_follow_declared_names():
    _, params, _ = parse_discrepancy_config({"fixed_params": [0.2, -0.5, 0.3, 1.2],
                                           "parameter_names": ["s", "r", "c_bar", "length_scale"]}, "a")
    assert params == DiscrepancyKernelParams(0.3, 1.2, -0.5, 0.2)
