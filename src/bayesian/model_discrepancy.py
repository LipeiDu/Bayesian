"""Covariance kernels and named parameter mapping for model discrepancy."""

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from bayesian import data_IO
from bayesian.emulation import base
from bayesian.prior import normalize_prior_config

logger = logging.getLogger(__name__)
KernelType = Literal["kernel1", "kernel2"]


@dataclass
class DiscrepancyKernelParams:
    c_bar: float
    length_scale: float
    r: float
    s: float = 0.0


def _validate_kernel_params(params):
    values = np.array([params.c_bar, params.length_scale, params.r, params.s], dtype=float)
    if (not np.all(np.isfinite(values)) or params.c_bar < 0 or params.s < 0 or params.length_scale <= 0):
        message = "Discrepancy parameters must be finite, with nonnegative amplitudes and positive length_scale"
        raise ValueError(message)


def _validate_parameter_names(names, kernel_type):
    required = {"c_bar", "length_scale", "r"}
    allowed = required | ({"s"} if kernel_type == "kernel1" else set())
    if not required.issubset(names) or not set(names).issubset(allowed) or len(set(names)) != len(names):
        message = f"Invalid discrepancy parameter_names for {kernel_type}: {names}; required {sorted(required)}"
        raise ValueError(message)


def build_discrepancy_covariance_matrix(x_array: np.ndarray, kernel_type: KernelType,
                                        kernel_params: DiscrepancyKernelParams) -> np.ndarray:
    """Evaluate K_ij = c² (x_i x_j)^r exp[-(x_i-x_j)²/(2 ell²)] + s².

    ``s²`` is included only for kernel1. Positive physical bin centers are
    required because the inferred exponent may be any real number. Repeated
    coordinates and zero amplitudes are valid; the kernel may be semidefinite.
    """
    if kernel_type not in ("kernel1", "kernel2"):
        message = f"Unsupported discrepancy kernel type: {kernel_type}"
        raise ValueError(message)
    _validate_kernel_params(kernel_params)
    x = np.atleast_1d(np.asarray(x_array, dtype=float))
    if x.ndim != 1 or not np.all(np.isfinite(x)) or np.any(x <= 0):
        message = "Discrepancy coordinates must be a finite one-dimensional array of positive bin centers"
        raise ValueError(message)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        amplitude = kernel_params.c_bar * x**kernel_params.r
        correlation = np.exp(-0.5 * ((x[:, None] - x) / kernel_params.length_scale)**2)
        covariance = np.outer(amplitude, amplitude) * correlation
        if kernel_type == "kernel1":
            covariance += kernel_params.s**2
    if not np.all(np.isfinite(covariance)):
        message = "Discrepancy covariance is not finite for the supplied coordinates and parameters"
        raise ValueError(message)
    return covariance


def _group_mapping(emulation_config, group):
    mapping = emulation_config.sort_observables_in_matrix.emulation_group_to_observable_matrix
    return sorted(((name, global_slice, local_slice) for name, (owner, global_slice, local_slice) in mapping.items()
                   if owner == group), key=lambda item: item[2].start)


def build_observable_xcoords_per_group(config, emulation_config: base.EmulatorOrganizationConfig,
                                        experimental_results: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Read physical bin centers in each kernel's local emulator-matrix order."""
    observations = data_IO.read_dict_from_h5(config.input_analysis_dir, config.observables_filename, verbose=False)
    if emulation_config.sort_observables_in_matrix.shape[1] != len(experimental_results["y"]):
        message = "Discrepancy observable mapping and experimental vector have different lengths"
        raise ValueError(message)
    coordinates = {}
    for group in emulation_config.emulation_groups_config:
        centers = []
        for observable, _global, local in _group_mapping(emulation_config, group):
            data = observations["Data"][observable]
            xmin, xmax = np.asarray(data["xmin"]), np.asarray(data["xmax"])
            if xmin.ndim != 1 or xmin.shape != xmax.shape or len(xmin) != local.stop - local.start:
                message = f"Discrepancy bin-coordinate shape mismatch for {observable}"
                raise ValueError(message)
            centers.extend(0.5 * (xmin + xmax))
        coordinates[group] = np.asarray(centers, dtype=float)
    return coordinates


def discrepancy_parameter_indices(discrepancy_config, param_names):
    """Resolve exact group-qualified names in configured hyperparameter order."""
    if len(set(param_names)) != len(param_names):
        message = "Sampled parameter names must be unique"
        raise ValueError(message)
    indices = {}
    for group, cfg in discrepancy_config.items():
        if cfg["infer_hyperparameters"]:
            required = [f"{group}__{name}" for name in cfg["param_names"]]
            missing = set(required) - set(param_names)
            if missing:
                message = f"Missing discrepancy parameters for {group}: {sorted(missing)}"
                raise ValueError(message)
            indices[group] = [param_names.index(name) for name in required]
    return indices


def discrepancy_parameters_from_values(names, values):
    """Construct kernel parameters from named values, including reordered YAML."""
    return DiscrepancyKernelParams(**dict(zip(names, values, strict=True)))


def _global_feature_indices(emulation_config, group, n_features):
    entries = _group_mapping(emulation_config, group)
    local = np.array([index for _, _, slc in entries for index in range(slc.start, slc.stop)], dtype=int)
    indices = np.array([index for _, slc, _ in entries for index in range(slc.start, slc.stop)], dtype=int)
    if (not len(indices) or len(local) != len(indices) or not np.array_equal(local, np.arange(len(local)))
            or len(set(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n_features)):
        message = f"Invalid discrepancy observable mapping for group '{group}'"
        raise ValueError(message)
    return indices


def add_discrepancy_covariance_all_groups(theta: np.ndarray, discrepancy_config: dict, observable_xcoords: dict,
                                         emulation_config, param_names: list[str], n_features: int,
                                         discrepancy_param_indices: dict[str, list[int]], *,
                                         discrepancy_enabled_groups: list[str] | None = None) -> dict[int, np.ndarray]:
    """Place each group kernel in global observable order; invalid blocks raise.

    ``None`` selects configured enabled groups; an empty list selects no groups.
    Covariance errors must never silently remove a discrepancy model from a fit.
    """
    enabled = set(discrepancy_config) if discrepancy_enabled_groups is None else set(discrepancy_enabled_groups)
    layouts = {}
    for group, cfg in discrepancy_config.items():
        if group in enabled and cfg.get("enabled", True):
            indices = _global_feature_indices(emulation_config, group, n_features)
            if len(observable_xcoords[group]) != len(indices):
                message = f"Discrepancy coordinate count does not match mapped bins in group '{group}'"
                raise ValueError(message)
            layouts[group] = indices
    expected = discrepancy_parameter_indices({g: discrepancy_config[g] for g in layouts}, param_names)
    if any(discrepancy_param_indices.get(group) != indices for group, indices in expected.items()):
        message = "Discrepancy parameter indices do not match configured names"
        raise ValueError(message)
    result = {}
    for i, sample in enumerate(theta):
        total = np.zeros((n_features, n_features))
        for group, indices in layouts.items():
            cfg = discrepancy_config[group]
            params = cfg["fixed_params"]
            if cfg["infer_hyperparameters"]:
                params = discrepancy_parameters_from_values(cfg["param_names"], sample[expected[group]])
            covariance = build_discrepancy_covariance_matrix(observable_xcoords[group], cfg["kernel_type"], params)
            total[np.ix_(indices, indices)] += covariance
        result[i] = total
    return result


def parse_discrepancy_config(config_dict: dict, group_name: str) -> tuple[KernelType, DiscrepancyKernelParams | None, bool]:
    """Validate kernel kind and fixed or inferred named hyperparameters."""
    kernel_type = config_dict.get("kernel_type", "kernel1")
    if kernel_type not in ("kernel1", "kernel2"):
        message = f"Unsupported discrepancy kernel type for {group_name}: {kernel_type}"
        raise ValueError(message)
    infer = config_dict.get("infer_hyperparameters", False)
    if infer:
        names = config_dict.get("parameter_names", [])
        _validate_parameter_names(names, kernel_type)
        return kernel_type, None, True
    values = config_dict.get("fixed_params", [])
    if len(values) not in (3, 4):
        message = f"Missing or incomplete fixed_params for discrepancy group {group_name}"
        raise ValueError(message)
    names = config_dict.get("parameter_names", ["c_bar", "length_scale", "r", "s"][:len(values)])
    # Legacy kernel2 files may include a fixed s; it has no effect on that kernel.
    _validate_parameter_names(names, "kernel1")
    if len(names) != len(values):
        message = f"Discrepancy parameter_names and fixed_params lengths differ for {group_name}"
        raise ValueError(message)
    params = discrepancy_parameters_from_values(names, np.asarray(values, dtype=float))
    _validate_kernel_params(params)
    return kernel_type, params, False


def _inferred_bounds(config, names, group):
    lower, upper = np.asarray(config.get("min", []), dtype=float), np.asarray(config.get("max", []), dtype=float)
    if (lower.shape != (len(names),) or upper.shape != lower.shape or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper)) or np.any(lower >= upper)):
        message = f"Invalid discrepancy bounds for group {group}"
        raise ValueError(message)
    for i, name in enumerate(names):
        if (name == "length_scale" and lower[i] <= 0) or (name in ("c_bar", "s") and lower[i] < 0):
            message = f"Invalid discrepancy bounds for {group}__{name}: positive length_scale and nonnegative amplitudes required"
            raise ValueError(message)
    return lower.tolist(), upper.tolist()


def parse_discrepancy_group_settings(emulator_groups: dict[str, dict[str, Any]], base_names: list[str],
                                      base_min: list[float], base_max: list[float], base_prior_config: dict | None
                                      ) -> tuple[list[str], list[float], list[float], dict, dict, list[str]]:
    """Append inferred discrepancy parameters and their independent priors."""
    names, lower, upper = list(base_names), list(base_min), list(base_max)
    prior = normalize_prior_config(base_prior_config, len(base_names))
    configs, enabled = {}, []
    for group, group_config in emulator_groups.items():
        disc = group_config.get("discrepancy", {})
        if not disc.get("enabled", False):
            continue
        kernel, fixed, infer = parse_discrepancy_config(disc, group)
        parameters = list(disc["parameter_names"]) if infer else []
        if infer:
            lo, hi = _inferred_bounds(disc, parameters, group)
            names.extend(f"{group}__{name}" for name in parameters)
            lower.extend(lo)
            upper.extend(hi)
            disc_prior = normalize_prior_config(disc.get("prior"), len(parameters))
            for key in ("type", "mean", "std"):
                prior[key].extend(disc_prior[key])
        configs[group] = {"enabled": True, "kernel_type": kernel, "infer_hyperparameters": infer,
                          "fixed_params": fixed, "param_names": parameters}
        enabled.append(group)
    if len(set(names)) != len(names):
        message = "Duplicate parameter names when adding discrepancy parameters"
        raise ValueError(message)
    return names, lower, upper, prior, configs, enabled
