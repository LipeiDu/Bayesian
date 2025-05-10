import os
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Literal, Callable, Any, Optional
from collections import Counter
from bayesian.emulation import base
from bayesian import data_IO
import h5py

import logging
logger = logging.getLogger(__name__)

########################################################################################################
KernelType = Literal["kernel1", "kernel2"]

@dataclass
class DiscrepancyKernelParams:
    c_bar: float
    length_scale: float
    r: float
    s: float = 0.0  # Only used in kernel1

def build_discrepancy_covariance_matrix(
    x_array: np.ndarray,
    kernel_type: KernelType,
    kernel_params: DiscrepancyKernelParams
) -> np.ndarray:
    """
    Build covariance matrix Σ_δ(x_i, x_j) for model discrepancy using specified kernel.

    The kernel implicitly assumes that: 
    - All observables in the group can be embedded on a shared 1D axis (e.g., pT), 
      and their discrepancy is spatially correlated on that axis.
    - Discrepancy is not just per observable, but per physical bin location, so bins at similar pT across different observables are coupled.
    - This allows the kernel to capture correlated model bias, e.g., if the model underpredicts jet RAA at pT∼30 GeV across multiple systems, 
      the kernel will connect those residuals and not treat them independently.
    """
    x_array = np.atleast_1d(x_array).astype(np.float64)
    n = len(x_array)

    # Vectorized construction
    X, Y = np.meshgrid(x_array, x_array, indexing='ij')
    if kernel_type == "kernel1":
        K = (
            kernel_params.s**2 +
            kernel_params.c_bar**2 *
            (X * Y)**kernel_params.r *
            np.exp(-0.5 * ((X - Y) / kernel_params.length_scale)**2)
        )
    elif kernel_type == "kernel2":
        K = (
            kernel_params.c_bar**2 *
            (X * Y)**kernel_params.r *
            np.exp(-0.5 * ((X - Y) / kernel_params.length_scale)**2)
        )
    else:
        raise ValueError(f"Unsupported kernel type: '{kernel_type}'")

    return K

########################################################################################################
def build_observable_xcoords_per_group(
    config,
    emulation_config: base.EmulatorOrganizationConfig,
    experimental_results: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Construct observable x-coordinates (e.g., pt bin centers) for each observable group.
    The code treats x_coords as:
        - A 1D concatenation of bin centers across all observables in a group.
        - A shared physical axis over which discrepancy is modeled.
        - A coordinate system for spatially-correlated uncertainty.

    x_coords is the concatenated list of bin centers (like pt values) across all observables in a single observable group — e.g., all jet__pt__... observables. 
    It defines the kinematic space over which the discrepancy kernel operates.

    Args:
        config: MCMCConfig object
        emulation_config: EmulatorOrganizationConfig object
        experimental_results: dict containing 'y' vector (used to identify keys)

    Returns:
        dict[group_name] -> np.ndarray of x bin centers
    """
    from bayesian.mcmc import MCMCConfig

    observable_xcoords = {}

    # Load full observable dictionary once
    observable_dict = data_IO.read_dict_from_h5(
        config.input_analysis_dir,
        config.observables_filename,
        verbose=False
    )

    # Loop over observable groups
    for group_name, group_cfg in config.analysis_config['parameters']['emulators'].items():
        group_cfg_obj = emulation_config.emulation_groups_config[group_name]
        group_filter = group_cfg_obj.observable_filter

        # Get the sorted list of observables in this group
        observable_list = data_IO.sorted_observable_list_from_dict(observable_dict, observable_filter=group_filter)

        x_coords = []
        for obs_key in observable_list:
            xmin = observable_dict["Data"][obs_key]["xmin"]
            xmax = observable_dict["Data"][obs_key]["xmax"]
            x_center = 0.5 * (np.array(xmin) + np.array(xmax))
            x_coords.extend(x_center)

        observable_xcoords[group_name] = np.array(x_coords)

    return observable_xcoords

########################################################################################################
def add_discrepancy_covariance_all_groups(
    theta: npt.NDArray[np.float64],
    discrepancy_config: dict,
    observable_xcoords: dict[str, npt.NDArray[np.float64]],
    emulation_config,
    param_names: list[str],
    n_features: int,
    discrepancy_param_indices: dict[str, list[int]],
    *,
    discrepancy_enabled_groups: Optional[list[str]] = None
) -> dict[int, np.ndarray]:
    """
    Build discrepancy covariance blocks for all observable groups and return
    a dict mapping sample index → discrepancy covariance matrix (n_features, n_features).
    Used during MCMC sampling — no saving or plotting.
    """
    n_samples = theta.shape[0]
    discrepancy_blocks = {}

    for i in range(n_samples):
        total_cov = np.zeros((n_features, n_features))

        for group_name, cfg in discrepancy_config.items():
            if discrepancy_enabled_groups and group_name not in discrepancy_enabled_groups:
                continue
            if not cfg["infer_hyperparameters"] and cfg["fixed_params"] is None:
                continue

            x_coords = observable_xcoords[group_name]

            # Get observable slices for this group
            slices = getattr(emulation_config.emulation_groups_config[group_name], "observable_slices", None)
            if slices is None:
                raise RuntimeError(f"Missing observable_slices for group '{group_name}'")

            # Ensure all slices are Python slice objects
            if slices and isinstance(slices[0], (list, tuple, np.ndarray)):
                slices = [slice(int(s[0]), int(s[1])) for s in slices]

            feature_indices = [s.start + j for s in slices for j in range(s.stop - s.start)]
            idxs = np.array(feature_indices)

            try:
                # Fixed or inferred kernel parameters
                if not cfg["infer_hyperparameters"]:
                    kernel_params = cfg["fixed_params"]
                else:
                    theta_vals = theta[i, discrepancy_param_indices[group_name]]
                    kernel_params = DiscrepancyKernelParams(
                        c_bar=theta_vals[0],
                        length_scale=theta_vals[1],
                        r=theta_vals[2],
                        s=theta_vals[3] if len(theta_vals) > 3 else 0.0
                    )

                cov_d = build_discrepancy_covariance_matrix(
                    x_coords, cfg["kernel_type"], kernel_params
                )

                # Basic checks
                if cov_d.ndim != 2 or cov_d.shape[0] != cov_d.shape[1]:
                    logger.warning(f"[WARNING] Invalid shape for discrepancy covariance in group '{group_name}': {cov_d.shape}")
                    continue
                if not np.all(np.isfinite(cov_d)):
                    logger.warning(f"[WARNING] NaN or Inf detected in discrepancy covariance for group '{group_name}'.")
                    continue
                if not np.allclose(cov_d, cov_d.T, atol=1e-10):
                    logger.warning(f"[WARNING] Discrepancy covariance not symmetric for group '{group_name}'.")
                    continue
                if np.any(np.linalg.eigvalsh(cov_d) < -1e-12):
                    logger.warning(f"[WARNING] Discrepancy covariance for group '{group_name}' is not positive semi-definite.")
                    continue

                total_cov[np.ix_(idxs, idxs)] += cov_d

            except Exception as e:
                logger.warning(f"[ERROR] Exception while building discrepancy block for group '{group_name}': {e}")
                continue

        discrepancy_blocks[i] = total_cov

    return discrepancy_blocks

########################################################################################################
def parse_discrepancy_config(config_dict: dict, group_name: str) -> tuple[KernelType, DiscrepancyKernelParams | None, bool]:
    """
    Parse discrepancy kernel settings from YAML config.
    Returns kernel_type, kernel_params (or None), and infer_flag.
    """
    kernel_type: KernelType = config_dict.get("kernel_type", "kernel1")
    infer = config_dict.get("infer_hyperparameters", False)

    # 1. For hyperparameter inference mode
    if infer:
        param_names = config_dict.get("parameter_names", [])
        if not param_names:
            raise ValueError("You must provide parameter_names if infer_hyperparameters: True")
        logger.info(f"Inference enabled for group: {group_name} with discrepancy hyperparameters: {param_names}")

        return kernel_type, None, True

    # 2. For fixed mode
    fixed_values = config_dict.get("fixed_params", [])
    param_names = config_dict.get("parameter_names", [])

    logger.info(f"Inference disabled for group: {group_name} with fixed discrepancy hyperparameters: {param_names}")

    if not fixed_values or len(fixed_values) < 3:
        raise ValueError("Missing or incomplete 'fixed_params' for discrepancy kernel")

    # Map by convention: [c_bar, length_scale, r, s]
    params = DiscrepancyKernelParams(
        c_bar=float(fixed_values[0]),
        length_scale=float(fixed_values[1]),
        r=float(fixed_values[2]),
        s=float(fixed_values[3]) if len(fixed_values) > 3 else 0.0
    )

    return kernel_type, params, False

########################################################################################################
def parse_discrepancy_group_settings(
    emulator_groups: dict[str, dict[str, Any]],
    base_names: list[str],
    base_min: list[float],
    base_max: list[float],
    base_prior_config: dict | None,
) -> tuple[dict[str, Any], list[str], list[float], list[float], dict]:
    """
    Gather model discrepancy configuration per observable group.
    Extend the main parameter list (model parameters) and prior config if discrepancy parameters are to be inferred.

    Returns:
        - discrepancy_config: dict[group_name] → discrepancy kernel info
        - updated names, parameter_min, parameter_max
        - updated prior_config
        - discrepancy_enabled_groups: list of group names with discrepancy enabled
    """

    # Start with fresh copies of incoming parameter lists to avoid mutating them
    names = list(base_names)
    parameter_min = list(base_min)
    parameter_max = list(base_max)

    # Clone prior config if given, or initialize default
    if base_prior_config is None:
        prior_config = {"type": [], "mean": [], "std": []}
    else:
        prior_config = {
            "type": list(base_prior_config.get("type", [])),
            "mean": list(base_prior_config.get("mean", [])),
            "std": list(base_prior_config.get("std", [])),
        }

    discrepancy_config = {}
    discrepancy_enabled_groups = []

    for group_name, group_cfg in emulator_groups.items():
        disc_cfg = group_cfg.get("discrepancy", {})
        if disc_cfg.get("enabled", False):
            discrepancy_enabled_groups.append(group_name)
            kernel_type, fixed_params, infer_flag = parse_discrepancy_config(disc_cfg, group_name)

            if infer_flag:
                param_names = disc_cfg["parameter_names"]
                param_min = disc_cfg["min"]
                param_max = disc_cfg["max"]

                full_param_names = [f"{group_name}__{p}" for p in param_names]

                # Prevent duplicates
                for pname in full_param_names:
                    if pname in names:
                        raise ValueError(f"Duplicate parameter name '{pname}' detected when adding discrepancy parameters.")

                names += full_param_names
                parameter_min += param_min
                parameter_max += param_max

                # Extend prior config
                disc_prior = disc_cfg.get("prior", {})
                type_ = disc_prior.get("type", ["uniform"] * len(param_names))
                mean_ = disc_prior.get("mean", [None] * len(param_names))
                std_  = disc_prior.get("std",  [None] * len(param_names))

                # Validate lengths
                if not (len(type_) == len(mean_) == len(std_) == len(param_names)):
                    raise ValueError(
                        f"Mismatch in prior config lengths for group '{group_name}': "
                        f"{len(type_)} types, {len(mean_)} means, {len(std_)} stds, {len(param_names)} parameters"
                    )

                prior_config["type"] += type_
                prior_config["mean"] += mean_
                prior_config["std"]  += std_

            else:
                param_names = []

            discrepancy_config[group_name] = {
                "enabled": True,
                "kernel_type": kernel_type,
                "infer_hyperparameters": infer_flag,
                "fixed_params": fixed_params,
                "param_names": param_names,
            }

    # Post-check for duplicates in all parameters
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"[ERROR] Duplicate parameter names found in final list: {duplicates}")

    return names, parameter_min, parameter_max, prior_config, discrepancy_config, discrepancy_enabled_groups
