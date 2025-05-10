import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from typing import Any, Optional
from matplotlib.lines import Line2D
from bayesian import data_IO
from bayesian.emulation import base
from bayesian.model_discrepancy import build_discrepancy_covariance_matrix, build_observable_xcoords_per_group
from bayesian.model_discrepancy import parse_discrepancy_group_settings, DiscrepancyKernelParams

sns.set_context('paper', rc={'font.size': 14, 'axes.titlesize': 14, 'axes.labelsize': 14})

logger = logging.getLogger(__name__)

####################################################################################################################
def plot(config):
    """
    Master function to generate discrepancy plots. Generates and saves kernels.
    """
    file_path = os.path.join(config.output_dir, 'discrepancy_kernels.h5')

    logger.info(f"[Discrepancy] Kernel file generating now...")

    # Load experimental data
    experimental_results = data_IO.data_array_from_h5(
        config.input_analysis_dir, 'observables.h5',
        pseudodata_index=-1,
        observable_filter=None
    )

    # Get observable x-coords
    emulation_config = base.EmulatorOrganizationConfig.from_config_file(
        analysis_name=config.analysis_name,
        parameterization=config.parameterization,
        analysis_config=config.analysis_config,
        config_file=config.config_file,
    )
    observable_xcoords = build_observable_xcoords_per_group(config, emulation_config, experimental_results)

    # Save kernels
    generate_and_save_discrepancy_kernels(config, experimental_results, observable_xcoords)

    # Load and plot
    kernel_data = load_discrepancy_kernels(file_path)

    for group_name in config.analysis_config['parameters']['emulators'].keys():
        if config.analysis_config['parameters']['emulators'][group_name].get('discrepancy', {}).get('enabled', False):
            if group_name in kernel_data:
                logger.info(f"[Discrepancy] Plotting discrepancy for group '{group_name}'...")
                plot_discrepancy(config, group_name, kernel_data[group_name])
            else:
                logger.warning(f"[Discrepancy] Group '{group_name}' enabled but no kernel saved.")

####################################################################################################################
def generate_and_save_discrepancy_kernels(config, experimental_results, observable_xcoords):
    """
    Generate representative discrepancy kernels for each group (fixed or inferred) and save to HDF5.
    Intended only for plotting and diagnostics — not used during MCMC sampling.
    """
    from bayesian.emulation.base import SortEmulationGroupObservables

    # Load model parameter config
    param_cfg = config.analysis_config['parameterization'][config.parameterization]
    names = param_cfg['names']
    parameter_min = param_cfg['min']
    parameter_max = param_cfg['max']
    model_prior_config = param_cfg.get("prior", None)
    emulator_groups = config.analysis_config['parameters']['emulators']

    # Extend param list with discrepancy parameters
    names, parameter_min, parameter_max, _, discrepancy_config, discrepancy_enabled_groups = parse_discrepancy_group_settings(
        emulator_groups, names, parameter_min, parameter_max, model_prior_config
    )

    param_names = names
    theta = np.array([(lo + hi) / 2 for lo, hi in zip(parameter_min, parameter_max)])[None, :]  # shape (1, n_params)

    # Index classification
    discrepancy_param_indices = {}
    model_param_indices = []
    for i, name in enumerate(param_names):
        matched = False
        for group in discrepancy_config:
            prefix = f"{group}__"
            if name.startswith(prefix):
                discrepancy_param_indices.setdefault(group, []).append(i)
                matched = True
                break
        if not matched:
            model_param_indices.append(i)

    # Emulator config
    emulation_config = base.EmulatorOrganizationConfig.from_config_file(
        analysis_name=config.analysis_name,
        parameterization=config.parameterization,
        config_file=config.config_file,
        analysis_config=config.analysis_config,
    )

    # Observable mapping metadata
    mapping_obj = SortEmulationGroupObservables.learn_mapping(emulation_config)
    group_metadata: dict[str, dict[str, Any]] = {}
    for observable_name, (group_name, global_slice, _) in mapping_obj.emulation_group_to_observable_matrix.items():
        if group_name not in group_metadata:
            group_metadata[group_name] = {"observable_slices": [], "observable_labels": []}
        group_metadata[group_name]["observable_slices"].append(global_slice)
        group_metadata[group_name]["observable_labels"].append(observable_name)

    # Run emulator once to extract diagonals
    emulator_predictions = base.predict(
        theta[:, model_param_indices],
        emulation_config=emulation_config,
        emulator_cov_unexplained=None
    )
    diag_emul = np.diagonal(emulator_predictions['cov'][0])
    diag_exp = experimental_results['y_err'] ** 2

    # Per-group diagonals
    diag_emul_group = {}
    diag_exp_group = {}
    for group in discrepancy_enabled_groups:
        slices = group_metadata[group]["observable_slices"]
        indices = [i for s in slices for i in range(s.start, s.stop)]
        diag_emul_group[group] = diag_emul[indices]
        diag_exp_group[group] = diag_exp[indices]

    # Build representative kernels
    cached_kernels = {}
    for group_name in discrepancy_enabled_groups:
        cfg = discrepancy_config[group_name]
        x_coords = observable_xcoords[group_name]

        if not cfg["infer_hyperparameters"]:
            kernel_params = cfg["fixed_params"]
        else:
            theta_vals = theta[0, discrepancy_param_indices[group_name]]
            kernel_params = DiscrepancyKernelParams(
                c_bar=theta_vals[0],
                length_scale=theta_vals[1],
                r=theta_vals[2],
                s=theta_vals[3] if len(theta_vals) > 3 else 0.0
            )

        try:
            cov_d = build_discrepancy_covariance_matrix(
                x_coords, cfg["kernel_type"], kernel_params
            )

            # Perform validation
            if cov_d.ndim != 2 or cov_d.shape[0] != cov_d.shape[1]:
                logger.warning(f"[WARNING] Invalid shape for discrepancy kernel in group '{group_name}': {cov_d.shape}")
                continue
            if not np.all(np.isfinite(cov_d)):
                logger.warning(f"[WARNING] NaN or Inf in discrepancy kernel for group '{group_name}'")
                continue
            if not np.allclose(cov_d, cov_d.T, atol=1e-10):
                logger.warning(f"[WARNING] Kernel not symmetric for group '{group_name}'")
                continue
            if np.any(np.linalg.eigvalsh(cov_d) < -1e-12):
                logger.warning(f"[WARNING] Kernel not positive semi-definite for group '{group_name}'")
                continue

            cached_kernels[group_name] = cov_d

        except Exception as e:
            logger.warning(f"[ERROR] Exception while building kernel for group '{group_name}': {e}")
            continue

    # Save to HDF5
    if not cached_kernels:
        logger.warning("[Discrepancy] No valid kernels to save.")
        return

    save_path = os.path.join(config.output_dir, 'discrepancy_kernels.h5')
    save_discrepancy_kernels_h5(
        save_path=save_path,
        cached_kernels=cached_kernels,
        observable_xcoords=observable_xcoords,
        group_metadata=group_metadata,
        diag_emulator_cov=diag_emul_group,
        diag_exp_cov=diag_exp_group,
        discrepancy_enabled_groups=discrepancy_enabled_groups
    )

#---------------------------------------------------------------
def save_discrepancy_kernels_h5(
    save_path: str,
    cached_kernels: dict[str, np.ndarray],
    observable_xcoords: dict[str, np.ndarray],
    group_metadata: dict[str, dict[str, Any]],
    diag_emulator_cov: Optional[dict[str, np.ndarray]] = None,
    diag_exp_cov: Optional[dict[str, np.ndarray]] = None,
    discrepancy_enabled_groups: Optional[list[str]] = None
):
    """
    Save discrepancy kernel matrices and metadata to HDF5.
    """
    logger.info(f"[Discrepancy] Saving kernels to {save_path}")

    with h5py.File(save_path, 'w') as f:
        for group_name, kernel in cached_kernels.items():
            if discrepancy_enabled_groups and group_name not in discrepancy_enabled_groups:
                continue

            group = f.create_group(group_name)
            group.create_dataset("kernel_matrix", data=kernel)
            group.create_dataset("x_coords", data=observable_xcoords[group_name])

            slice_array = np.array([[s.start, s.stop] for s in group_metadata[group_name]["observable_slices"]])
            label_array = np.array(group_metadata[group_name]["observable_labels"], dtype=h5py.string_dtype())

            group.create_dataset("observable_slices", data=slice_array)
            group.create_dataset("observable_labels", data=label_array)

            if diag_emulator_cov and group_name in diag_emulator_cov:
                group.create_dataset("diag_emulator_cov", data=diag_emulator_cov[group_name])
            if diag_exp_cov and group_name in diag_exp_cov:
                group.create_dataset("diag_exp_cov", data=diag_exp_cov[group_name])

#---------------------------------------------------------------
def plot_discrepancy(config, group_name: str, kernel_data: dict):
    """Generate plots for discrepancy effects for a specific emulator group."""
    K = kernel_data['kernel_matrix']
    x_coords = kernel_data['x_coords']
    slices = kernel_data['observable_slices']
    diag_emul = kernel_data.get('diag_emulator_cov', np.zeros(len(x_coords)))
    diag_exp = kernel_data.get('diag_exp_cov', np.zeros(len(x_coords)))
    observable_labels = kernel_data.get('observable_labels', [f"obs_{i}" for i in range(len(slices))])

    plot_dir = os.path.join(config.output_dir, f'plot_discrepancy_group_{group_name}')
    os.makedirs(plot_dir, exist_ok=True)

    _plot_kernel_heatmap(K, x_coords, slices, plot_dir)
    _plot_uncertainties_by_observable(diag_emul, diag_exp, np.diag(K), x_coords, slices, observable_labels, plot_dir)
    _plot_block_contributions(K, slices, plot_dir)

#---------------------------------------------------------------
def load_discrepancy_kernels(file_path: str) -> dict[str, dict[str, np.ndarray]]:
    data = {}
    with h5py.File(file_path, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            labels = [s.decode() for s in group['observable_labels'][()]] if 'observable_labels' in group else []
            data[group_name] = {
                'kernel_matrix': group['kernel_matrix'][()],
                'x_coords': group['x_coords'][()],
                'observable_slices': [slice(int(s[0]), int(s[1])) for s in group['observable_slices'][()]],
                'diag_emulator_cov': group['diag_emulator_cov'][()] if 'diag_emulator_cov' in group else None,
                'diag_exp_cov': group['diag_exp_cov'][()] if 'diag_exp_cov' in group else None,
                'observable_labels': labels,
            }
    return data

#---------------------------------------------------------------
def _plot_kernel_heatmap(K, x_coords, slices, plot_dir):
    plt.figure(figsize=(8, 7))
    sns.heatmap(K, cmap='viridis')
    for slc in slices:
        plt.axhline(slc.start, color='white', linestyle='--', linewidth=0.5)
        plt.axvline(slc.start, color='white', linestyle='--', linewidth=0.5)
    plt.title("Discrepancy Kernel Matrix")
    plt.xlabel("Bin Index")
    plt.ylabel("Bin Index")
    plt.savefig(os.path.join(plot_dir, 'kernel_heatmap.pdf'))
    plt.close()

#---------------------------------------------------------------
def _plot_uncertainties_by_observable(diag_emul, diag_exp, diag_disc, x_coords, slices, labels, plot_dir):
    """
    Save one uncertainty component plot per observable.
    """
    for idx, (slc, label) in enumerate(zip(slices, labels)):
        x = x_coords[slc]
        emul = diag_emul[slc]
        exp = diag_exp[slc]
        disc = diag_disc[slc]

        # Sort by x
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        emul_sorted = emul[sort_idx]
        exp_sorted = exp[sort_idx]
        disc_sorted = disc[sort_idx]

        plt.figure(figsize=(8, 5))
        plt.plot(x_sorted, exp_sorted, linestyle='-', color='black', marker='o', label='Exp')
        plt.plot(x_sorted, emul_sorted, linestyle='--', color='blue', marker='s', fillstyle='none', label='Emul')
        plt.plot(x_sorted, disc_sorted, linestyle=':', color='red', marker='d', label='Disc')

        plt.xlabel("x (e.g. pT)")
        plt.ylabel("Variance")
        plt.title(f"Uncertainty for {label}")
        plt.legend()
        plt.tight_layout()

        fname = f'uncertainty_components_{label.replace("/", "_")}.pdf'
        plt.savefig(os.path.join(plot_dir, fname))
        plt.close()

#---------------------------------------------------------------
def _plot_block_contributions(K, slices, plot_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(K, cmap='Blues', cbar=True, ax=ax)
    for idx, slc in enumerate(slices):
        ax.axhline(slc.start, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(slc.start, color='gray', linestyle='--', linewidth=0.5)
        mid = (slc.start + slc.stop) // 2
        ax.text(mid, 0.5, f'Obs {idx}', va='bottom', ha='center', color='gray', fontsize=8)
    ax.set_title("Block Structure of Discrepancy Covariance")
    ax.set_xlabel("Bin Index")
    ax.set_ylabel("Bin Index")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'block_covariance_structure.pdf'))
    plt.close()
