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
from bayesian.data_IO import data_array_from_h5, read_dict_from_h5

sns.set_context('paper', rc={'font.size': 14, 'axes.titlesize': 14, 'axes.labelsize': 14})

logger = logging.getLogger(__name__)

####################################################################################################################
def plot(config, n_posterior_samples=3, kernel_file="discrepancy_kernels.h5"):
    """
    Master function to generate, save, and plot discrepancy kernels for both fixed and inferred (MAP + posterior) parameters.
    All plots are saved to PDF per group/sample combination.
    """
    logger.info("[Discrepancy] Starting full discrepancy kernel generation and plotting.")

    # Load base data
    experimental_results = data_IO.data_array_from_h5(
        config.input_analysis_dir, 'observables.h5', pseudodata_index=-1, observable_filter=None
    )
    emulation_config = base.EmulatorOrganizationConfig.from_config_file(
        analysis_name=config.analysis_name,
        parameterization=config.parameterization,
        analysis_config=config.analysis_config,
        config_file=config.config_file,
    )
    observable_xcoords = build_observable_xcoords_per_group(config, emulation_config, experimental_results)

    # Step 1: Generate fixed kernels (infer_hyperparameters=False)
    fixed_kernels, fixed_metadata, fixed_diag_emul, fixed_diag_exp = generate_discrepancy_kernels_from_fixed_parameters(
        config, experimental_results, observable_xcoords
    )

    # Step 2: Load posterior and draw MAP + posterior samples
    mcmc_data = data_IO.read_dict_from_h5(config.output_dir, config.mcmc_outputfilename, verbose=True)
    posterior = mcmc_data['chain'].reshape((-1, mcmc_data['chain'].shape[2]))
    log_prob = mcmc_data['log_prob'].reshape(-1)
    theta_map = posterior[np.argmax(log_prob)]
    # log_and_verify_map_kernel(config, theta_map, observable_xcoords)

    sample_idxs = np.random.choice(len(posterior), size=n_posterior_samples, replace=False)
    theta_samples = [theta_map] + [posterior[i] for i in sample_idxs]
    sample_labels = ["MAP"] + [f"Posterior_{i}" for i in range(n_posterior_samples)]

    # Step 3: Generate inferred kernels (infer_hyperparameters=True)
    sample_kernels, sample_metadata, diag_emul_group, diag_exp_group = generate_discrepancy_kernels_from_samples(
        config,
        experimental_results,
        observable_xcoords,
        theta_samples=theta_samples,
        sample_labels=sample_labels
    )

    # Step 4: Merge fixed and sampled kernels from all groups
    all_kernels = {}
    for group in set(fixed_kernels) | set(sample_kernels):
        kernels = []
        labels = []

        if group in fixed_kernels and fixed_kernels[group]:
            kernels.extend(fixed_kernels[group])
        if group in sample_kernels and sample_kernels[group]:
            kernels.extend(sample_kernels[group])

        all_kernels[group] = kernels

    # Merge metadata and diag covs
    group_metadata = sample_metadata.copy()
    for group in fixed_metadata:
        if group not in group_metadata:
            group_metadata[group] = fixed_metadata[group]
            diag_emul_group[group] = fixed_diag_emul[group]
            diag_exp_group[group] = fixed_diag_exp[group]

    # Combine sample labels
    sample_labels_per_group = {}
    for group in all_kernels:
        n = len(all_kernels[group])
        sample_labels_per_group[group] = (
            ["fixed"] + sample_labels if group in fixed_kernels else sample_labels[:n]
        )

    # Step 5: Save all kernels
    save_discrepancy_kernels_h5(
        save_path=os.path.join(config.output_dir, kernel_file),
        cached_kernels=all_kernels,
        observable_xcoords=observable_xcoords,
        group_metadata=group_metadata,
        diag_emulator_cov=diag_emul_group,
        diag_exp_cov=diag_exp_group,
        discrepancy_enabled_groups=list(all_kernels.keys()),
        sample_labels_per_group=sample_labels_per_group
    )

    # Step 6: Plot results
    plot_all_discrepancy_samples(config, kernel_file=kernel_file)

    logger.info("[Discrepancy] Completed kernel generation and plotting.")

####################################################################################################################
def plot_all_discrepancy_samples(config, kernel_file="discrepancy_kernels.h5"):
    """
    Plot diagonal entries from discrepancy kernels for all saved samples per group,
    and generate per-observable uncertainty component plots across all samples.
    """
    file_path = os.path.join(config.output_dir, kernel_file)
    if not os.path.exists(file_path):
        logger.warning(f"[Discrepancy] Kernel file not found: {file_path}")
        return

    all_kernels = load_discrepancy_kernels_multi_sample(file_path)

    for group_name, sample_dict in all_kernels.items():
        # --- Plot total diagonal overlay ---
        plt.figure(figsize=(7, 5))
        for i, (label, entry) in enumerate(sample_dict.items()):
            x = entry['x_coords']
            K = entry['kernel_matrix']
            diag = np.diagonal(K)
            color = 'red' if label.lower() == 'map' else 'black' if label.lower() == 'fixed' else 'blue'
            alpha = 1.0 if label.lower() in ['fixed', 'map'] else 0.4
            lw = 2 if label.lower() in ['fixed', 'map'] else 1
            plt.plot(x, diag, label=label, color=color, alpha=alpha, linewidth=lw)

        plt.title(f"Discrepancy Kernel Diagonal – {group_name}")
        plt.xlabel("Observable x")
        plt.ylabel("Variance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_dir = os.path.join(config.output_dir, f'plot_discrepancy_group_{group_name}')
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, "diagonal_kernel_comparison.pdf")
        plt.savefig(save_path)
        plt.close()

        # --- Per-observable uncertainty breakdown ---
        _plot_uncertainties_by_observable_multi_sample(sample_dict, group_name, config.output_dir)

####################################################################################################################
def _plot_uncertainties_by_observable_multi_sample(
    sample_dict: dict[str, dict[str, Any]],
    group_name: str,
    output_dir: str
):
    """
    Save one plot per observable slice, with multiple discrepancy samples (e.g., fixed, MAP, posterior).
    Each plot shows: experimental, emulator, and multiple discrepancy curves.
    """
    # Use any sample to get shared metadata (slices, x, etc.)
    first_entry = next(iter(sample_dict.values()))
    slices = first_entry['observable_slices']
    x_coords = first_entry['x_coords']
    labels = first_entry.get('observable_labels', [f"obs_{i}" for i in range(len(slices))])

    plot_dir = os.path.join(output_dir, f'plot_discrepancy_group_{group_name}')
    os.makedirs(plot_dir, exist_ok=True)

    for idx, (slc, label) in enumerate(zip(slices, labels)):
        x = x_coords[slc]
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]

        plt.figure(figsize=(8, 5))

        # Experimental and emulator uncertainty are the same across samples
        exp = first_entry['diag_exp_cov'][slc][sort_idx] if first_entry['diag_exp_cov'] is not None else np.zeros_like(x_sorted)
        emul = first_entry['diag_emulator_cov'][slc][sort_idx] if first_entry['diag_emulator_cov'] is not None else np.zeros_like(x_sorted)
        plt.plot(x_sorted, exp, linestyle='-', color='black', marker='o', label='Exp')
        plt.plot(x_sorted, emul, linestyle='--', color='blue', marker='s', fillstyle='none', label='Emul')

        # Loop through all samples and plot the diagonal of the discrepancy kernel
        for label_name, entry in sample_dict.items():
            K = entry['kernel_matrix']
            diag_disc = np.diagonal(K)[slc][sort_idx]
            color = 'red' if label_name.lower() == 'map' else 'black' if label_name.lower() == 'fixed' else 'blue'
            alpha = 1.0 if label_name.lower() in ['fixed', 'map'] else 0.4
            lw = 2 if label_name.lower() in ['fixed', 'map'] else 1
            plt.plot(x_sorted, diag_disc, linestyle=':', color=color, alpha=alpha, linewidth=lw, label=f'Disc ({label_name})')

        plt.xlabel("x (e.g. pT)")
        plt.ylabel("Variance")
        plt.title(f"Uncertainty for {label}")
        plt.legend()
        plt.tight_layout()

        fname = f'uncertainty_components_{label.replace("/", "_")}_multi_sample.pdf'
        plt.savefig(os.path.join(plot_dir, fname))
        plt.close()

####################################################################################################################
def generate_discrepancy_kernels_from_fixed_parameters(config, experimental_results, observable_xcoords):
    """
    Generate and return fixed discrepancy kernels for groups with infer_hyperparameters=False.
    Returns kernel dictionary, metadata, and diag covariances.
    """
    (
        param_names, parameter_min, parameter_max,
        discrepancy_config, discrepancy_enabled_groups,
        discrepancy_param_indices, model_param_indices,
        emulation_config, group_metadata,
        diag_exp_group, diag_emul_group
    ) = generate_common_discrepancy_inputs(config, experimental_results)

    theta = np.array([(lo + hi) / 2 for lo, hi in zip(parameter_min, parameter_max)])[None, :]

    cached_kernels = {}
    any_fixed = False

    for group in discrepancy_enabled_groups:
        cfg = discrepancy_config[group]
        if not cfg["infer_hyperparameters"]:
            any_fixed = True
            x_coords = observable_xcoords[group]
            kernel_params = cfg["fixed_params"]
            try:
                cov_d = build_discrepancy_covariance_matrix(x_coords, cfg["kernel_type"], kernel_params)
                cached_kernels[group] = [cov_d]
                logger.info(f"[Discrepancy] Saving fixed kernel for group {group} with fixed parameters")
            except Exception as e:
                logger.warning(f"[Discrepancy] Error building fixed kernel for group '{group}': {e}")
                cached_kernels[group] = [None]
        else:
            logger.info(f"[Discrepancy] Skipping group '{group}' with fixed parameters — hyperparameters will be inferred.")

    if not any_fixed:
        logger.info("[Discrepancy] No fixed-kernel groups found. Skipping save.")
        return {}, {}, {}, {}

    return cached_kernels, group_metadata, diag_emul_group, diag_exp_group

####################################################################################################################
def generate_discrepancy_kernels_from_samples(
    config,
    experimental_results,
    observable_xcoords,
    theta_samples: list[np.ndarray],
    sample_labels: list[str]
) -> tuple[dict[str, list[np.ndarray]], dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Generate and return discrepancy kernels for each theta sample, per group.
    Also returns metadata needed for saving.
    """
    (
        param_names, parameter_min, parameter_max,
        discrepancy_config, discrepancy_enabled_groups,
        discrepancy_param_indices, model_param_indices,
        emulation_config, group_metadata,
        diag_exp_group, diag_emul_group
    ) = generate_common_discrepancy_inputs(config, experimental_results)

    all_kernels = {g: [] for g in discrepancy_enabled_groups}
    for theta in theta_samples:
        theta = np.asarray(theta).reshape(1, -1)
        for group in discrepancy_enabled_groups:
            cfg = discrepancy_config[group]
            if not cfg["infer_hyperparameters"]:
                continue
            x_coords = observable_xcoords[group]
            try:
                theta_vals = theta[0, discrepancy_param_indices[group]]
                kernel_params = DiscrepancyKernelParams(
                    c_bar=theta_vals[0],
                    length_scale=theta_vals[1],
                    r=theta_vals[2],
                    s=theta_vals[3] if len(theta_vals) > 3 else 0.0
                )
                cov_d = build_discrepancy_covariance_matrix(x_coords, cfg["kernel_type"], kernel_params)
                all_kernels[group].append(cov_d)
            except Exception as e:
                logger.warning(f"[Discrepancy] Kernel generation failed for group {group}: {e}")
                all_kernels[group].append(None)

    return all_kernels, group_metadata, diag_emul_group, diag_exp_group

####################################################################################################################
def generate_common_discrepancy_inputs(config, experimental_results):
    """
    Shared setup for fixed and sampled discrepancy kernel generation.
    Returns:
        param_names, parameter_min, parameter_max,
        discrepancy_config, discrepancy_enabled_groups,
        discrepancy_param_indices, model_param_indices,
        emulation_config, group_metadata,
        diag_exp_group, diag_emul_group
    """
    param_cfg = config.analysis_config['parameterization'][config.parameterization]
    names = param_cfg['names']
    parameter_min = param_cfg['min']
    parameter_max = param_cfg['max']
    model_prior_config = param_cfg.get("prior", None)
    emulator_groups = config.analysis_config['parameters']['emulators']

    names, parameter_min, parameter_max, _, discrepancy_config, discrepancy_enabled_groups = parse_discrepancy_group_settings(
        emulator_groups, names, parameter_min, parameter_max, model_prior_config
    )
    param_names = names

    discrepancy_param_indices = {
        group: [i for i, name in enumerate(param_names) if name.startswith(f"{group}__")]
        for group in discrepancy_config
    }

    model_param_indices = [
        i for i, name in enumerate(param_names)
        if not any(name.startswith(f"{g}__") for g in discrepancy_config)
    ]

    emulation_config = base.EmulatorOrganizationConfig.from_config_file(
        analysis_name=config.analysis_name,
        parameterization=config.parameterization,
        config_file=config.config_file,
        analysis_config=config.analysis_config,
    )

    group_metadata = get_group_metadata(emulation_config)
    diag_exp = experimental_results['y_err'] ** 2

    # Emulator uncertainty is always evaluated at prior midpoint
    theta_mid = np.array([(lo + hi) / 2 for lo, hi in zip(parameter_min, parameter_max)])[None, :]
    emulator_predictions = base.predict(theta_mid[:, model_param_indices], emulation_config)
    diag_emul = np.diagonal(emulator_predictions['cov'][0])

    diag_exp_group = extract_groupwise_diagonals(diag_exp, group_metadata)
    diag_emul_group = extract_groupwise_diagonals(diag_emul, group_metadata)

    return (
        param_names, parameter_min, parameter_max,
        discrepancy_config, discrepancy_enabled_groups,
        discrepancy_param_indices, model_param_indices,
        emulation_config, group_metadata,
        diag_exp_group, diag_emul_group
    )

#---------------------------------------------------------------
def get_group_metadata(emulation_config):
    mapping_obj = base.SortEmulationGroupObservables.learn_mapping(emulation_config)
    group_metadata = {}
    for obs_name, (group, slc, _) in mapping_obj.emulation_group_to_observable_matrix.items():
        if group not in group_metadata:
            group_metadata[group] = {"observable_slices": [], "observable_labels": []}
        group_metadata[group]["observable_slices"].append(slc)
        group_metadata[group]["observable_labels"].append(obs_name)
    return group_metadata

def get_emulator_uncertainty_at_midpoint(config, param_min, param_max, emulation_config):
    param_cfg = config.analysis_config['parameterization'][config.parameterization]
    names = param_cfg['names']
    theta_mid = np.array([(lo + hi) / 2 for lo, hi in zip(param_min, param_max)])[None, :]
    model_indices = [i for i, name in enumerate(names) if not any(name.startswith(f"{g}__") for g in config.analysis_config['parameters']['emulators'])]
    pred = base.predict(theta_mid[:, model_indices], emulation_config)
    return np.diagonal(pred['cov'][0])

def extract_groupwise_diagonals(diag_array, group_metadata):
    diag_group = {}
    for group, meta in group_metadata.items():
        slices = meta['observable_slices']
        indices = [i for s in slices for i in range(s.start, s.stop)]
        diag_group[group] = diag_array[indices]
    return diag_group

#---------------------------------------------------------------
def log_and_verify_map_kernel(config, theta_map, observable_xcoords):
    """
    Log MAP parameters and verify that kernel generated from MAP is stable across runs.
    """
    from bayesian.model_discrepancy import (
        build_discrepancy_covariance_matrix,
        parse_discrepancy_group_settings,
        DiscrepancyKernelParams
    )

    param_cfg = config.analysis_config['parameterization'][config.parameterization]
    names = param_cfg['names']
    parameter_min = param_cfg['min']
    parameter_max = param_cfg['max']
    model_prior_config = param_cfg.get("prior", None)
    emulator_groups = config.analysis_config['parameters']['emulators']

    names, _, _, _, discrepancy_config, discrepancy_enabled_groups = parse_discrepancy_group_settings(
        emulator_groups, names, parameter_min, parameter_max, model_prior_config
    )

    param_names = names
    discrepancy_param_indices = {
        group: [i for i, name in enumerate(param_names) if name.startswith(f"{group}__")]
        for group in discrepancy_config
    }

    print("="*60)
    print("[Diagnostic] MAP parameter values used for discrepancy kernels:")
    for i, name in enumerate(param_names):
        print(f"  {name:40s} = {theta_map[i]:.6f}")
    print("="*60)

    for group in discrepancy_enabled_groups:
        cfg = discrepancy_config[group]
        if not cfg["infer_hyperparameters"]:
            continue  # fixed kernels already handled

        x_coords = observable_xcoords[group]
        idxs = discrepancy_param_indices[group]
        theta_vals = theta_map[idxs]

        kernel_params = DiscrepancyKernelParams(
            c_bar=theta_vals[0],
            length_scale=theta_vals[1],
            r=theta_vals[2],
            s=theta_vals[3] if len(theta_vals) > 3 else 0.0
        )

        kernel = build_discrepancy_covariance_matrix(x_coords, cfg["kernel_type"], kernel_params)
        diag = np.diagonal(kernel)

        print(f"[{group}] Kernel diagonal (first 10 entries): {diag[:10]}")

#---------------------------------------------------------------
def save_discrepancy_kernels_h5(
    save_path: str,
    cached_kernels: dict[str, list[np.ndarray]],
    observable_xcoords: dict[str, np.ndarray],
    group_metadata: dict[str, dict[str, Any]],
    diag_emulator_cov: Optional[dict[str, np.ndarray]] = None,
    diag_exp_cov: Optional[dict[str, np.ndarray]] = None,
    discrepancy_enabled_groups: Optional[list[str]] = None,
    sample_labels_per_group: dict[str, list[str]] = None
):
    """
    Save discrepancy kernel matrices and metadata to HDF5.
    If multiple samples exist per group, store them as subgroups.
    """
    import h5py

    logger.info(f"[Discrepancy] Saving kernels to {save_path}")

    with h5py.File(save_path, 'w') as f:
        for group_name, kernels in cached_kernels.items():
            if discrepancy_enabled_groups and group_name not in discrepancy_enabled_groups:
                continue

            labels = sample_labels_per_group[group_name]
            for i, kernel in enumerate(kernels):
                if kernel is None:
                    continue
                sample_label = labels[i]

                label_safe = sample_label.replace("/", "_")
                group_path = f"{group_name}/{label_safe}"
                group = f.create_group(group_path)

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
def load_discrepancy_kernels_multi_sample(file_path: str) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Load discrepancy kernels saved for multiple samples (fixed, MAP, posterior).
    Returns:
        dict[group_name][sample_label] → {
            'kernel_matrix', 'x_coords', 'observable_slices',
            'diag_emulator_cov', 'diag_exp_cov', 'observable_labels'
        }
    """
    data = {}

    with h5py.File(file_path, 'r') as f:
        for group_name in f.keys():
            data[group_name] = {}
            group = f[group_name]
            for sample_label in group.keys():
                g = group[sample_label]

                # Load observable labels safely
                if 'observable_labels' in g:
                    raw_labels = g['observable_labels'][()]
                    labels = [s.decode() if isinstance(s, bytes) else str(s) for s in raw_labels]
                else:
                    labels = []

                data[group_name][sample_label] = {
                    'kernel_matrix': np.array(g['kernel_matrix']),
                    'x_coords': np.array(g['x_coords']),
                    'observable_slices': [slice(int(s[0]), int(s[1])) for s in np.array(g['observable_slices'])],
                    'diag_emulator_cov': np.array(g['diag_emulator_cov']) if 'diag_emulator_cov' in g else None,
                    'diag_exp_cov': np.array(g['diag_exp_cov']) if 'diag_exp_cov' in g else None,
                    'observable_labels': labels,
                }

    return data

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
