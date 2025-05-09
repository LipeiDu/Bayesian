import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from matplotlib.lines import Line2D

sns.set_context('paper', rc={'font.size': 14, 'axes.titlesize': 14, 'axes.labelsize': 14})

logger = logging.getLogger(__name__)

####################################################################################################################
def plot(config):
    """
    Master function to generate discrepancy plots for all groups with discrepancy enabled.
    """
    file_path = os.path.join(config.output_dir, 'discrepancy_kernels.h5')
    if not os.path.exists(file_path):
        logger.info(f"[Discrepancy] No discrepancy kernel file found: {file_path}")
        return

    kernel_data = load_discrepancy_kernels(file_path)

    for group_name in config.analysis_config['parameters']['emulators'].keys():
        if config.analysis_config['parameters']['emulators'][group_name].get('discrepancy', {}).get('enabled', False):
            if group_name in kernel_data:
                logger.info(f"[Discrepancy] Plotting discrepancy for group '{group_name}'...")
                plot_discrepancy(config, group_name, kernel_data[group_name])
            else:
                logger.warning(f"[Discrepancy] Group '{group_name}' enabled but no kernel saved.")

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
