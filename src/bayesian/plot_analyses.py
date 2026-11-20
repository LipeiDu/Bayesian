"""Plot across analyses

authors: J. Mulligan, R. Ehlers
"""

from __future__ import annotations
from typing import Any

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

from bayesian import data_IO, mcmc
from bayesian import plot_qhat
from bayesian.parameterization import parameterization_info
from bayesian.posterior_utils import model_posterior_samples, select_posterior_samples, maximum_posterior_sample

logger = logging.getLogger(__name__)


def plot(analyses: dict[str, Any], config_file: str, output_dir: str) -> None:
    """Plot across selected analyses

    :param dict[str, MCMCConfig] configs: dictionary of MCMCConfig objects, with keys corresponding to analysis names

    :return None: we save plots to disk
    """
    # Setup
    configs = {}
    for analysis_name, analysis_config in analyses.items():
        for parameterization in analysis_config['parameterizations']:
            configs[f"{analysis_name}_{parameterization}"] = mcmc.MCMCConfig(
                analysis_name=analysis_name,
                parameterization=parameterization,
                analysis_config=analysis_config,
                config_file=config_file
            )

    # Validation and setup
    results = {}
    posteriors = {}
    for analysis_name, config in configs.items():
        # Check if mcmc.h5 file exists
        if not os.path.exists(config.mcmc_outputfile):
            logger.info(f'MCMC output does not exist: {config.mcmc_outputfile}')
            return

        # Get results from file
        results[analysis_name] = data_IO.read_dict_from_h5(config.output_dir, config.mcmc_outputfilename, verbose=True)
        info = parameterization_info(config.analysis_config, config.parameterization)
        posteriors[analysis_name], _weights = model_posterior_samples(results[analysis_name], info)

    # Plot output dir
    plot_dir = os.path.join(output_dir, 'plot_analyses')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_qhat_across_analyses(
        results=results,
        posteriors=posteriors,
        configs=configs,
        plot_dir=plot_dir,
        E=100,
        cred_level=0.9,
        n_samples=5000,
        plot_mean=False,
    )


#---------------------------------------------------------------[]
def plot_qhat_across_analyses(
    results,
    posteriors,
    plot_dir,
    configs, E=0, T=0, cred_level=0.9, n_samples=5000, n_x=50,
    plot_prior=True, plot_mean=True, plot_map=False, target_design_point=np.array([])):
    '''
    Plot qhat credible interval from posterior samples,
    as a function of either E or T (with the other held fixed)

    Pretty much copied from plot_qhat, hastily modified due to QM.

    :param 2darray posterior: posterior samples -- shape (n_walkers*n_steps, n_params)
    :param float E: fix jet energy (GeV), and plot as a function of T
    :param float T: fix temperature (GeV), and plot as a function of E
    :param float cred_level: credible interval level
    :param int n_samples: number of posterior samples to use for plotting
    :param int n_x: number of T or E points to plot
    :param 1darray target_design_point: if closure test, design point corresponding to "truth" qhat value
    '''

    fig, ax = plt.subplots()
    colors = plt.get_cmap("tab10")
    ymax = 0.0
    qhat_closure = {}
    for index, (analysis_name, result) in enumerate(results.items()):
        posterior, config = posteriors[analysis_name], configs[analysis_name]
        samples, weights = select_posterior_samples(
            posterior, result.get("weights"), max_samples=n_samples,
            random_state=getattr(config, "random_seed", None)
        )
        color = colors(index % 10)
        x_array = np.linspace(0.16, 0.5, n_x) if E else np.linspace(5, 200, n_x)

        def evaluate(points, model_config=config, grid=x_array):
            return np.array([plot_qhat.qhat_over_T_cubed(points, model_config, T=x if E else T, E=E if E else x)
                             for x in grid])

        values = evaluate(samples)
        mean = values @ weights
        intervals = np.array([mcmc.credible_interval(row, confidence=cred_level, weights=weights) for row in values])
        ax.fill_between(x_array, intervals[:, 0], intervals[:, 1], color=color, alpha=0.25,
                        label=f"{analysis_name}: {int(cred_level*100)}% posterior interval")
        ymax = max(ymax, float(intervals[:, 1].max()))
        if plot_mean:
            ax.plot(x_array, mean, "--", color=color, label=f"{analysis_name}: mean")
        if plot_map:
            point = maximum_posterior_sample(posterior, result.get("log_target"))
            ax.plot(x_array, evaluate(np.atleast_2d(point)).reshape(-1), color=color,
                    label=f"{analysis_name}: highest-target sample")
        if plot_prior:
            prior = evaluate(plot_qhat._generate_prior_samples(config, n_samples=len(samples)))
            interval = np.array([mcmc.credible_interval(row, confidence=cred_level) for row in prior])
            ax.fill_between(x_array, interval[:, 0], interval[:, 1], color=color, alpha=0.1,
                            label=f"{analysis_name}: configured prior")
            ymax = max(ymax, float(interval[:, 1].max()))
        if target_design_point.size:
            truth = evaluate(target_design_point).reshape(-1)
            ax.plot(x_array, truth, color="black", linestyle=":")
            qhat_closure = {"qhat_closure_array": (truth >= intervals[:, 0]) & (truth <= intervals[:, 1]),
                            "qhat_mean": mean, "x_array": x_array, "cred_level": cred_level}
    suffix = f"E{E}" if E else f"T{T}"
    ax.set(xlabel="T (GeV)" if E else "E (GeV)", ylabel=r'$\hat{q}/T^3$', ylim=(0, max(ymax*1.15, 1)))
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f'{plot_dir}/qhat_{suffix}.pdf')
    plt.close(fig)
    if target_design_point.size:
        return qhat_closure
