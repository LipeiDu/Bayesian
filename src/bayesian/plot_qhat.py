'''
Module related to generating qhat plots

authors: J.Mulligan, R.Ehlers
'''

import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

from bayesian import data_IO, mcmc, plot_utils
from bayesian.emulation import base
from bayesian.parameterization import parameterization_info
from bayesian.posterior_utils import (model_posterior_samples, select_posterior_samples, maximum_posterior_sample)
from bayesian.prior import sample_prior
from bayesian.sequential import prior as sequential_prior

sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

logger = logging.getLogger(__name__)

####################################################################################################################
def _normalize_parameter_name(name: str) -> str:
    """Normalize parameter names so config labels can be matched robustly."""
    return (
        name.replace("$", "")
        .replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .replace("^", "")
        .replace("_", "")
        .replace(" ", "")
        .lower()
    )


def _alpha_s_parameter_index(config) -> int:
    """Return the model-parameter index corresponding to the qhat coupling."""
    parameter_config = config.analysis_config['parameterization'][config.parameterization]
    names = parameterization_info(config.analysis_config, config.parameterization).full_names

    configured_name = parameter_config.get("qhat_alpha_s_parameter")
    candidate_names = [configured_name] if configured_name else []
    candidate_names.extend([
        "AlphaS",
        "alpha_s",
        "alphaS",
        "alpha_s_fix",
        "alphaSfix",
        r"$\alpha_S^{\rm{fix}}$",
    ])
    normalized_candidates = {_normalize_parameter_name(name) for name in candidate_names if name}

    for index, name in enumerate(names):
        if _normalize_parameter_name(name) in normalized_candidates:
            return index

    msg = (
        f"Could not determine the alpha_s parameter for qhat plotting. "
        f"Checked names={names}. Set 'qhat_alpha_s_parameter' in the parameterization config."
    )
    raise ValueError(msg)


####################################################################################################################
def plot(config):
    '''
    Generate qhat plots, using data written to mcmc.h5 file in analysis step.
    If no file is found at expected location, no plotting will be done.

    :param MCMCConfig config: we take an instance of MCMCConfig as an argument to keep track of config info.
    '''

    # Check if mcmc.h5 file exists
    if not Path(config.mcmc_outputfile).exists():
        logger.info(f'MCMC output does not exist: {config.mcmc_outputfile}')
        return

    # Get results from file
    results = data_IO.read_dict_from_h5(config.output_dir, config.mcmc_outputfilename, verbose=True)
    parameter_info = parameterization_info(config.analysis_config, config.parameterization)
    model_samples, weights = model_posterior_samples(results, parameter_info)

    # Plot output dir
    plot_dir = Path(config.output_dir) / 'plot_qhat'
    plot_dir.mkdir(parents=True, exist_ok=True)

    # qhat plots
    plot_qhat(model_samples, plot_dir, config, E=100, cred_level=0.9, n_samples=1000, weights=weights)
    plot_qhat(model_samples, plot_dir, config, T=0.3, cred_level=0.9, n_samples=1000, weights=weights)

    # Observable sensitivity plots
    _plot_observable_sensitivity(model_samples, plot_dir, config, delta=0.1, n_samples=1000, weights=weights)

#---------------------------------------------------------------[]
def plot_qhat(posterior, plot_dir, config, E=0, T=0, cred_level=0.9, n_samples=5000, n_x=50,
              plot_prior=True, plot_mean=True, plot_map=False, target_design_point: npt.NDArray[np.int64] | None = None,
              weights=None, log_target=None):
    '''
    Plot qhat credible interval from posterior samples,
    as a function of either E or T (with the other held fixed)

    :param 2darray posterior: posterior samples -- shape (n_walkers*n_steps, n_params)
    :param float E: fix jet energy (GeV), and plot as a function of T
    :param float T: fix temperature (GeV), and plot as a function of E
    :param float cred_level: credible interval level
    :param int n_samples: number of posterior samples to use for plotting
    :param int n_x: number of T or E points to plot
    :param 1darray target_design_point: if closure test, design point corresponding to "truth" qhat value
    '''
    # Validation
    if target_design_point is None:
        target_design_point = np.array([])

    map_point = maximum_posterior_sample(posterior, log_target) if plot_map else None
    posterior_samples, sample_weights = select_posterior_samples(
        posterior, weights, max_samples=n_samples, random_state=getattr(config, 'random_seed', None)
    )
    n_samples = len(posterior_samples)

    if bool(E) == bool(T):
        message = "Specify exactly one fixed E or T for a qhat curve"
        raise ValueError(message)
    xlabel, suffix, label = ("T (GeV)", f"E{E}", f"E = {E} GeV") if E else ("E (GeV)", f"T{T}", f"T = {T} GeV")
    x_array = np.linspace(0.16, 0.5, n_x) if E else np.linspace(5, 200, n_x)

    def evaluate(points):
        return np.array([qhat_over_T_cubed(points, config, T=x if E else T, E=E if E else x) for x in x_array])

    qhat_posteriors = evaluate(posterior_samples)

    # Calculate the weighted mean even when its curve is hidden in a closure plot.
    qhat_mean = qhat_posteriors @ sample_weights
    if plot_mean:
        plt.plot(x_array, qhat_mean, sns.xkcd_rgb['denim blue'],
                linewidth=2., linestyle='--', label='Mean')

    # Plot the MAP value as well for each T or E
    if plot_map:
        qhat_map = evaluate(map_point)
        plt.plot(x_array, qhat_map, sns.xkcd_rgb['medium green'],
                linewidth=2., linestyle='--', label='Highest-target saved sample')

    # Get credible interval for each T or E
    h = [mcmc.credible_interval(qhat_values, confidence=cred_level, weights=sample_weights) for qhat_values in qhat_posteriors]
    credible_low = [i[0] for i in h]
    credible_up =  [i[1] for i in h]
    plt.fill_between(x_array, credible_low, credible_up, color=sns.xkcd_rgb['light blue'],
                     label=f'Posterior {int(cred_level*100)}% Credible Interval')

    # Plot prior as well, for comparison
    # TODO: one could also plot some type of "information gain" metric, e.g. KL divergence
    if plot_prior:

        # Generate samples
        prior_samples = _generate_prior_samples(config, n_samples=n_samples)

        qhat_priors = evaluate(prior_samples)

        # Get credible interval for each T or E
        h_prior = [mcmc.credible_interval(qhat_values, confidence=cred_level) for qhat_values in qhat_priors]
        credible_low_prior = [i[0] for i in h_prior]
        credible_up_prior =  [i[1] for i in h_prior]
        plt.fill_between(x_array, credible_low_prior, credible_up_prior, color=sns.xkcd_rgb['light blue'],
                         alpha=0.3, label=f'Prior {int(cred_level*100)}% Credible Interval')

    # If closure test: Plot truth qhat value
    # We will return a dict of info needed for plotting closure plots, including a
    #   boolean array (as a fcn of T or E) of whether the truth value is contained within credible region
    if target_design_point.size:
        qhat_truth = evaluate(target_design_point).reshape(-1)
        plt.plot(x_array, qhat_truth, sns.xkcd_rgb['pale red'],
                linewidth=2., label='Target')

        qhat_closure = {}
        qhat_closure['qhat_closure_array'] = np.array([((qhat_truth[i] < credible_up[i]) and (qhat_truth[i] > credible_low[i])) for i,_ in enumerate(x_array)]).squeeze()
        qhat_closure['qhat_mean'] = qhat_mean
        qhat_closure['x_array'] = x_array
        qhat_closure['cred_level'] = cred_level

    # Plot formatting
    plt.xlabel(xlabel)
    plt.ylabel(r'$\hat{q}/T^3$')
    ymin = 0
    if plot_mean:
        ymax = 2*max(qhat_mean)
    elif plot_map:
        ymax = 2*max(qhat_map)
    axes = plt.gca()
    #axes.set_ylim([ymin, ymax])
    axes.set_ylim([0, 12])
    plt.legend(title=f'{label}', title_fontsize=12,
               loc='upper right', fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/qhat_{suffix}.pdf')
    plt.close('all')

    if target_design_point.size:
        return qhat_closure

#---------------------------------------------------------------
def _plot_observable_sensitivity(posterior, plot_dir, config, delta=0.1, n_samples=1000, weights=None):
    '''
    Plot local sensitivity index (at posterior median parameters) for each parameter x_i to each observable O_j:
        S(x_i, O_j, delta) = 1/delta * [O_j(x_i') - O_j(x_i)] / O_j(x_i)
    where x_i'=(1+delta)x_i and delta is a fixed parameter.

    Note: this is just a normalized partial derivative, dO_j/dx_i * (x_i/O_j)

    Based on:
      - https://arxiv.org/abs/2011.01430
      - https://link.springer.com/article/10.1007/BF00547132
    '''

    # Use weighted marginal medians as a representative point
    map_parameters = mcmc.representative_parameters(posterior, weights=weights)

    # Plot sensitivity index for each parameter
    info = parameterization_info(config.analysis_config, config.parameterization)
    for i_parameter, name in enumerate(info.full_names):
        if name in info.fixed_parameters:
            continue
        _plot_single_parameter_observable_sensitivity(map_parameters, i_parameter,
                                                      plot_dir, config, delta=delta)

    # TODO: Plot sensitivity for qhat:
    #   S(qhat, O_j, delta) = 1/delta * [O_j(qhat_map') - O_j(qhat_map)] / O_j(qhat)
    # In the current qhat formulation, qhat = qhat(x_0=alpha_s_fix) only depends on x_0=alpha_s_fix.
    # So this information is already captured in the x_0 sensitivity plot above.
    # If we want to explicitly compute S(qhat), we need to evaluate the emulator at qhat_map'=(1+delta)*qhat_map.
    # In principle one should find the x_0 corresponding to (1+delta)*qhat_map.
    # For simplicity we can just evaluate x_0'=x_0(1+delta) and then redefine delta=qhat(x_0')-qhat(x_0) -- but
    #   this is exactly the same as the S(x_0) plot above, up the redefinition of delta.
    # It may nevertheless be nice to add since a plot of S(qhat) will likely be more salient to viewers.

#---------------------------------------------------------------
def _plot_single_parameter_observable_sensitivity(map_parameters, i_parameter, plot_dir, config, delta=0.1):
    '''
    Plot local sensitivity index (at posterior median parameters) for a single parameter x_i to each observable O_j:
        S(x_i, O_j, delta) = 1/delta * [O_j(x_i') - O_j(x_i)] / O_j(x_i)
    where x_i'=(1+delta)x_i and delta is a fixed parameter.

    TODO: We probably want to add higher level summary plot, e.g. take highest 5 observables for each parameter,
            or average bins over a given observable.
          Could also construct a 2D plot where color shows the sensitivity
    '''

    # Define the two parameter points we would like to evaluate
    x = map_parameters.copy()
    x_prime = map_parameters.copy()
    x_prime[i_parameter] = (1+delta)*x_prime[i_parameter]
    x = np.expand_dims(x, axis=0)
    x_prime = np.expand_dims(x_prime, axis=0)
    parameter_info = parameterization_info(config.analysis_config, config.parameterization)
    parameter_names_for_plot = (
        parameter_info.full_names
        if x.shape[1] == len(parameter_info.full_names)
        else parameter_info.sampled_names
    )
    x = parameter_info.expand_sampled_to_full(x, sampled_names=parameter_names_for_plot)
    x_prime = parameter_info.expand_sampled_to_full(x_prime, sampled_names=parameter_names_for_plot)

    # Get emulator predictions at the two points
    emulation_config = base.EmulatorOrganizationConfig.from_config_file(
        analysis_name=config.analysis_name,
        parameterization=config.parameterization,
        analysis_config=config.analysis_config,
        config_file=config.config_file,
    )
    emulation_results = emulation_config.read_all_emulator_groups()
    emulator_predictions_x = base.predict(x, emulation_config, emulation_group_results=emulation_results)
    emulator_predictions_x_prime = base.predict(x_prime, emulation_config, emulation_group_results=emulation_results)

    # Convert to dict: emulator_predictions[observable_label]
    observables = data_IO.read_dict_from_h5(config.input_analysis_dir, 'observables.h5', verbose=False)
    emulator_predictions_x_dict = data_IO.observable_dict_from_matrix(emulator_predictions_x['central_value'],
                                                                      observables, observable_filter=emulation_config.observable_filter)
    emulator_predictions_x_prime_dict = data_IO.observable_dict_from_matrix(emulator_predictions_x_prime['central_value'],
                                                                            observables, observable_filter=emulation_config.observable_filter)

    # Construct dict of sensitivity index, in same format as emulator_predictions['central_value']
    sensitivity_index_dict = emulator_predictions_x_prime_dict['central_value'].copy()
    sorted_observable_list = data_IO.sorted_observable_list_from_dict(observables)
    for observable_label in sorted_observable_list:
        x = emulator_predictions_x_dict['central_value'][observable_label]
        x_prime = emulator_predictions_x_prime_dict['central_value'][observable_label]
        sensitivity_index_dict[observable_label] = 1/delta * (x_prime - x) / x

    # Plot
    plot_list = [sensitivity_index_dict]
    columns = [0]
    labels = [rf'Sensitivity index at posterior median, $\delta={delta}$']
    colors = [sns.xkcd_rgb['dark sky blue']]
    param_label = parameter_names_for_plot[i_parameter]
    param = param_label[1:-1].replace('{', '{{').replace('}', '}}') if param_label.startswith("$") else param_label
    ylabel = rf'$S({param}, \mathcal{{O}}, \delta)$'
    #ylabel = rf'$S({param}, \mathcal{{O}}, \delta) = \frac{{1}}{{\delta}} \frac{{\mathcal{{O}}([1+\delta] {param})-\mathcal{{O}}({param})}}{{\mathcal{{O}}({param})}}$'
    filename = f'sensitivity_index_{i_parameter}.pdf'
    plot_utils.plot_observable_panels(plot_list, labels, colors, columns, config, plot_dir, filename,
                                      linewidth=1, ymin=-5, ymax=5, ylabel=ylabel, plot_exp_data=False, bar_plot=True)

#---------------------------------------------------------------
def _running_alpha_s(mu_square: float | npt.NDArray[np.float64], alpha_s: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
    """ Running alpha_s for HTL-qhat

    Extracted from MATTER:
    https://github.com/JETSCAPE/JETSCAPE/blob/935b69291f0fd319f42dc6a9fb5960a4f814e16f/src/jet/Matter.cc#L3944-L3953

    We have a separate implementation verified by the theorists:
    https://github.com/FHead/PhysicsJetScape/blob/c3c9adfeee72e1f9ce34728e174e35ca8a70065b/JetRAAPaper/26363_HPPaperPlots/QHat.h#L10-L19

    Note:
        lambda_square_QCD_HTL is determined using alpha^fix_s such that the running alpha_s
        coincide with alpha^fix_s at scale mu^2= 1 GeV^2.

    Args:
        mu_square: Virtuality of the parton.
        alpha_s: Coupling constant (here, this is will be alpha^fix_s).

    Returns:
        float: running alpha_s
    """
    if mu_square <= 1.0:
        return alpha_s

    active_flavor: Final[int] = 3
    square_lambda_QCD_HTL = np.exp(-12 * np.pi / ((33 - 2 * active_flavor) * alpha_s))
    return 12 * np.pi / ((33 - 2 * active_flavor) * np.log(mu_square / square_lambda_QCD_HTL))


#---------------------------------------------------------------
def qhat_over_T_cubed(posterior_samples, config, T=0, E=0) -> float:
    '''
    Evaluate qhat/T^3 from posterior samples of parameters,
    for fixed E and T

    See: https://github.com/FHead/PhysicsJetScape/blob/c3c9adfeee72e1f9ce34728e174e35ca8a70065b/JetRAAPaper/26363_HPPaperPlots/QHat.h#L21-L35
    (which itself is derived from the MATTER code in jetscape).

    :param 2darray parameters: posterior samples of parameters -- shape (n_samples, n_params)
    :return 1darray: qhat/T^3 -- shape (n_samples,)
    '''

    if posterior_samples.ndim == 1:
        posterior_samples = np.expand_dims(posterior_samples, axis=0)

    if config.parameterization == "exponential":

        # Inputs
        alpha_s_fix = posterior_samples[:, _alpha_s_parameter_index(config)]
        # Constants
        active_flavor: Final[int] = 3
        # The JETSCAPE framework calculates qhat using the gluon Casimir factor, but
        # we by convention we typically report the quark qhat value, so we need to use
        # the quark Casimir factor.
        C_a: Final[float] = 4.0 / 3.0

        # From GeneralQhatFunction
        debye_mass_square = alpha_s_fix * 4 * np.pi * np.power(T, 2.0) * (6.0 + active_flavor) / 6
        # This is the virtuality of the parton
        # See info from Abhijit here: https://jetscapeworkspace.slack.com/archives/C025X5NE9SN/p1648404101376299
        # as well as Yi's code:
        # https://github.com/FHead/PhysicsJetScape/blob/c3c9adfeee72e1f9ce34728e174e35ca8a70065b/JetRAAPaper/26363_HPPaperPlots/QHat.h#L21-L35
        scale_net = np.maximum(2 * E * T, 1.0)

        running_alpha_s = _running_alpha_s(scale_net, alpha_s_fix)
        answer = (C_a * 50.4864 / np.pi) * running_alpha_s * alpha_s_fix * np.abs(np.log(scale_net / debye_mass_square))

        # If we wanted to return just qhat (rather than qhat/T^3), we could use the following conversion:
        #return answer * 0.19732698   # 1/GeV to fm
        # qhat/T^3 is dimensionless, so we don't need to convert units
        return answer  # noqa: RET504

    msg = f"qhat_over_T_cubed not implemented for parameterization: {config.parameterization}"
    raise RuntimeError(msg)

#---------------------------------------------------------------
def _generate_prior_samples(config, n_samples=100):
    '''
    Generate samples of prior parameters

    The prior is uniform in the parameter space -- except for c1,c2,c3 it is the log that is uniform.

    :param 2darray parameters: posterior samples of parameters -- shape (n_samples, n_params)
    :return 2darray: samples -- shape (n_samples,n_params)
    '''
    info = parameterization_info(config.analysis_config, config.parameterization)
    settings = config.analysis_config['parameterization'][config.parameterization]
    prior = info.filter_prior_config(settings.get('prior'))
    rng = np.random.default_rng(getattr(config, 'random_seed', None))
    sequential = sequential_prior.build_sequential_log_prior_fn(
        sequential_config=getattr(config, 'sequential_inference_config', None), combined_prior_config=prior,
        sampled_parameter_names=info.sampled_names, sampled_parameter_min=info.sampled_min,
        sampled_parameter_max=info.sampled_max,
    )
    if sequential is not None:
        draws = sequential.sample(n_samples, random_state=rng)
    else:
        draws = sample_prior(prior, info.sampled_names, info.sampled_min, info.sampled_max, n_samples, rng)
    return info.expand_sampled_to_full(draws)
