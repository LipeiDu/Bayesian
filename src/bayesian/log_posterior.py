"""Define the likelihood separately for performance reasons

In doing so, we can use global variables. This isn't a nice thing to do from a coding perspective,
but it gives a significant improvement in MCMC performance during multiprocessing.
For the initial concept, see: https://emcee.readthedocs.io/en/stable/tutorials/parallel/#parallel

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
.. codeauthor:: James Mulligan
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy.linalg import lapack

from bayesian.emulation import base
from bayesian import prior as prior_module
from bayesian.model_discrepancy import add_discrepancy_covariance_all_groups
from bayesian.parameterization import ParameterizationInfo
from bayesian import sequential_prior
from bayesian.sequential_inference import SequentialInferenceConfig

logger = logging.getLogger(__name__)


g_min: npt.NDArray[np.float64] = None
g_max: npt.NDArray[np.float64] = None
g_emulation_config: base.EmulatorOrganizationConfig = None
g_emulation_results: dict[str, dict[str, npt.NDArray[np.float64]]] = None
g_experimental_results: dict = None
g_emulator_cov_unexplained: dict = None
g_prior_config: dict = None
g_log_prior_fn = None  # Prior callable loaded at init
g_discrepancy_config: dict = {}
g_observable_xcoords: dict = {}  # observable bin centers
g_param_names: dict[str, list[int]] = {}
g_model_param_indices: list[int] = []
g_discrepancy_param_indices: dict[str, list[int]] = {}
g_discrepancy_enabled: bool
g_full_model_param_names: list[str] = []
g_fixed_model_parameters: dict[str, float] = {}
g_sequential_inference_config: SequentialInferenceConfig | None = None
g_sequential_log_prior_fn = None

def initialize_pool_variables(local_min, local_max, local_emulation_config, local_emulation_results,local_experimental_results, local_emulator_cov_unexplained,
    local_prior_config, local_discrepancy_config, local_observable_xcoords, param_names, local_discrepancy_enabled,
    local_full_model_param_names, local_fixed_model_parameters, local_sequential_inference_config=None
) -> None:
    global g_min  # noqa: PLW0603
    global g_max  # noqa: PLW0603
    global g_emulation_config  # noqa: PLW0603
    global g_emulation_results  # noqa: PLW0603
    global g_experimental_results  # noqa: PLW0603
    global g_emulator_cov_unexplained  # noqa: PLW0603
    global g_prior_config
    global g_log_prior_fn
    global g_discrepancy_config, g_observable_xcoords, g_param_names
    global g_model_param_indices, g_discrepancy_param_indices
    global g_discrepancy_enabled
    global g_full_model_param_names, g_fixed_model_parameters
    global g_sequential_inference_config, g_sequential_log_prior_fn

    g_min = local_min
    g_max = local_max
    g_emulation_config = local_emulation_config
    g_emulation_results = local_emulation_results
    g_experimental_results = local_experimental_results
    g_emulator_cov_unexplained = local_emulator_cov_unexplained
    g_prior_config = local_prior_config
    g_discrepancy_config = local_discrepancy_config
    g_observable_xcoords = local_observable_xcoords
    g_param_names = param_names
    g_discrepancy_enabled = local_discrepancy_enabled
    g_full_model_param_names = local_full_model_param_names
    g_fixed_model_parameters = local_fixed_model_parameters
    g_sequential_inference_config = local_sequential_inference_config

    # Identify prefixes used for discrepancy parameters
    # Discrepancy parameters are named with prefix: {group}__{param}
    discrepancy_prefixes = {
        f"{group}__" for group, cfg in g_discrepancy_config.items() if cfg.get("infer_hyperparameters", False)
    }

    # Identify model parameters (those that do not start with any discrepancy prefix)
    g_model_param_indices = [
        i for i, name in enumerate(g_param_names)
        if not any(name.startswith(prefix) for prefix in discrepancy_prefixes)
    ]

    # Identify discrepancy parameters by group, using sorted indices for consistency
    g_discrepancy_param_indices = {}
    for group, cfg in g_discrepancy_config.items():
        if cfg.get("infer_hyperparameters", False):
            prefix = f"{group}__"
            indices = sorted(
                i for i, name in enumerate(g_param_names) if name.startswith(prefix)
            )
            g_discrepancy_param_indices[group] = indices
            if not indices:
                logger.warning(f"Group '{group}' has 'infer=True' but no matching parameters with prefix '{prefix}'")

    # Build the prior function
    g_log_prior_fn = prior_module.make_log_prior_fn(g_prior_config, g_param_names)
    g_sequential_log_prior_fn = sequential_prior.build_sequential_log_prior_fn(
        sequential_config=g_sequential_inference_config,
        combined_prior_config=g_prior_config,
        sampled_parameter_names=g_param_names,
        sampled_parameter_min=g_min,
        sampled_parameter_max=g_max,
    )

#---------------------------------------------------------------
def log_posterior(X, *, set_to_infinite_outside_bounds: bool = True) -> npt.NDArray[np.float64]:
    """
    Function to evaluate the log-posterior for a given set of input parameters.

    This function is called by https://emcee.readthedocs.io/en/stable/user/sampler/

    :param X input ndarray of parameter space values
    :param min list of minimum boundaries for each emulator parameter
    :param max list of maximum boundaries for each emulator parameter
    :param config emulation_configuration object
    :param emulation_results dict of emulation groups
    :param experimental_results arrays of experimental results
    """

    # Convert to 2darray of shape (n_samples, n_parameters)
    X = np.array(X, copy=False, ndmin=2)

    # Initialize log-posterior array, which we will populate and return
    log_posterior = np.zeros(X.shape[0])

    # Check if any samples are outside the parameter bounds, and set log-posterior to -inf for those
    inside = np.all((X > g_min) & (X < g_max), axis=1)  # noqa: SIM300
    # -1e300 is apparently preferred for pocoMC
    log_posterior[~inside] = -np.inf if set_to_infinite_outside_bounds else -1e300

    # Evaluate log-posterior for samples inside parameter bounds
    # n_samples: number of design points = number of training samples
    # n_features: total number of observables
    n_samples = np.count_nonzero(inside)
    n_features = g_experimental_results['y'].shape[0]

    if n_samples > 0:

        # Get experimental data
        data_y = g_experimental_results['y']
        data_y_err = g_experimental_results['y_err']

        # Compute emulator prediction
        # Returns dict of matrices of emulator predictions:
        #     emulator_predictions['central_value'] -- (n_samples, n_features)
        #     emulator_predictions['cov'] -- (n_samples, n_features, n_features)

        # LDU: The discrepancy parameters are irrelavant to emulation
        # X[inside][:, g_model_param_indices] ensures emulators only see the model parameters, excluding discrepancy parameters
        sampled_model_parameters = X[inside][:, g_model_param_indices]
        full_model_parameters = ParameterizationInfo(
            full_names=g_full_model_param_names,
            full_min=[],
            full_max=[],
            fixed_parameters=g_fixed_model_parameters,
        ).expand_sampled_to_full(
            sampled_model_parameters,
            sampled_names=[g_param_names[i] for i in g_model_param_indices],
        )
        emulator_predictions = base.predict(full_model_parameters, g_emulation_config,
                                                 emulation_group_results=g_emulation_results,
                                                 emulator_cov_unexplained=g_emulator_cov_unexplained)

        # Construct array to store the difference between emulator prediction and experimental data
        # (using broadcasting to subtract each data point from each emulator prediction)
        assert data_y.shape[0] == emulator_predictions['central_value'].shape[1]
        dY = emulator_predictions['central_value'] - data_y

        # Sanity check: emulator output should match the number of features expected from experimental data
        assert emulator_predictions['central_value'].shape[1] == n_features, (
            f"Mismatch in number of observables: emulator predicts {emulator_predictions['central_value'].shape[1]} "
            f"but expected {n_features}. This may occur if PCA is active (reducing dimension), or if observable slicing "
            f"is inconsistent. Ensure that n_pc is None for all groups if PCA is disabled, and observable filters match."
        )

        # Construct the covariance matrix
        # NOTE-STAT TODO: include full experimental data covariance matrix -- currently we only include uncorrelated data uncertainty
        #-------------------------
        covariance_matrix = np.zeros((n_samples, n_features, n_features))
        covariance_matrix += emulator_predictions['cov']
        covariance_matrix += np.diag(data_y_err**2)

        # Add model discrepancy covariance per group into each sample’s full covariance matrix.
        # This accounts for systematic model deficiencies across kinematic regions.
        # Inject discrepancy covariances per group, only if enabled
        if g_discrepancy_enabled:
            discrepancy_blocks = add_discrepancy_covariance_all_groups(
                theta=X[inside],
                discrepancy_config=g_discrepancy_config,
                observable_xcoords=g_observable_xcoords,
                emulation_config=g_emulation_config,
                param_names=g_param_names,
                n_features=n_features,
                discrepancy_param_indices=g_discrepancy_param_indices,
                discrepancy_enabled_groups=g_discrepancy_enabled,
            )

            for i in range(n_samples):
                covariance_matrix[i] += discrepancy_blocks[i]

        # Compute log likelihood at each point in the sample
        # We take constant priors, so the log-likelihood is just the log-posterior
        # (since above we set the log-posterior to -inf for samples outside the parameter bounds)
        log_posterior[inside] += list(map(_loglikelihood, dY, covariance_matrix))

        # Add prior term. In sequential mode this replaces the soft prior subspace with the learned soft posterior.
        active_log_prior_fn = g_sequential_log_prior_fn if g_sequential_log_prior_fn is not None else g_log_prior_fn
        log_posterior[inside] += active_log_prior_fn(X[inside])

        # NOTE-STAT: We don't support the extra_std term here.

    return log_posterior

#---------------------------------------------------------------
def _loglikelihood(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    The likelihood is NOT NORMALIZED, since this does not affect MCMC.
    The normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """
    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        msg = 'lapack dpotrf error: '
        msg += f'the {-info}-th argument had an illegal value'
        raise ValueError(msg)
    if info < 0:
        msg = 'lapack dpotrf error: '
        msg += f'the leading minor of order {info} is not positive definite'
        raise np.linalg.LinAlgError(msg)

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        msg = 'lapack dpotrs error: '
        msg += f'the {-info}-th argument had an illegal value'
        raise ValueError(
        )

    return -.5*np.dot(y, alpha) - np.log(L.diagonal()).sum()
