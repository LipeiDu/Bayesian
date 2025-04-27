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
from scipy.stats import norm

from bayesian.emulation import base
from bayesian import prior as prior_module

logger = logging.getLogger(__name__)


g_min: npt.NDArray[np.float64] = None
g_max: npt.NDArray[np.float64] = None
g_emulation_config: base.EmulatorOrganizationConfig = None
g_emulation_results: dict[str, dict[str, npt.NDArray[np.float64]]] = None
g_experimental_results: dict = None
g_emulator_cov_unexplained: dict = None
g_prior_config: dict = None
g_log_prior_fn = None  # Prior callable loaded at init

def initialize_pool_variables(local_min, local_max, local_emulation_config, local_emulation_results, local_experimental_results, local_emulator_cov_unexplained, local_prior_config) -> None:
    global g_min  # noqa: PLW0603
    global g_max  # noqa: PLW0603
    global g_emulation_config  # noqa: PLW0603
    global g_emulation_results  # noqa: PLW0603
    global g_experimental_results  # noqa: PLW0603
    global g_emulator_cov_unexplained  # noqa: PLW0603
    global g_prior_config
    global g_log_prior_fn

    g_min = local_min
    g_max = local_max
    g_emulation_config = local_emulation_config
    g_emulation_results = local_emulation_results
    g_experimental_results = local_experimental_results
    g_emulator_cov_unexplained = local_emulator_cov_unexplained
    g_prior_config = local_prior_config

    # Initialize log prior function
    if g_prior_config is not None:
        if "prior_source" in g_prior_config:
            logger.info("Loading prior from posterior file via KDE or Gaussian...")
            g_log_prior_fn = prior_module.load_posterior_as_prior(
                posterior_file=g_prior_config["prior_source"]["posterior_file"],
                method=g_prior_config["prior_source"].get("method", "kde")
            )
        else:
            # if specified: uniform, gaussian, log, etc.
            g_log_prior_fn = lambda X: prior_module.log_prior(X, g_prior_config)
    else:
        # default: uniform prior
        g_log_prior_fn = lambda X: np.zeros(X.shape[0])

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
        emulator_predictions = base.predict(X[inside], g_emulation_config,
                                                 emulation_group_results=g_emulation_results,
                                                 emulator_cov_unexplained=g_emulator_cov_unexplained)

        # Construct array to store the difference between emulator prediction and experimental data
        # (using broadcasting to subtract each data point from each emulator prediction)
        assert data_y.shape[0] == emulator_predictions['central_value'].shape[1]
        dY = emulator_predictions['central_value'] - data_y

        # Construct the covariance matrix
        # NOTE-STAT TODO: include full experimental data covariance matrix -- currently we only include uncorrelated data uncertainty
        #-------------------------
        covariance_matrix = np.zeros((n_samples, n_features, n_features))
        covariance_matrix += emulator_predictions['cov']
        covariance_matrix += np.diag(data_y_err**2)

        # Compute log likelihood at each point in the sample
        # We take constant priors, so the log-likelihood is just the log-posterior
        # (since above we set the log-posterior to -inf for samples outside the parameter bounds)
        log_posterior[inside] += list(map(_loglikelihood, dY, covariance_matrix))

        # Add log prior term to support non-uniform priors; LDU Apr 21, 2025
        log_posterior[inside] += g_log_prior_fn(X[inside])

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
