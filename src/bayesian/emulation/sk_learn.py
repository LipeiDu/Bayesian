'''
Module related to emulators, with functionality to train and call emulators for a given analysis run

The main functionalities are:
 - fit_emulators() performs PCA, fits an emulator to each PC, and writes the emulator to file
 - predict() construct mean, std of emulator for a given set of parameter values

A configuration class EmulationConfig provides simple access to emulation settings

authors: J.Mulligan, R.Ehlers
Based in part on JETSCAPE/STAT code.
'''

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import sklearn.decomposition as sklearn_decomposition
import sklearn.gaussian_process as sklearn_gaussian_process
import sklearn.preprocessing as sklearn_preprocessing
import yaml

from bayesian import common_base, data_IO
from bayesian.emulation import base as emulation_base

logger = logging.getLogger(__name__)

_register_name = "sk_learn"

####################################################################################################################
def fit_emulator_group(config: emulation_base.ConcreteEmulatorConfig) -> dict[str, Any]:
    '''
    Do PCA, fit emulators, and write to file for an individual emulation group.

    The first config.n_pc principal components (PCs) are emulated by independent Gaussian processes (GPs)
    The emulators map design points to PCs; the output will need to be inverted from PCA space to physical space.

    :param EmulationConfig config: we take an instance of EmulationConfig as an argument to keep track of config info.

    LDU, May 2025: PCA disabled case is useful when considering model discrepancy
    If PCA is enabled (config.n_pc is not None):
        - Perform PCA on the entire observable matrix.
        - Train one emulator per principal component.

    If PCA is disabled (config.n_pc is None):
        - Train one emulator per observable slice.
        - No transformation into PCA space is performed.
    '''
    # Check if emulator already exists
    if config.emulation_outputfile.exists():
        if config.force_retrain:
            config.emulation_outputfile.unlink()
            logger.info(f'Removed {config.emulation_outputfile}')
        else:
            logger.info(f'Emulators already exist: {config.emulation_outputfile} (to force retrain, set force_retrain: True)')
            return {}

    # Initialize predictions into a single 2D array: (design_point_index, observable_bins) i.e. (n_samples, n_features)
    # A consistent order of observables is enforced internally in data_IO
    # NOTE: One sample corresponds to one design point, while one feature is one bin of one observable
    Y = data_IO.predictions_matrix_from_h5(config.output_dir, filename=config.observables_filename, observable_filter=config.observable_filter)

    # Get design
    design = data_IO.design_array_from_h5(config.output_dir, filename=config.observables_filename)

    # Define GP kernel (covariance function)
    min = np.array(config.analysis_config['parameterization'][config.parameterization]['min'])
    max = np.array(config.analysis_config['parameterization'][config.parameterization]['max'])

    kernel = None
    for kernel_type, kernel_args in config.active_kernels.items():
        if kernel_type == "matern":
            length_scale = max - min
            length_scale_bounds_factor = kernel_args['length_scale_bounds_factor']
            length_scale_bounds = (np.outer(length_scale, tuple(length_scale_bounds_factor)))
            nu = kernel_args['nu']
            kernel = sklearn_gaussian_process.kernels.Matern(length_scale=length_scale,
                                                             length_scale_bounds=length_scale_bounds,
                                                             nu=nu,
                                                            )
        if kernel_type == 'rbf':
            length_scale = max - min
            length_scale_bounds_factor = kernel_args['length_scale_bounds_factor']
            length_scale_bounds = (np.outer(length_scale, tuple(length_scale_bounds_factor)))
            kernel = sklearn_gaussian_process.kernels.RBF(length_scale=length_scale,
                                                          length_scale_bounds=length_scale_bounds
                                                         )
        if kernel_type == 'constant':
            constant_value = kernel_args["constant_value"]
            constant_value_bounds = kernel_args["constant_value_bounds"]
            kernel_constant = sklearn_gaussian_process.kernels.ConstantKernel(constant_value=constant_value,
                                                                              constant_value_bounds=constant_value_bounds
                                                                             )
            kernel = (kernel + kernel_constant)
        if kernel_type == 'noise':
            kernel_noise = sklearn_gaussian_process.kernels.WhiteKernel(
                noise_level=kernel_args["args"]["noise_level"],
                noise_level_bounds=kernel_args["args"]["noise_level_bounds"],
            )
            kernel = (kernel + kernel_noise)

    # --- CASE 1: PCA is DISABLED ---
    if config.n_pc is None:
        logger.info("PCA is OFF...")
        logger.info("Training one emulator per observable slice")
        slices = config.observable_slices
        logger.info(f"Training {len(slices)} emulators")
        if slices is None:
            raise ValueError("[fit_emulator_group] PCA is off, but observable_slices is not defined.")

        emulators = []
        for s in slices:
            y_slice = Y[:, s]
            emulator = sklearn_gaussian_process.GaussianProcessRegressor(kernel=kernel,
                                                                          alpha=config.alpha,
                                                                          n_restarts_optimizer=config.n_restarts,
                                                                          copy_X_train=False)
            emulator.fit(design, y_slice)
            emulators.append(emulator)

        output_dict: dict[str, Any] = {"emulators": emulators}
        return output_dict

    # --- CASE 2: PCA is ENABLED ---
    # Use sklearn to:
    #  - Center and scale each feature (and later invert)
    #  - Perform PCA to reduce to config.n_pc features.
    #      This amounts to finding the matrix S that diagonalizes the covariance matrix C = Y.T*Y = S*D^2*S.T
    #      Or equivalently the right singular vectors S.T in the SVD decomposition of Y: Y = U*D*S.T
    #      Given S, we can transform from feature space to PCA space with: Y_PCA = Y*S
    #               and from PCA space to feature space with: Y = Y_PCA*S.T
    #
    # The input Y is a 2D array of format (n_samples, n_features).
    #
    # The output of pca.fit_transform() is a 2D array of format (n_samples, n_components),
    #   which is equivalent to:
    #     Y_pca = Y.dot(pca.components_.T), where:
    #       pca.components_ are the principal axes, sorted by decreasing explained variance -- shape (n_components, n_features)
    #     In the notation above, pca.components_ = S.T, i.e.
    #       the rows of pca.components_ are the sorted eigenvectors of the covariance matrix of the scaled features.
    #
    # We can invert this back to the original feature space by: pca.inverse_transform(Y_pca),
    #   which is equivalent to:
    #     Y_reconstructed = Y_pca.dot(pca.components_)
    #
    # Then, we still need to make sure to undo the preprocessing (centering and scaling) by:
    #     Y_reconstructed_unscaled = scaler.inverse_transform(Y_reconstructed)
    #
    # See docs for StandardScaler and PCA for further details.
    # This post explains exactly what fit_transform,inverse_transform do: https://stackoverflow.com/a/36567821
    #
    # TODO: Do we want whiten the PCs, i.e. to scale the variances of each PC to 1?
    #       I don't see a compelling reason to do this...We are fitting separate GPs to each PC,
    #       so standardizing the variance of each PC is not important.
    #       (NOTE: whitening can be done with whiten=True -- beware that inverse_transform also undoes whitening)
    logger.info('Doing PCA...')
    scaler = sklearn_preprocessing.StandardScaler()
    # This adopts the sklearn convention, but then sets a max cap of 30 PCs (arbitrarily chosen) to
    # reduce computation time.
    max_n_components = config.max_n_components_to_calculate
    if max_n_components is not None:
        logger.info(f"Running with max n_pc={max_n_components}")
    # NOTE-STAT: Whiten=True, but here, Whiten=False.
    # NOTE-STAT: RJE thinks this doesn't matter, based on the comments above.
    pca = sklearn_decomposition.PCA(n_components=max_n_components, svd_solver='full', whiten=False) # Include all PCs here, so we can access them later
    # Scale data and perform PCA
    Y_pca = pca.fit_transform(scaler.fit_transform(Y))
    Y_pca_truncated = Y_pca[:,:config.n_pc]    # Select PCs here
    # Invert PCA and undo the scaling
    Y_reconstructed_truncated = Y_pca_truncated.dot(pca.components_[:config.n_pc,:])
    Y_reconstructed_truncated_unscaled = scaler.inverse_transform(Y_reconstructed_truncated)
    explained_variance_ratio = pca.explained_variance_ratio_
    logger.info(f'  Variance explained by first {config.n_pc} components: {np.sum(explained_variance_ratio[:config.n_pc])}')

    # Fit a GP (optimize the kernel hyperparameters) to map each design point to each of its PCs
    # Note that Y_PCA=(n_samples, n_components), so each PC corresponds to a row (i.e. a column of Y_PCA.T)
    logger.info("")
    logger.info('Fitting GPs...')
    logger.info(f'  The design has {design.shape[1]} parameters')
    emulators = [sklearn_gaussian_process.GaussianProcessRegressor(kernel=kernel,
                                                             alpha=config.alpha,
                                                             n_restarts_optimizer=config.n_restarts,
                                                             copy_X_train=False).fit(design, y) for y in Y_pca_truncated.T]

    # Print hyperparameters
    logger.info("")
    logger.info('Kernel hyperparameters:')
    [logger.info(f'  {emulator.kernel_}') for emulator in emulators]  # type: ignore[func-returns-value]
    logger.info("")

    # Write all info we want to file
    output_dict: dict[str, Any] = {}
    output_dict['PCA'] = {}
    output_dict['PCA']['Y'] = Y
    output_dict['PCA']['Y_pca'] = Y_pca
    output_dict['PCA']['Y_pca_truncated'] = Y_pca_truncated
    output_dict['PCA']['Y_reconstructed_truncated'] = Y_reconstructed_truncated
    output_dict['PCA']['Y_reconstructed_truncated_unscaled'] = Y_reconstructed_truncated_unscaled
    output_dict['PCA']['pca'] = pca
    output_dict['PCA']['scaler'] = scaler
    output_dict['emulators'] = emulators

    return output_dict


####################################################################################################################
class EmulatorConfig(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, analysis_name='', parameterization='', analysis_config='', config_file='', emulation_group_name: str | None = None):

        self.analysis_name = analysis_name
        self.parameterization = parameterization
        self.analysis_config = analysis_config
        self.config_file = config_file

        with Path(self.config_file).open() as stream:
            config = yaml.safe_load(stream)

        # Observable inputs
        self.observable_table_dir = config['observable_table_dir']
        self.observable_config_dir = config['observable_config_dir']
        self.observables_filename = config["observables_filename"]

        ########################
        # Emulator configuration
        ########################
        if emulation_group_name is None:
            emulator_configuration = self.analysis_config["parameters"]["emulators"]
        else:
            emulator_configuration = self.analysis_config["parameters"]["emulators"][emulation_group_name]
        self.force_retrain = emulator_configuration['force_retrain']
        self.n_pc = emulator_configuration['n_pc']
        self.max_n_components_to_calculate = emulator_configuration.get("max_n_components_to_calculate", None)

        # Kernels
        self.active_kernels = {}
        for kernel_type in emulator_configuration['kernels']['active']:
            self.active_kernels[kernel_type] = emulator_configuration['kernels'][kernel_type]

        # Validate that we have exactly one of matern, rbf
        reference_strings = ["matern", "rbf"]
        assert sum([s in self.active_kernels for s in reference_strings]) == 1, "Must provide exactly one of 'matern', 'rbf' kernel"

        # Validation for noise configuration
        if 'noise' in self.active_kernels:
            # Check we have the appropriate keys
            assert [k in self.active_kernels['noise'] for k in ["type", "args"]], "Noise configuration must have keys 'type' and 'args'"
            if self.active_kernels['noise']["type"] == "white":
                # Validate arguments
                # We don't want to do too much since we'll just be reinventing the wheel, but a bit can be helpful.
                assert set(self.active_kernels['noise']["args"]) == set(["noise_level", "noise_level_bounds"]), "Must provide arguments 'noise_level' and 'noise_level_bounds' for white noise kernel"  # noqa: C405
            else:
                msg = "Unsupported noise kernel"
                raise ValueError(msg)

        # GPR
        self.n_restarts = emulator_configuration["GPR"]['n_restarts']
        self.alpha = emulator_configuration["GPR"]["alpha"]

        # Observable list
        # None implies a convention of accepting all available data
        self.observable_filter = None
        observable_list = emulator_configuration.get("observable_list", [])
        observable_exclude_list = emulator_configuration.get("observable_exclude_list", [])
        if observable_list or observable_exclude_list:
            self.observable_filter = data_IO.ObservableFilter(
                include_list=observable_list,
                exclude_list=observable_exclude_list,
            )

        # Output options
        self.output_dir = Path(config['output_dir']) / f'{analysis_name}_{parameterization}'
        emulation_outputfile_name = 'emulation.pkl'
        if emulation_group_name is not None:
            emulation_outputfile_name = f'emulation_group_{emulation_group_name}.pkl'
        self.emulation_outputfile = Path(self.output_dir) /  emulation_outputfile_name
