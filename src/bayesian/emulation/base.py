'''
Module related to emulators, with functionality to train and call emulators for a given analysis run

The main functionalities are:
 - fit_emulators() performs PCA, fits an emulator to each PC, and writes the emulator to file
 - predict() construct mean, std of emulator for a given set of parameter values

A configuration class EmulationConfig provides simple access to emulation settings

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
Based in part on JETSCAPE/STAT code.
'''

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import attrs
import numpy as np
import numpy.typing as npt
import yaml

from bayesian import common_base, data_IO, register_modules

logger = logging.getLogger(__name__)

_emulators: dict[str, ModuleType] = {}


def _validate_emulator(name: str, module: Any) -> None:
    """
    Validate that an emulator module follows the expected interface.
    """
    if not hasattr(module, "fit_emulator_group"):
        msg = f"Emulator module {name} does not have a required 'fit_emulator_group' method"
        raise ValueError(msg)
    # TODO: Re-enable when things stablize a bit.
    # if not hasattr(module, "predict"):
    #     msg = f"Emulator module {name} does not have a required 'predict' method"
    #     raise ValueError(msg)

def fit_emulators(emulation_config: EmulatorOrganizationConfig) -> None:
    """ Do PCA, fit emulators, and write to file.

    :param EmulationConfig config: Configuration for the emulators, including all groups.
    """
    # Fit the emulator for each emulation group
    emulator_groups_output = {}

    for emulation_group_name, emulation_group_config in emulation_config.emulation_groups_config.items():
        # Use emulator_name from the config of the group
        emulator_name = getattr(emulation_group_config, "emulator_name", "sk_learn")

        try:
            emulator = _emulators[emulator_name]
        except KeyError as e:
            raise KeyError(f"Emulator backend '{emulator_name}' not registered or available") from e

        logger.info(f"Fitting emulator for group '{emulation_group_name}' using backend '{emulator_name}'")

        emulator_groups_output[emulation_group_name] = emulator.fit_emulator_group(emulation_group_config)
        # NOTE: If it returns early because an emulator already exists, then we don't want to overwrite it!
        if emulator_groups_output[emulation_group_name]:
            write_emulators(config=emulation_group_config, output_dict=emulator_groups_output[emulation_group_name])
    # NOTE: We store everything in a dict so we can later return these if we decide it's helpful. However,
    #       it doesn't appear to be at the moment (August 2023), so we leave as is.


def predict_from_emulator(
    parameters: npt.NDArray[np.float64],
    emulation_config: EmulatorOrganizationConfig,
    merge_predictions_over_groups: bool = True,
    emulation_group_results: dict[str, dict[str, Any]] | None = None,
    emulator_cov_unexplained: dict | None = None
) -> dict[str, npt.NDArray[np.float64]]:
    # Call from MCMC
    ...

def predict(
    parameters: npt.NDArray[np.float64],
    emulation_config: EmulatorOrganizationConfig,
    *,
    merge_predictions_over_groups: bool = True,
    emulation_group_results: dict | None = None,
    emulator_cov_unexplained: dict | None = None) -> dict[str, npt.NDArray[np.float64]]:
    """
    Construct dictionary of emulator predictions for each observable

    :param ndarray[float] parameters: list of parameter values (e.g. [tau0, c1, c2, ...]), with shape (n_samples, n_parameters)
    :param EmulationConfig emulation_config: configuration object for the overall emulator (including all groups)
    :param bool merge_predictions_over_groups: whether to merge predictions over emulation groups (True)
                                               or return a dictionary of predictions for each group (False). Default: True
    :param dict emulator_group_results: dictionary containing results from each emulation group. If None, read from file.
    :param dict emulator_cov_unexplained: dictionary containing the unexplained variance due to PC truncation for each emulation group.
                                          Generally we will precompute this in mcmc.py to save time,
                                          but if it is not precomputed (e.g. when plotting) we automatically compute it here.
    :return dict emulator_predictions: dictionary containing matrices of central values and covariance
    """
    if emulation_group_results is None:
        emulation_group_results = {}
    if emulator_cov_unexplained is None:
        emulator_cov_unexplained = {}

    predict_output = {}
    for emulation_group_name, emulation_group_config in emulation_config.emulation_groups_config.items():
        emulation_group_result = emulation_group_results.get(emulation_group_name)
        # Only load the emulator group directly from file if needed. If called frequently
        # (eg. in the MCMC), it's probably better to load it once and pass it in.
        # NOTE: I know that get() can provide a second argument as the default, but a quick check showed that
        #       `read_emulators` was executing far more than expected (maybe trying to determine some default value?).
        #       However, separating it out like this seems to avoid the issue, but better to just avoid the issue.
        if emulation_group_result is None:
            emulation_group_result = read_emulators(emulation_group_config)

        # Compute unexplained variance due to PC truncation for this emulator group, if not already precomputed
        if emulator_cov_unexplained:
            emulator_group_cov_unexplained = emulator_cov_unexplained[emulation_group_name]
        else:
            emulator_group_cov_unexplained = compute_emulator_group_cov_unexplained(emulation_group_config, emulation_group_result)

        predict_output[emulation_group_name] = predict_emulation_group(
            parameters,
            emulation_group_result,
            emulation_group_config,
            emulator_group_cov_unexplained=emulator_group_cov_unexplained
        )

    # Allow the option to return immediately to allow the study of performance per emulation group
    if not merge_predictions_over_groups:
        return predict_output

    # Now, we want to merge predictions over groups
    return emulation_config.sort_observables_in_matrix.convert(group_matrices=predict_output)


def predict_emulation_group(parameters, results, emulation_group_config, emulator_group_cov_unexplained: npt.NDArray[np.float64] | None = None):
    '''
    Construct dictionary of emulator predictions for each observable in an emulation group.

    :param ndarray[float] parameters: list of parameter values (e.g. [tau0, c1, c2, ...]), with shape (n_samples, n_parameters)
    :param str results: dictionary that stores emulator

    :return dict emulator_predictions: dictionary containing matrices of central values and covariance

    Note: One can easily construct a dict of predictions with format emulator_predictions[observable_label]
          from the returned matrix as follows (useful for plotting / troubleshooting):
              observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5', verbose=False)
              emulator_predictions = data_IO.observable_dict_from_matrix(emulator_central_value_reconstructed,
                                                                         observables,
                                                                         cov=emulator_cov_reconstructed,
                                                                         validation_set=validation_set)
    '''

    # The emulators are stored as a list (one for each PC)
    emulators = results['emulators']

    if emulator_group_cov_unexplained is None:
        emulator_group_cov_unexplained = compute_emulator_group_cov_unexplained(emulation_group_config, results)

    # Get predictions (in PC space) from each emulator and concatenate them into a numpy array with shape (n_samples, n_PCs)
    # Note: we just get the std rather than cov, since we are interested in the predictive uncertainty
    #       of a given point, not the correlation between different sample points.
    n_samples = parameters.shape[0]

    # Case 1: PCA is OFF (n_pc is None)
    # When PCA is disabled, we emulate each observable slice directly in observable space.
    if emulation_group_config.n_pc is None:
        observable_slices = emulation_group_config.observable_slices

        if observable_slices is None:
            raise RuntimeError(f"[predict_emulation_group] observable_slices is not set for group '{emulation_group_config.base_config.emulation_group_name}'")

        # When PCA is off, the number of emulators should match the number of observable_slices (actual observables)
        if len(emulators) != len(observable_slices):
            raise ValueError(
                f"Mismatch between number of emulators ({len(emulators)}) and observable slices ({len(observable_slices)}) "
                f"in group '{emulation_group_config.base_config.emulation_group_name}'"
            )

        n_features = sum(s.stop - s.start for s in observable_slices)
        emulator_central_value = np.zeros((n_samples, n_features))
        emulator_cov = np.zeros((n_samples, n_features, n_features))

        for i, emulator in enumerate(emulators):
            y_mean, y_std = emulator.predict(parameters, return_std=True)

            if y_mean.ndim == 1:
                y_mean = y_mean[:, np.newaxis]
                y_std = y_std[:, np.newaxis]

            start = observable_slices[i].start
            stop = observable_slices[i].stop
            width = stop - start

            if y_mean.shape != (n_samples, width):
                raise ValueError(
                    f"Emulator output shape mismatch in group '{emulation_group_config.base_config.emulation_group_name}':\n"
                    f"Expected shape: ({n_samples}, {width}) from slice {start}:{stop}, but got {y_mean.shape}"
                )

            emulator_central_value[:, start:stop] = y_mean

            for j in range(n_samples):
                emulator_cov[j, start:stop, start:stop] = np.diag(y_std[j] ** 2)

        return {'central_value': emulator_central_value, 'cov': emulator_cov}

    # Case 2: PCA is ON
    emulator_central_value = np.zeros((n_samples, emulation_group_config.n_pc))
    emulator_variance = np.zeros((n_samples, emulation_group_config.n_pc))
    for i,emulator in enumerate(emulators):
        y_central_value, y_std = emulator.predict(parameters, return_std=True) # Alternately: return_cov=True
        emulator_central_value[:,i] = y_central_value
        emulator_variance[:,i] = y_std**2
    # Construct (diagonal) covariance matrix from the variances, for use in uncertainty propagation
    emulator_cov = np.apply_along_axis(np.diagflat, 1, emulator_variance)
    assert emulator_cov.shape == (n_samples, emulation_group_config.n_pc, emulation_group_config.n_pc)

    # Reconstruct the physical space from the PCs, and invert preprocessing.
    # Note we use array broadcasting to calculate over all samples.
    pca = results['PCA']['pca']
    scaler = results['PCA']['scaler']
    emulator_central_value_reconstructed_scaled = emulator_central_value.dot(pca.components_[:emulation_group_config.n_pc,:])
    emulator_central_value_reconstructed = scaler.inverse_transform(emulator_central_value_reconstructed_scaled)

    # Propagate uncertainty through the linear transformation back to feature space.
    # Note that for a vector f = Ax, the covariance matrix of f is C_f = A C_x A^T.
    #   (see https://en.wikipedia.org/wiki/Propagation_of_uncertainty)
    #   (Note also that even if C_x is diagonal, C_f will not be)
    # In our case, we have Y[i].T = S*Y_PCA[i].T for each point i in parameter space, where
    #    Y[i].T is a column vector of features -- shape (n_features,)
    #    Y_PCA[i].T is a column vector of corresponding PCs -- shape (n_pc,)
    #    S is the transfer matrix described above -- shape (n_features, n_pc)
    # So C_Y[i] = S * C_Y_PCA[i] * S^T.
    # Note: should be equivalent to: https://github.com/jdmulligan/STAT/blob/master/src/emulator.py#L145
    # TODO: one can make this faster with broadcasting/einsum
    # TODO: NOTE-STAT: Compare this more carefully with STAT L286 and on.
    n_features = pca.components_.shape[1]
    S = pca.components_.T[:,:emulation_group_config.n_pc]
    emulator_cov_reconstructed_scaled = np.zeros((n_samples, n_features, n_features))
    for i_sample in range(n_samples):
        emulator_cov_reconstructed_scaled[i_sample] = S.dot(emulator_cov[i_sample].dot(S.T))
    assert emulator_cov_reconstructed_scaled.shape == (n_samples, n_features, n_features)

    # Include predictive variance due to truncated PCs.
    # See comments in mcmc.py for further details.
    for i_sample in range(n_samples):
        emulator_cov_reconstructed_scaled[i_sample] += emulator_group_cov_unexplained / n_samples

    # Propagate uncertainty: inverse preprocessing
    # We only need to undo the unit variance scaling, since the shift does not affect the covariance matrix.
    # We can do this by computing an outer product (i.e. product of each pairwise scaling),
    #   and multiplying each element of the covariance matrix by this.
    scale_factors = scaler.scale_
    emulator_cov_reconstructed = emulator_cov_reconstructed_scaled*np.outer(scale_factors, scale_factors)

    # Return the stacked matrices:
    #   Central values: (n_samples, n_features)
    #   Covariances: (n_samples, n_features, n_features)
    emulator_predictions = {}
    emulator_predictions['central_value'] = emulator_central_value_reconstructed
    emulator_predictions['cov'] = emulator_cov_reconstructed

    return emulator_predictions


def read_emulators(config: EmulatorConfig) -> dict[str, Any]:
    """
    Read emulators from file.
    """
    # Validation
    filename = Path(config.emulation_outputfile)

    with filename.open("rb") as f:
        results: dict[str, Any] = pickle.load(f)
    return results


def write_emulators(config: EmulatorConfig, output_dict: dict[str, Any]) -> None:
    """
    Write emulators stored in a result from `fit_emulator_group` to file.
    """
    # Validation
    filename = Path(config.emulation_outputfile)

    with filename.open('wb') as f:
        pickle.dump(output_dict, f)

class EmulatorConfig(Protocol):
    """
    Protocol for an emulator configuration.
    """
    emulator_name: str
    base_config: EmulatorBaseConfig
    settings: dict[str, Any]


@attrs.define
class ConcreteEmulatorConfig:
    emulator_name: str
    base_config: EmulatorBaseConfig
    settings: dict[str, Any]
    observable_slices: list[slice] = attrs.field(factory=list)

    @property
    def observable_filter(self) -> data_IO.ObservableFilter | None:
        observable_list = self.settings.get("observable_list", [])
        observable_exclude_list = self.settings.get("observable_exclude_list", [])
        if observable_list or observable_exclude_list:
            return data_IO.ObservableFilter(
                include_list=observable_list,
                exclude_list=observable_exclude_list,
            )
        return None

    # Expose base_config fields
    @property
    def emulation_outputfile(self) -> Path:
        return self.base_config.emulation_outputfile

    @property
    def output_dir(self) -> Path:
        return self.base_config.emulation_outputfile.parent

    @property
    def observables_filename(self) -> str:
        return self.base_config.observables_filename

    @property
    def observable_table_dir(self) -> Path | str:
        return self.base_config.observables_table_dir

    @property
    def observable_config_dir(self) -> Path | str:
        return self.base_config.observables_config_dir

    @property
    def config(self) -> dict[str, Any]:
        return self.base_config.config

    @property
    def analysis_config(self) -> dict[str, Any]:
        return self.base_config.analysis_config

    @property
    def parameterization(self) -> str:
        return self.base_config.parameterization

    @property
    def analysis_name(self) -> str:
        return self.base_config.analysis_name

    @property
    def n_pc(self) -> int:
        return self.settings["n_pc"]

    @property
    def max_n_components_to_calculate(self) -> int | None:
        return self.settings.get("max_n_components_to_calculate", None)

    @property
    def force_retrain(self) -> bool:
        return self.settings.get("force_retrain", False)

    @property
    def active_kernels(self) -> dict[str, Any]:
        return self.settings["kernels"]

    @property
    def n_restarts(self) -> int:
        return self.settings["GPR"]["n_restarts"]

    @property
    def alpha(self) -> float:
        return self.settings["GPR"]["alpha"]


@attrs.define
class EmulatorBaseConfig:
    """
    Base configuration for an emulator.

    Store this class in your specialized emulator config class.
    Composition is preferred to inheritance.
    """
    emulator_name: str
    analysis_name: str
    parameterization: str
    config_file: Path = attrs.field(converter=Path)
    analysis_config: dict[str, Any] = attrs.field(factory=dict)
    emulation_group_name: str | None = None  # <-- optional, passed from higher-level config
    config: dict[str, Any] = attrs.field(init=False)
    observables_table_dir: Path | str = attrs.field(init=False)
    observables_config_dir: Path | str = attrs.field(init=False)
    observables_filename: str = attrs.field(init=False)
    emulation_outputfile: Path = attrs.field(init=False)

    def __attrs_post_init__(self):
        """
        Post-creation customization of the emulator configuration.
        """
        with Path(self.config_file).open() as stream:
            config = yaml.safe_load(stream)

        # Observable inputs
        self.config = config
        self.observables_table_dir = config['observable_table_dir']
        self.observables_config_dir = config['observable_config_dir']
        self.observables_filename = config["observables_filename"]

        # Build the output directory
        output_dir = Path(config['output_dir']) / f'{self.analysis_name}_{self.parameterization}'

        # Choose file name based on group name
        if self.emulation_group_name:
            emulation_outputfile_name = f'emulation_group_{self.emulation_group_name}.pkl'
        else:
            emulation_outputfile_name = 'emulation.pkl'

        self.emulation_outputfile = output_dir / emulation_outputfile_name

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> EmulatorBaseConfig:
        """
        Initialize the emulator configuration from a config file.
        """
        c = cls(
            emulator_name=config['emulator_name'],
            analysis_name=config['analysis_name'],
            parameterization=config['parameterization'],
            config_file=config['config_file'],
            emulation_group_name=config.get('emulation_group_name', None),
        )
        return c


@attrs.define
class EmulatorOrganizationConfig(common_base.CommonBase):
    """
    Configuration for an emulator.
    """
    analysis_name: str
    parameterization: str
    config_file: Path = attrs.field(converter=Path)
    analysis_config: dict[str, Any] = attrs.field(factory=dict)
    emulation_groups_config: dict[str, EmulatorConfig] = attrs.field(factory=dict)
    config: dict[str, Any] = attrs.field(init=False)
    observable_table_dir: Path | str = attrs.field(init=False)
    observable_config_dir: Path | str = attrs.field(init=False)
    observables_filename: str = attrs.field(init=False)
    output_dir: Path = attrs.field(init=False)
    # Optional objects that may provide useful additional functionality
    _observable_filter: data_IO.ObservableFilter | None = attrs.field(init=False, default=None)
    _sort_observables_in_matrix: SortEmulationGroupObservables | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        """
        Post-creation customization of the emulation configuration.
        """
        with self.config_file.open() as stream:
            self.config = yaml.safe_load(stream)

        # Retrieve parameters from the config
        # Observables
        self.observable_table_dir = self.config['observable_table_dir']
        self.observable_config_dir = self.config['observable_config_dir']
        self.observables_filename = self.config["observables_filename"]
        # I/O
        self.output_dir = Path(self.config['output_dir']) / f'{self.analysis_name}_{self.parameterization}'

    @classmethod
    def from_config_file(cls, analysis_name: str, parameterization: str, config_file: Path, analysis_config: dict[str, Any]):
        """
        Initialize the emulation configuration from a config file.
        """
        c = cls(
            analysis_name=analysis_name,
            parameterization=parameterization,
            config_file=config_file,
            analysis_config=analysis_config,
        )
        # Initialize the config for each emulation group
        c.emulation_groups_config = {
            k: ConcreteEmulatorConfig(
                emulator_name = group_cfg.get("emulator_name", "sk_learn"),
                base_config=EmulatorBaseConfig(
                    emulator_name=c.analysis_config["parameters"]["emulators"][k]["emulator_name"],
                    analysis_name=c.analysis_name,
                    parameterization=c.parameterization,
                    config_file=c.config_file,
                    analysis_config=c.analysis_config,
                    emulation_group_name=k,
                ),
                settings = c.analysis_config["parameters"]["emulators"][k]
            )
            for k, group_cfg in analysis_config["parameters"]["emulators"].items()
        }

        # Learn mapping to initialize observable_slices
        # - Determine how each observable bin maps to the full matrix
        # - Figure out where each group’s observables live in the full output array
        # - Return a mapping object (sorter) that can help in matrix construction
        sorter = SortEmulationGroupObservables.learn_mapping(c)
        c._sort_observables_in_matrix = sorter

        # Assign observable_slices to each group config
        # For each observable, assign its slice (a range of indices) to the right group config
        # These slices are used only if PCA is disabled, to emulate each slice separately
        for observable_key, (group_name, _, slice_in_group) in sorter.emulation_group_to_observable_matrix.items():
            group_config = c.emulation_groups_config[group_name]
            if not hasattr(group_config, "observable_slices"):
                group_config.observable_slices = []
            group_config.observable_slices.append(slice_in_group)

        return c

    def read_all_emulator_groups(self) -> dict[str, dict[str, npt.NDArray[np.float64]]]:
        """ Read all emulator groups.

        Just a convenience function.
        """
        emulation_results = {}
        for emulation_group_name, emulation_group_config in self.emulation_groups_config.items():
            emulation_results[emulation_group_name] = read_emulators(emulation_group_config)
        return emulation_results

    @property
    def observable_filter(self) -> data_IO.ObservableFilter:
        if self._observable_filter is None:
            if not self.emulation_groups_config:
                msg = "Need to specify emulation groups to provide an observable filter"
                raise ValueError(msg)
            # Accumulate the include and exclude lists from all emulation groups
            include_list: list[str] = []
            exclude_list: list[str] = self.config.get("global_observable_exclude_list", [])
            for emulation_group_config in self.emulation_groups_config.values():
                group_filter = emulation_group_config.observable_filter
                if group_filter:
                    include_list.extend(group_filter.include_list)  # type: ignore[union-attr]
                    exclude_list.extend(group_filter.exclude_list)  # type: ignore[union-attr]

            self._observable_filter = data_IO.ObservableFilter(
                include_list=include_list,
                exclude_list=exclude_list,
            )
        return self._observable_filter

    @property
    def sort_observables_in_matrix(self) -> SortEmulationGroupObservables:
        if self._sort_observables_in_matrix is None:
            if not self.emulation_groups_config:
                msg = "Need to specify emulation groups to provide an sorting for observable group observables"
                raise ValueError(msg)
            # Accumulate the include and exclude lists from all emulation groups
            self._sort_observables_in_matrix = SortEmulationGroupObservables.learn_mapping(self)
        return self._sort_observables_in_matrix


@attrs.define
class SortEmulationGroupObservables:
    """ Class to track and convert between emulation group matrices to match sorted observables.

    emulation_group_to_observable_matrix: Mapping from emulation group matrix to the matrix of observables. Format:
        {observable_name: (emulator_group_name, slice in output_matrix, slice in emulator_group_matrix)}
    shape: Shape of matrix output. Format: (n_design_points, n_features). Note that we may only be predicting
        one design point at a time, so we pick out the number of design points for the output based on the provided
        group outputs (which implicitly contains the required number of design points).
    available_value_types: Available value types in the group matrices. These will be extracted when the mapping is learned.
    """
    emulation_group_to_observable_matrix: dict[str, tuple[str, slice, slice]]
    shape: tuple[int, int]
    _available_value_types: set[str] | None = attrs.field(init=False, default=None)

    @classmethod
    def learn_mapping(cls, emulation_config: EmulatorOrganizationConfig) -> SortEmulationGroupObservables:
        """ Construct this object by learning the mapping from the emulation group prediction matrices to the sorted and merged matrices.

        :param EmulationConfig emulation_config: Configuration for the emulator(s).
        :return: Constructed object.
        """
        # NOTE: This could be configurable (eg. for validation). However, we don't seem to immediately
        #       need this functionality, so we'll omit it for now.
        prediction_key = "Prediction"

        # Now we need the mapping from emulator groups to observables with the right indices.
        # First, we need to start with all available observables (beyond just what's in any given group)
        # to learn the entire mapping
        # NOTE: It doesn't matter what observables file we use here since it's just to find all of the observables which are used.
        all_observables = data_IO.read_dict_from_h5(emulation_config.output_dir, 'observables.h5')
        current_position = 0
        observable_slices = {}
        for observable_key in data_IO.sorted_observable_list_from_dict(all_observables[prediction_key]):
            n_bins = all_observables[prediction_key][observable_key]['y'].shape[0]
            observable_slices[observable_key] = slice(current_position, current_position + n_bins)
            current_position += n_bins

        # Now, take advantage of the ordering in the emulator groups. (ie. the ordering in the group
        # matrix is consistent with the order of the observable names).
        observable_emulation_group_map = {}
        for emulation_group_name, emulation_group_config in emulation_config.emulation_groups_config.items():
            emulation_group_observable_keys = data_IO.sorted_observable_list_from_dict(all_observables[prediction_key], observable_filter=emulation_group_config.observable_filter)
            current_group_bin = 0
            for observable_key in emulation_group_observable_keys:
                observable_slice = observable_slices[observable_key]
                observable_emulation_group_map[observable_key] = (
                    emulation_group_name,
                    observable_slice,
                    slice(current_group_bin, current_group_bin + (observable_slice.stop - observable_slice.start))
                )
                current_group_bin += (observable_slice.stop - observable_slice.start)
                logger.debug(f"{observable_key=}, {observable_emulation_group_map[observable_key]=}, {current_group_bin=}")
        logger.debug(f"Sorted order: {observable_slices=}")

        # And then finally put them in the proper sorted observable order
        observable_emulation_group_map = {
            k: observable_emulation_group_map[k]
            for k in observable_slices
        }

        # We want the shape to allow us to preallocate the array:
        # Default shape: (n_design_points, n_features)
        last_observable = list(observable_slices)[-1]
        shape = (all_observables[prediction_key][observable_key]['y'].shape[1], observable_slices[last_observable].stop)
        logger.debug(f"{shape=} (note: for all design points)")

        return cls(
            emulation_group_to_observable_matrix=observable_emulation_group_map,
            shape=shape,
        )

    def convert(self, group_matrices: dict[str, dict[str, npt.NDArray[np.float64]]]) -> dict[str, npt.NDArray[np.float64]]:
        """ Convert a matrix to match the sorted observables.

        :param group_matrices: Matrixes to convert by emulation group. eg:
            {"group_1": {"central_value": np.array, "cov": [...]}, "group_2": np.array}
        :return: Converted matrix for each available value type.
        """
        if self._available_value_types is None:
            self._available_value_types = set([  # noqa: C403
                value_type
                for group in group_matrices.values()
                for value_type in group
            ])

        output = {}
        # Requires special handling since we're adding matrices (ie. 3d rather than 2d)
        if "cov" in self._available_value_types:
            # Setup
            value_type = "cov"

            # We have to sort them according to the mapping that we've derived.
            # However, it's not quite as trivial to just insert them (as we do for the central values),
            # so we'll use the output matrix slice as the key to sort by below.
            inputs_for_block_diag = {}
            for observable_name, (emulation_group_name, slice_in_output_matrix, slice_in_emulation_group_matrix) in self.emulation_group_to_observable_matrix.items():  # noqa: B007
                emulation_group_matrix = group_matrices[emulation_group_name]
                # NOTE: The slice_in_output_matrix.start should provide unique integers to sort by
                #       (basically, we just use the starting position instead of inserting it directly).
                inputs_for_block_diag[slice_in_output_matrix.start] = emulation_group_matrix[value_type][:, slice_in_emulation_group_matrix, slice_in_emulation_group_matrix]

            # And then merge them together in a block diagonal, sorting to put them in the right order
            output[value_type] = nd_block_diag(
                # sort based on the start value of the slice in the output matrix.
                [
                    # NOTE: We don't want to pass the key, but we need it for sorting, so we then
                    #       have to explicitly select the actual matrices (ie. the v of the k, v pair)
                    #       to pass along.
                    m[1]
                    for m in sorted(
                        inputs_for_block_diag.items(), key=lambda x: x[0]
                    )
                ]
            )

        # Handle the other values (as of 14 August 2023, it's just "central_value")
        for value_type in self._available_value_types:
            # Skip over "cov" since we handled it explicitly above.
            if value_type == "cov":
                continue

            # Since the number of design points that we want to predict varies, we can't define the output
            # until we can extract it from one group output. So we wait to initialize the output matrix until
            # we have the first group output.
            output[value_type] = None
            for observable_name, (emulation_group_name, slice_in_output_matrix, slice_in_emulation_group_matrix) in self.emulation_group_to_observable_matrix.items():  # noqa: B007
                emulation_group_matrix = group_matrices[emulation_group_name]
                if output[value_type] is None:
                    output[value_type] = np.zeros((emulation_group_matrix[value_type].shape[0], *self.shape[1:]))
                output[value_type][:, slice_in_output_matrix] = emulation_group_matrix[value_type][:, slice_in_emulation_group_matrix]

        return output


def nd_block_diag(arrays):
    """ Add 2D matrices into a block diagonal matrix in n-dimensions.

    See: https://stackoverflow.com/q/62384509

    :param arrays list[np.array]: List of arrays to block diagonalize.
    """
    shapes = np.array([i.shape for i in arrays])

    out = np.zeros(np.append(np.amax(shapes[:,:-2],axis=0), [shapes[:,-2].sum(), shapes[:,-1].sum()]))
    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes[:,-2:]):
        out[..., r:r + rr, c:c + cc] = arrays[i]
        r += rr
        c += cc

    return out


def compute_emulator_cov_unexplained(emulation_config, emulation_results) -> dict:
    """
    Compute the predictive variance due to PC truncation, for all emulator groups.
    See further details in compute_emulator_group_cov_unexplained().
    """
    emulator_cov_unexplained = {}
    if not emulation_results:
        emulation_results = emulation_config.read_all_emulator_groups()
    for emulation_group_name, emulation_group_config in emulation_config.emulation_groups_config.items():
        emulation_group_result = emulation_results.get(emulation_group_name)
        emulator_cov_unexplained[emulation_group_name] = compute_emulator_group_cov_unexplained(emulation_group_config, emulation_group_result)
    return emulator_cov_unexplained


def compute_emulator_group_cov_unexplained(emulation_group_config, emulation_group_result):
    '''
    Compute the predictive variance due to PC truncation, for a given emulator group.
    We can do this by decomposing the original covariance in feature space:
      C_Y = S D^2 S^T
          = S_{<=n_pc} D^2_{<=n_pc} S_{<=n_pc}^T + S_{>n_pc} D^2_{>n_pc} S_{>n_pc}^T
    In general, we want to estimate the covariance as a function of theta.
    We can do this for the first term by estimating it with the emulator covariance constructed above,
      as a function of theta.
    We can't do this with the second term, since we didn't emulate it -- so we estimate it,
      treating it as independent of theta, and add it to the emulator covariance:
        Sigma_unexplained = 1/n_samples * S_{>n_pc} D^2_{>n_pc} S_{>n_pc}^T,
      where we will include the 1/n_samples factor to account for the fact that we are estimating the covariance from a set of samples.
    See eqs 21-22 of https://arxiv.org/pdf/2102.11337.pdf
    TODO: double check this (and compare to https://github.com/jdmulligan/STAT/blob/master/src/emulator.py#L145)

    We will generally pre-compute this once in mcmc.py to save time, although we define this function
    here to allow us to re-compute it as needed if it is not pre-computed (e.g. when plotting).

    LDU, May 2025: If PCA is disabled, we return a zero matrix of shape (n_features, n_features).
    '''
    if emulation_group_config.n_pc is None:
        # PCA is disabled, return zero covariance
        slices = emulation_group_config.observable_slices
        n_features = sum(s.stop - s.start for s in slices)
        return np.zeros((n_features, n_features))

    # PCA is enabled, compute unexplained covariance from truncated PCs
    # TODO: NOTE-STAT: Compare this more carefully with STAT L145 and on.
    pca = emulation_group_result['PCA']['pca']
    S_unexplained = pca.components_.T[:,emulation_group_config.n_pc:]
    D_unexplained = np.diag(pca.explained_variance_[emulation_group_config.n_pc:])
    emulator_cov_unexplained = S_unexplained.dot(D_unexplained.dot(S_unexplained.T))

    # NOTE-STAT: bayesian-inference does not include a small term for numerical stability
    return emulator_cov_unexplained  # noqa: RET504


# Actually perform the discovery and registration of the emulators
if not _emulators:
    _emulators.update(
        register_modules.discover_and_register_modules(
            calling_module_name=__name__,
            required_attributes=[],
            validation_function=_validate_emulator,
        )
    )
