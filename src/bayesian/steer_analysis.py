'''
Main script to steer Bayesian inference studies for heavy-ion jet analysis

authors: J.Mulligan, R.Ehlers
Based in part on JETSCAPE/STAT code.
'''

import argparse
import logging
import os
import shutil
import yaml
from pathlib import Path

from bayesian import data_IO, preprocess_input_data, mcmc
from bayesian import plot_input_data, plot_emulation, plot_mcmc, plot_qhat, plot_closure, plot_analyses, plot_discrepancy

from bayesian import common_base, helpers
from bayesian.emulation import base
from bayesian.inference_workflows import process_inference_workflow  # Module for joint and multistep inference

logger = logging.getLogger(__name__)


####################################################################################################################
class SteerAnalysis(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file='', **kwargs):

        # Initialize config file
        self.config_file = config_file
        self.initialize()

        logger.info(self)

    #---------------------------------------------------------------
    # Initialize config
    #---------------------------------------------------------------
    def initialize(self):
        logger.info('Initializing class objects')

        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.output_dir = config['output_dir']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Data inputs
        self.observable_table_dir = config['observable_table_dir']
        self.observable_config_dir = config['observable_config_dir']

        # Configure which functions to run
        self.initialize_observables = config['initialize_observables']
        self.preprocess_input_data = config['preprocess_input_data']
        self.fit_emulators = config['fit_emulators']
        self.run_mcmc = config['run_mcmc']
        self.run_closure_tests = config['run_closure_tests']
        self.plot = config['plot']

        self.skip_standard_analyses = config.get("skip_standard_analyses", False)
        self.skip_inference_workflows = config.get("skip_inference_workflows", False)

        # Configuration of different analyses
        self.analyses = config['analyses']
        self.inference_workflows = config.get('inference_workflows', {})

    #---------------------------------------------------------------
    # Main function
    #---------------------------------------------------------------
    def run_analysis(self):
        # Add logging to file
        _root_log = logging.getLogger()
        _root_log.addHandler(logging.FileHandler(os.path.join(self.output_dir, 'steer_analysis.log'), 'w'))

        # Also write analysis config to shared directory
        shutil.copy(self.config_file, Path(self.output_dir) / "steer_analysis_config.yaml")

        # Loop through each analysis
        with helpers.progress_bar() as progress:
            # Standard individual analyses
            if not self.skip_standard_analyses:
                self._run_standard_analyses(progress)
                self._run_plotting()

            # Combined analyses, involving multiple standard analyses
            if self.inference_workflows and not self.skip_inference_workflows:
                self._run_inference_workflows(progress)

    #---------------------------------------------------------------
    # Standard Bayesian inference
    def _run_standard_analyses(self, progress):

        if not self.analyses:
            logger.info("No standard analyses found in 'analyses'. Skipping standard inference...")
            return

        analysis_task = progress.add_task("[deep_sky_blue1]Running analysis...", total=len(self.analyses))

        # Loop through each standard analysis
        for analysis_name, analysis_config in self.analyses.items():

            # Loop through the parameterizations
            parameterization_task = progress.add_task("[deep_sky_blue2]parameterization", total=len(analysis_config['parameterizations']))
            for parameterization in analysis_config['parameterizations']:
                if self.initialize_observables:
                    self._initialize_observables(progress, analysis_name, analysis_config, parameterization)
                if self.preprocess_input_data:
                    self._preprocess_data(progress, analysis_name, analysis_config, parameterization)
                if self.fit_emulators:
                    self._fit_emulators(progress, analysis_name, analysis_config, parameterization)
                if self.run_mcmc:
                    self._run_mcmc(progress, analysis_name, analysis_config, parameterization)
                if self.run_closure_tests:
                    self._run_closure(progress, analysis_name, analysis_config, parameterization)
                progress.update(parameterization_task, advance=1)
            # Hide once we're done!
            progress.update(parameterization_task, visible=False)
            progress.update(analysis_task, advance=1)

    #---------------------------------------------------------------
    # Combined analyses, such as two-step inference and joint calibration
    def _run_inference_workflows(self, progress):
        process_inference_workflow(
            config_file=self.config_file,
            output_dir=self.output_dir,
            workflows=self.inference_workflows,
            all_analyses=self.analyses
        )

    #---------------------------------------------------------------
    # Task functions
    #---------------------------------------------------------------
    # Initialize design points, predictions, data, and uncertainties
    # We store them in a dict and write/read it to HDF5
    def _initialize_observables(self, progress, name, config, param):
        # Just indicate that it's working
        task = progress.add_task("[deep_sky_blue4]Initializing...", total=None)
        progress.start_task(task)
        logger.info("")
        logger.info('========================================================================')
        logger.info(f"Initializing model: {name} ({param} parameterization)...")
        observables = data_IO.initialize_observables_dict_from_tables(self.observable_table_dir, config, param)
        data_IO.write_dict_to_h5(observables, os.path.join(self.output_dir, f'{name}_{param}'), filename='observables.h5')
        progress.update(task, advance=100, visible=False)

    def _preprocess_data(self, progress, name, config, param):
        # Just indicate that it's working
        task = progress.add_task("[deep_sky_blue4]Preprocessing...", total=None)
        progress.start_task(task)
        logger.info("")
        logger.info('------------------------------------------------------------------------')
        logger.info(f"Preprocessing input data: {name} ({param} parameterization)...")
        preprocessing_config = preprocess_input_data.PreprocessingConfig(
            analysis_name=name,
            parameterization=param,
            analysis_config=config,
            config_file=self.config_file
        )
        # NOTE: Strictly speaking, we don't want the emulation config here. However,
        #       We often need the observable filter, and it doesn't cost anything to
        #       construct here, so we just go for it.
        #emulation_config = emulation.EmulationConfig.from_config_file(
        #    analysis_name=analysis_name,
        #    parameterization=parameterization,
        #    analysis_config=analysis_config,
        #    config_file=self.config_file,
        #)
        observables_smoothed = preprocess_input_data.preprocess(preprocessing_config=preprocessing_config)
        data_IO.write_dict_to_h5(observables_smoothed, os.path.join(self.output_dir, f'{name}_{param}'), filename='observables_preprocessed.h5')
        progress.update(task, advance=100, visible=False)

    # Fit emulators and write them to file
    def _fit_emulators(self, progress, name, config, param):
        # Just indicate that it's working
        task = progress.add_task("[deep_sky_blue4]Emulating...", total=None)
        progress.start_task(task)
        logger.info("")
        logger.info('------------------------------------------------------------------------')
        logger.info(f"Fitting emulators for {name}_{param}...")
        emulation_config = base.EmulatorOrganizationConfig.from_config_file(
            analysis_name=name,
            parameterization=param,
            analysis_config=config,
            config_file=self.config_file
        )
        base.fit_emulators(emulation_config)
        progress.update(task, advance=100, visible=False)

    # Run MCMC
    def _run_mcmc(self, progress, name, config, param):
        # Just indicate that it's working
        task = progress.add_task("[deep_sky_blue4]Running MCMC...", total=None)
        progress.start_task(task)
        logger.info("")
        logger.info('------------------------------------------------------------------------')
        logger.info(f"Running MCMC for {name}_{param}...")
        mcmc_config = mcmc.MCMCConfig(
            analysis_name=name,
            parameterization=param,
            analysis_config=config,
            config_file=self.config_file
        )
        mcmc.run_mcmc(mcmc_config)
        progress.update(task, advance=100, visible=False)

    # Run closure tests -- one for each validation design point
    #   - Use validation point as pseudodata
    #   - Use emulator already trained on training points
    def _run_closure(self, progress, name, config, param):
        # Just indicate that it's working
        n_points = config['validation_indices'][1] - config['validation_indices'][0]
        task = progress.add_task("[deep_sky_blue4]Running closure tests...", total=n_points)
        progress.start_task(task)
        logger.info("")
        logger.info('------------------------------------------------------------------------')
        for idx in range(n_points):
            logger.info(f'Closure test: {name}_{param}, index={idx}')
            mcmc_config = mcmc.MCMCConfig(
                analysis_name=name,
                parameterization=param,
                analysis_config=config,
                config_file=self.config_file,
                closure_index=idx
            )
            mcmc.run_mcmc(mcmc_config, closure_index=idx)
            progress.update(task, advance=1)
        progress.update(task, visible=False)

    def _run_plotting(self):
        # Plots for individual analysis
        for analysis_name, analysis_config in self.analyses.items():
            for parameterization in analysis_config['parameterizations']:

                if any(self.plot.values()):
                    logger.info('========================================================================')
                    logger.info(f'Plotting for {analysis_name} ({parameterization} parameterization)...')
                    logger.info("")

                if self.plot["input_data"]:
                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Plotting input data for {analysis_name}_{parameterization}...')
                    emulation_config = base.EmulatorOrganizationConfig.from_config_file(
                        analysis_name=analysis_name,
                        parameterization=parameterization,
                        config_file=self.config_file,
                        analysis_config=analysis_config)
                    plot_input_data.plot(emulation_config)
                    logger.info(f'Done!')
                    logger.info("")

                if self.plot['emulators']:
                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Plotting emulators for {analysis_name}_{parameterization}...')
                    emulation_config = base.EmulatorOrganizationConfig.from_config_file(
                        analysis_name=analysis_name,
                        parameterization=parameterization,
                        config_file=self.config_file,
                        analysis_config=analysis_config)
                    plot_emulation.plot(emulation_config)
                    logger.info(f'Done!')
                    logger.info("")

                if self.plot['mcmc']:
                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Plotting MCMC for {analysis_name}_{parameterization}...')
                    mcmc_config = mcmc.MCMCConfig(analysis_name=analysis_name,
                                                parameterization=parameterization,
                                                analysis_config=analysis_config,
                                                config_file=self.config_file)
                    plot_mcmc.plot(mcmc_config)
                    logger.info(f'Done!')
                    logger.info("")

                if self.plot['qhat']:
                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Plotting qhat results {analysis_name}_{parameterization}...')
                    mcmc_config = mcmc.MCMCConfig(analysis_name=analysis_name,
                                                parameterization=parameterization,
                                                analysis_config=analysis_config,
                                                config_file=self.config_file)
                    plot_qhat.plot(mcmc_config)
                    logger.info(f'Done!')
                    logger.info("")

                if self.plot['closure_tests']:
                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Plotting closure test results {analysis_name}_{parameterization}...')
                    mcmc_config = mcmc.MCMCConfig(analysis_name=analysis_name,
                                                parameterization=parameterization,
                                                analysis_config=analysis_config,
                                                config_file=self.config_file)
                    plot_closure.plot(mcmc_config)
                    logger.info(f'Done!')
                    logger.info("")

                if self.plot.get('discrepancy', False):
                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Plotting discrepancy effects {analysis_name}_{parameterization}...')
                    mcmc_config = mcmc.MCMCConfig(analysis_name=analysis_name,
                                                parameterization=parameterization,
                                                analysis_config=analysis_config,
                                                config_file=self.config_file)
                    plot_discrepancy.plot(mcmc_config)
                    logger.info(f'Done!')
                    logger.info("")

        # Plots across multiple analyses
        if self.plot['across_analyses']:
            # NOTE: This is a departure from the standard API, but we need a convention for how
            #       to pass multiple analyses, so we'll just go with it for now.
            plot_analyses.plot(self.analyses, self.config_file, self.output_dir)


####################################################################################################################
if __name__ == '__main__':
    helpers.setup_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Jet Bayesian Analysis')
    parser.add_argument('-c', '--configFile',
                        help='Path of config file for analysis',
                        action='store', type=str,
                        default='../config/jet_substructure.yaml', )
    args = parser.parse_args()

    logger.info('Configuring...')
    logger.info(f'  configFile: {args.configFile}')

    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        msg = f'File {args.configFile} does not exist! Exiting!'
        logger.info(msg)
        raise ValueError(msg)

    analysis = SteerAnalysis(config_file=args.configFile)
    analysis.run_analysis()
