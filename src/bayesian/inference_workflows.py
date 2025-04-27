import logging
import os
from pathlib import Path
import shutil
from bayesian import mcmc, plot_mcmc, plot_qhat, plot_closure
from bayesian.emulation import base
from bayesian.prior import verify_prior_vs_posterior

logger = logging.getLogger(__name__)


###################################################################################################
def process_inference_workflow(config_file: Path, output_dir: Path, workflows: dict[str, dict], all_analyses: dict) -> None:
    """
    Process extended inference workflows like joint or multistep inference.

    Args:
        config_file (Path): Path to the main configuration YAML.
        output_dir (Path): Output directory.
        workflows (dict): YAML block under 'inference_workflows'.
        all_analyses (dict): All analysis definitions from the 'analyses' section.
    """
    for workflow_name, workflow_config in workflows.items():

        workflow_type = workflow_config['type']
        logger.info('========================================================================')
        logger.info(f"Running inference workflow: {workflow_name} [type={workflow_type}]")

        if workflow_type == "joint":
            _run_joint_inference(workflow_name, workflow_config, config_file, output_dir, all_analyses)

        elif workflow_type == "multistep":
            _run_multistep_inference(workflow_name, workflow_config, config_file, output_dir, all_analyses)

        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")


###################################################################################################
def _run_joint_inference(workflow_name: str, workflow_config: dict, config_file: Path, output_dir: Path, all_analyses: dict) -> None:
    """
    Run joint calibration by combining all observables from specified analyses.
    TO DO: not tested; not finished
    """
    combined_analysis_name = workflow_name
    sub_analysis_names = workflow_config['analyses']

    # Create combined analysis config
    combined_analysis_config = {
        "parameterizations": all_analyses[sub_analysis_names[0]]['parameterizations'],
        "parameterization": all_analyses[sub_analysis_names[0]].get("parameterization", "exponential"),
        "parameters": {
            "preprocessing": all_analyses[sub_analysis_names[0]]['parameters']['preprocessing'],
            "emulators": {},
            "mcmc": all_analyses[sub_analysis_names[0]]['parameters']['mcmc'],
            "closure": all_analyses[sub_analysis_names[0]]['parameters']['closure'],
        },
        "cuts": {},
        "plot_panel_shapes": []
    }

    for sub_name in sub_analysis_names:
        sub_analysis = all_analyses[sub_name]
        emulators = sub_analysis['parameters']['emulators']
        for group_name, group_cfg in emulators.items():
            tag = f"{sub_name}__{group_name}"
            combined_analysis_config['parameters']['emulators'][tag] = group_cfg

    logger.info(f"Constructed combined analysis config for: {combined_analysis_name}")

    # Run standard MCMC
    for parameterization in combined_analysis_config['parameterizations']:
        mcmc_config = mcmc.MCMCConfig(
            analysis_name=combined_analysis_name,
            parameterization=parameterization,
            analysis_config=combined_analysis_config,
            config_file=config_file
        )
        mcmc.run_mcmc(mcmc_config)


###################################################################################################
def should_verify_prior(workflow_config: dict) -> bool:
    """Decide whether to verify prior vs posterior based on workflow config."""
    return workflow_config.get('verify_prior_vs_posterior', False)

def _run_multistep_inference(workflow_name: str, workflow_config: dict, config_file: Path, output_dir: Path, all_analyses: dict) -> None:
    """
    Run multistep inference where each step optionally uses posterior from the previous step.

    Args:
        all_analyses (dict): All analysis definitions from the 'analyses' section.
    """

    steps = workflow_config['steps']

    # Check if the length of steps is not 2; currently only support two-step inference
    if len(steps) != 2:
        print("Error: The number of steps must be exactly 2. The current workflow doesn't support steps more than 2 ...")
        sys.exit(1)

    # Create an output directory for the full workflow
    step_output_dir = Path(
        output_dir,
        workflow_name + "_" + "_".join(steps)
    )
    step_output_dir.mkdir(parents=True, exist_ok=True)

    # Loop over the steps in the multistep inference
    for i, step_analysis_name in enumerate(steps):
        analysis_name = step_analysis_name
        # Config of a standard analysis corresponding to analysis_name
        analysis_config = all_analyses[analysis_name]

        logger.info("")
        logger.info('------------------------------------------------------------------------')
        logger.info(f"\nRunning step {i+1}/{len(steps)}: {analysis_name}")

        # Verify previous posterior as prior in multistep inference by comparing sampled posterior vs loaded prior
        if i == len(steps) - 2 and should_verify_prior(workflow_config):
            for parameterization in analysis_config['parameterizations']:
                verify_prior_vs_posterior(
                    posterior_file=Path(output_dir) / f"{analysis_name}_{parameterization}" / "posterior.h5",
                    method=workflow_config.get("prior_sources", {}).get(steps[-1], {}).get("method", "kde"),
                    save_plot_to=step_output_dir / f"prior_vs_posterior_check_{parameterization}.pdf",
                )

        # Only execute the final step
        if i < len(steps) - 1:
            logger.info(f"Skipping step {analysis_name}: already completed via standard analyses.")
            continue

        # Update config if prior_source is specified
        if isinstance(step_analysis_name, dict):
            raise ValueError("Expected 'steps' to be a list of analysis names (strings), not dicts.")

        if 'prior_source' in workflow_config:
            logger.info(f"Using posterior from previous step as prior in {analysis_name}")
            analysis_config = analysis_config.copy()
            analysis_config['parameters'] = analysis_config['parameters'].copy()
            analysis_config['parameters']['prior_source'] = workflow_config['prior_source']

        for parameterization in analysis_config['parameterizations']:

            # Override output folder for workflow
            analysis_config['output_dir'] = step_output_dir

            mcmc_config = mcmc.MCMCConfig(
                analysis_name=analysis_name,
                parameterization=parameterization,
                analysis_config=analysis_config,
                config_file=config_file,
                # Reused inputs (read from base analysis repo): emulation.pkl, observables.h5 and posterior.h5 (when used as prior for the next step)
                input_analysis_dir=Path(output_dir) / f"{analysis_name}_{parameterization}",
                # New outputs (written to workflow repo): mcmc.h5, posterior.h5 (sampled from this run), mcmc_sampler.pkl
                output_dir=step_output_dir
            )

            # Run MCMC if enabled
            if workflow_config.get("run_mcmc", True):
                mcmc.run_mcmc(mcmc_config)

            # Run closure tests if enabled
            if workflow_config.get("run_closure_tests", False):
                n_points = analysis_config.get('validation_indices', [0, 0])
                for idx in range(n_points[1] - n_points[0]):
                    closure_config = mcmc.MCMCConfig(
                        analysis_name=analysis_name,
                        parameterization=parameterization,
                        analysis_config=analysis_config,
                        config_file=config_file,
                        closure_index=idx,
                        input_analysis_dir=Path(output_dir) / f"{analysis_name}_{parameterization}",
                        output_dir=step_output_dir
                    )
                    mcmc.run_mcmc(closure_config, closure_index=idx)

            assert mcmc_config.input_analysis_dir.exists(), "Input directory does not exist"
            assert mcmc_config.observables_file.exists(), f"Missing observables.h5 in {mcmc_config.input_analysis_dir}"

            # Run plotting if enabled
            plot_flags = workflow_config.get("plot", {})
            if plot_flags.get("mcmc", False):
                plot_mcmc.plot(mcmc_config)
            if plot_flags.get("qhat", False):
                plot_qhat.plot(mcmc_config)
            if plot_flags.get("closure_tests", False):
                plot_closure.plot(mcmc_config)