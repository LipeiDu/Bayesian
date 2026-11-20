import logging
from copy import deepcopy
from importlib import import_module
from pathlib import Path

from bayesian import mcmc
from bayesian.parameterization import parameterization_info
from bayesian.prior import load_posterior_as_prior, verify_prior_vs_posterior

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
            message = f"Unknown workflow type: {workflow_type}"
            raise ValueError(message)


###################################################################################################
def _run_joint_inference(workflow_name: str, workflow_config: dict, config_file: Path, _output_dir: Path, all_analyses: dict) -> None:
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

def _multistep_prior_source(workflow_config, first, second, parameterization, output_dir, all_analyses):
    """Resolve the source per parameterization before running or plotting stage B."""
    source = workflow_config.get("prior_sources", {}).get(second)
    if source is None:
        source = workflow_config.get("prior_source", {})
    source = deepcopy(source)
    source_path = Path(source.get("posterior_file", "posterior.h5"))
    if not source_path.is_absolute():
        source_path = Path(output_dir) / f"{first}_{parameterization}" / source_path
    source["posterior_file"] = str(source_path.resolve())
    source.setdefault("method", "kde")
    first_info = parameterization_info(all_analyses[first], parameterization)
    second_info = parameterization_info(all_analyses[second], parameterization)
    # A marginal cannot stand in for a conditional posterior when a previously
    # sampled model parameter is fixed in stage B (or vice versa).
    for name in set(first_info.full_names) & set(second_info.full_names):
        first_fixed = first_info.fixed_parameters.get(name)
        second_fixed = second_info.fixed_parameters.get(name)
        if first_fixed != second_fixed:
            message = f"Two-step inference requires the same fixed value/status for shared parameter '{name}'"
            raise ValueError(message)
    source.setdefault("parameter_names", second_info.sampled_names)
    if not set(source["parameter_names"]).issubset(second_info.sampled_names):
        message = "prior_source.parameter_names must be a subset of stage B sampled model parameters"
        raise ValueError(message)
    # Validate weights, columns, names and density fit before starting a sampler.
    load_posterior_as_prior(source_path, source["method"], param_names=source["parameter_names"])
    return source


def _run_multistep_inference(workflow_name: str, workflow_config: dict, config_file: Path, output_dir: Path, all_analyses: dict) -> None:
    """Use an existing stage A posterior as the prior for stage B.

    Relative source paths are anchored at ``output_dir/<stage A>_<parameterization>``.
    Multiple parameterizations receive separate workflow output subdirectories.
    Stage A is produced by the standard analysis before this function is called.
    """
    steps = workflow_config["steps"]
    first, second = _validate_multistep_steps(steps, all_analyses)
    target = all_analyses[second]
    parameterizations = target["parameterizations"]
    workflow_dir = Path(output_dir) / (workflow_name + "_" + "_".join(steps))
    for parameterization in parameterizations:
        if parameterization not in all_analyses[first]["parameterizations"]:
            message = f"Missing stage A parameterization: {parameterization}"
            raise ValueError(message)
        source = _multistep_prior_source(
            workflow_config, first, second, parameterization, output_dir, all_analyses
        )
        analysis_config = deepcopy(target)
        parameter_config = analysis_config["parameterization"][parameterization]
        prior = parameter_config.setdefault("prior", {})
        if prior is None:
            prior = parameter_config["prior"] = {}
        prior["prior_source"] = source
        step_output_dir = workflow_dir if len(parameterizations) == 1 else workflow_dir / parameterization
        mcmc_config = mcmc.MCMCConfig(
            analysis_name=second,
            parameterization=parameterization,
            analysis_config=analysis_config,
            config_file=config_file,
            input_analysis_dir=Path(output_dir) / f"{second}_{parameterization}",
            output_dir=step_output_dir,
        )
        if mcmc_config.sequential_inference_config is not None:
            message = "Two-step posterior priors cannot be combined with sequential_inference in stage B"
            raise ValueError(message)
        if not mcmc_config.observables_file.is_file():
            raise FileNotFoundError(mcmc_config.observables_file)
        step_output_dir.mkdir(parents=True, exist_ok=True)
        if should_verify_prior(workflow_config):
            verify_prior_vs_posterior(
                posterior_file=source["posterior_file"], method=source["method"],
                param_names=source["parameter_names"],
                save_plot_to=step_output_dir / f"prior_vs_posterior_check_{parameterization}.pdf",
            )
        if workflow_config.get("run_mcmc", True):
            mcmc.run_mcmc(mcmc_config)
        if workflow_config.get("run_closure_tests", False):
            validation = analysis_config.get("validation_indices", [0, 0])
            for idx in range(validation[1] - validation[0]):
                closure_config = mcmc.MCMCConfig(
                    analysis_name=second, parameterization=parameterization, analysis_config=deepcopy(analysis_config),
                    config_file=config_file, closure_index=idx,
                    input_analysis_dir=mcmc_config.input_analysis_dir, output_dir=step_output_dir,
                )
                mcmc.run_mcmc(closure_config, closure_index=idx)
        _plot_multistep_results(workflow_config.get("plot", {}), mcmc_config)


def _validate_multistep_steps(steps, all_analyses):
    if (not isinstance(steps, list) or len(steps) != 2
            or any(not isinstance(step, str) for step in steps) or steps[0] == steps[1]):
        message = "Two-step inference requires exactly two distinct analysis names (strings)"
        raise ValueError(message)
    first, second = steps
    for name in steps:
        if name not in all_analyses:
            message = f"Unknown two-step analysis: {name}"
            raise ValueError(message)
    return first, second


def _plot_multistep_results(flags, config):
    # Optional plotting dependencies are needed only when a plot is requested.
    for flag, module in (("mcmc", "plot_mcmc"), ("qhat", "plot_qhat"), ("closure_tests", "plot_closure")):
        if flags.get(flag, False):
            import_module(f"bayesian.{module}").plot(config)
