"""Run a sequential posterior using a selected subset of emulator groups.

This entry point reuses the emulator and observable artifacts from an existing
analysis while writing the new MCMC products to a separate directory. It is
intended for controlled hadron-only and jet-only comparisons without copying
large analysis artifacts or maintaining nearly identical YAML files.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import yaml

from bayesian import mcmc

logger = logging.getLogger(__name__)


def select_emulator_groups(analysis_config: dict, included_groups: list[str]) -> dict:
    """Return a copied analysis configuration containing only requested groups."""
    selected = copy.deepcopy(analysis_config)
    available = selected["parameters"]["emulators"]
    unknown = set(included_groups) - set(available)
    if unknown:
        msg = f"Unknown emulator groups {sorted(unknown)}; available groups are {sorted(available)}."
        raise ValueError(msg)
    if not included_groups:
        msg = "At least one emulator group must be selected."
        raise ValueError(msg)
    selected["parameters"]["emulators"] = {
        name: available[name] for name in included_groups
    }
    return selected


def _default_input_dir(config: dict, analysis_name: str, parameterization: str) -> Path:
    return Path(config["output_dir"]) / f"{analysis_name}_{parameterization}"


def _default_output_dir(
    analysis_name: str,
    parameterization: str,
    included_groups: list[str],
) -> Path:
    subset = "_and_".join(name.removesuffix("_group") for name in included_groups)
    return Path("output") / f"{analysis_name}_sequential_pocomc_kde_{subset}_only" / (
        f"{analysis_name}_{parameterization}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/reduced_soft_hard_sequential_pocomc.yaml"),
    )
    parser.add_argument("--analysis-name", default="reduced_soft_hard")
    parser.add_argument("--parameterization", default="exponential")
    parser.add_argument(
        "--include-groups",
        nargs="+",
        required=True,
        help="Emulator group names to include, for example hadron_group or jet_group.",
    )
    parser.add_argument(
        "--input-analysis-dir",
        type=Path,
        help="Directory containing the existing emulators and observables.h5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to receive mcmc.h5, posterior.h5, and the pocoMC state.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved run without starting pocoMC.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    if args.analysis_name not in config["analyses"]:
        msg = f"Analysis {args.analysis_name!r} is not present in {args.config}."
        raise ValueError(msg)

    analysis_config = select_emulator_groups(
        config["analyses"][args.analysis_name],
        args.include_groups,
    )
    if args.parameterization not in analysis_config["parameterizations"]:
        msg = (
            f"Parameterization {args.parameterization!r} is not enabled for "
            f"analysis {args.analysis_name!r}."
        )
        raise ValueError(msg)

    input_dir = args.input_analysis_dir or _default_input_dir(
        config,
        args.analysis_name,
        args.parameterization,
    )
    output_dir = args.output_dir or _default_output_dir(
        args.analysis_name,
        args.parameterization,
        args.include_groups,
    )
    required_inputs = [
        input_dir / "observables.h5",
        *[input_dir / f"emulation_group_{name}.pkl" for name in args.include_groups],
    ]
    missing = [str(path) for path in required_inputs if not path.exists()]
    if missing:
        msg = f"Missing required input artifacts: {missing}"
        raise FileNotFoundError(msg)
    existing_outputs = [path for path in (output_dir / "mcmc.h5", output_dir / "posterior.h5") if path.exists()]
    if existing_outputs and not args.dry_run:
        msg = f"Refusing to overwrite existing posterior products: {[str(path) for path in existing_outputs]}"
        raise FileExistsError(msg)

    run_summary = {
        "config": str(args.config.resolve()),
        "analysis_name": args.analysis_name,
        "parameterization": args.parameterization,
        "included_groups": args.include_groups,
        "input_analysis_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "mcmc_package": analysis_config["parameters"]["mcmc"].get("mcmc_package", "emcee"),
        "sequential_inference": analysis_config["parameters"]["mcmc"].get(
            "sequential_inference",
            {},
        ),
    }
    logger.info("Resolved subset run:\n%s", json.dumps(run_summary, indent=2))
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "subset_run_manifest.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    mcmc_config = mcmc.MCMCConfig(
        analysis_name=args.analysis_name,
        parameterization=args.parameterization,
        analysis_config=analysis_config,
        config_file=args.config,
        input_analysis_dir=input_dir,
        output_dir=output_dir,
    )
    mcmc.run_mcmc(mcmc_config)


if __name__ == "__main__":
    main()
