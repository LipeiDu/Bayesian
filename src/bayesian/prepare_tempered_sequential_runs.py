"""Prepare emulator-reusing configs for a tempered sequential path."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_BETAS = (0.10, 0.25, 0.50, 0.75)


def beta_tag(beta: float) -> str:
    return f"{beta:.2f}".replace(".", "p")


def build_tempered_config(base_config: dict[str, Any], *, beta: float) -> dict[str, Any]:
    if not 0.0 < beta <= 1.0:
        raise ValueError(f"Tempered run beta must lie in (0, 1], not {beta}.")
    config = copy.deepcopy(base_config)
    config["output_dir"] = f"output/reduced_soft_hard_sequential_tempered_beta_{beta_tag(beta)}"
    config["initialize_observables"] = False
    config["preprocess_input_data"] = False
    config["fit_emulators"] = False
    config["run_mcmc"] = True
    config["run_closure_tests"] = False
    config["plot"] = {key: False for key in config.get("plot", {})}

    for analysis in config["analyses"].values():
        sequential = analysis["parameters"]["mcmc"]["sequential_inference"]
        if not sequential.get("enabled", False):
            raise ValueError("Base config must enable sequential inference.")
        sequential["hard_likelihood_power"] = float(beta)
    return config


def stage_emulator_inputs(*, source_dir: Path, destination_dir: Path) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    source_files = [source_dir / "observables.h5"]
    source_files.extend(sorted(source_dir.glob("emulation_group_*.pkl")))
    missing = [path for path in source_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required emulator inputs: {missing}")
    if len(source_files) == 1:
        raise FileNotFoundError(f"No emulation_group_*.pkl files found in {source_dir}")

    staged: list[Path] = []
    for source in source_files:
        destination = destination_dir / source.name
        if destination.is_symlink() and destination.resolve() == source.resolve():
            staged.append(destination)
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to replace existing staged input {destination}. Remove it explicitly if obsolete."
            )
        destination.symlink_to(source.resolve())
        staged.append(destination)
    return staged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate configs and stage emulator symlinks for tempered sequential pocoMC runs."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("config/reduced_soft_hard_sequential_pocomc.yaml"),
    )
    parser.add_argument("--betas", type=float, nargs="+", default=list(DEFAULT_BETAS))
    parser.add_argument(
        "--source-analysis-dir",
        type=Path,
        default=Path(
            "output/reduced_soft_hard_sequential_pocomc_preprocessed_noPCA/"
            "reduced_soft_hard_exponential"
        ),
    )
    parser.add_argument("--output-config-dir", type=Path, default=Path("config/tempered_sequential"))
    args = parser.parse_args()

    base_config = yaml.safe_load(args.base_config.read_text())
    analysis_name = next(iter(base_config["analyses"]))
    parameterization = base_config["analyses"][analysis_name]["parameterizations"][0]
    args.output_config_dir.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    for beta in args.betas:
        config = build_tempered_config(base_config, beta=beta)
        tag = beta_tag(beta)
        config_path = args.output_config_dir / f"reduced_soft_hard_sequential_beta_{tag}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))

        destination_analysis_dir = (
            Path(config["output_dir"]) / f"{analysis_name}_{parameterization}"
        )
        staged = stage_emulator_inputs(
            source_dir=args.source_analysis_dir,
            destination_dir=destination_analysis_dir,
        )
        print(f"Prepared beta={beta:g}: {config_path}")
        print(f"  staged {len(staged)} emulator input symlinks in {destination_analysis_dir}")
        commands.append(f"PYTHONPATH=src python -m bayesian.steer_analysis -c {config_path}")

    print("\nRun commands:")
    for command in commands:
        print(command)


if __name__ == "__main__":
    main()
