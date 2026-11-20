"""Prepare analytic emulators and observables for a native Bayesian YAML example."""

import argparse
import hashlib
import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from bayesian import data_IO
from examples.inference.models import DATASETS, DISC_MATRIX, LINEAR_MATRIX, LinearEmulator, X, discrepancy_data

ROOT = Path(__file__).resolve().parents[2]


def _observation(y, error, x):
    return {
        "y": np.asarray(y),
        "y_err": np.broadcast_to(error, (len(y),)),
        "xmin": np.asarray(x) - 0.04,
        "xmax": np.asarray(x) + 0.04,
    }


def _inputs(kind, datasets, seed):
    observations, emulators = {"Data": {}, "Prediction": {}}, {}
    if kind.startswith("discrepancy"):
        mode = "fixed" if kind.endswith("fixed") else "inferred"
        y = discrepancy_data(mode, seed)
        chunks = [
            ("hadron__pt_a", np.arange(4), "a"),
            ("hadron__pt_m", np.arange(4, 7), "b"),
            ("hadron__pt_z", np.arange(7, 12), "a"),
        ]
        for label, indices, group in chunks:
            name = f"5020__PbPb__{label}____0-5"
            observations["Data"][name] = _observation(y[indices], 0.07, X[indices])
            emulators.setdefault(group, []).append(LinearEmulator(DISC_MATRIX[indices], 0.03, fixed_normalization=14))
    elif kind == "sequential":
        name = "5020__PbPb__hadron__pt_demo____0-5"
        observations["Data"][name] = _observation([0.32, 1.08], [0.025, 0.15], [1, 2])
        emulators["linear"] = [LinearEmulator([[1, 0, 0], [0, 1, 0.03]], [0.015, 0.03], [0, -0.45])]
    elif kind == "multistep":
        for dataset in datasets:
            name = f"5020__PbPb__hadron__pt_{dataset.lower()}____0-5"
            observations["Data"][name] = _observation(DATASETS[dataset], [0.2, 0.15], [1, 2])
            emulators.setdefault("linear", []).append(
                LinearEmulator(LINEAR_MATRIX, [0.04, 0.03], fixed_normalization=14)
            )
    else:
        msg = f"Unknown example kind {kind}"
        raise ValueError(msg)
    for name, values in observations["Data"].items():
        observations["Prediction"][name] = {"y": np.zeros((len(values["y"]), 4))}
    return observations, emulators


def prepare(config_file, output_dir=None, sampler=None):
    config_file = Path(config_file).resolve()
    config = yaml.safe_load(config_file.read_text())
    output = Path(output_dir or ROOT / config["output_dir"]).resolve()
    if output.exists():
        msg = f"Choose a new output directory: {output}"
        raise FileExistsError(msg)
    # Validate the recipe and paths before creating output files.
    config["output_dir"] = str(output)
    config["observable_table_dir"] = config["observable_config_dir"] = str(output)
    config["analyses"] = {name: deepcopy(analysis) for name, analysis in config["analyses"].items()}
    for analysis in config["analyses"].values():
        settings = analysis["parameters"]["mcmc"]
        if sampler:
            settings["mcmc_package"] = sampler
        sequential = settings.get("sequential_inference", {})
        if sequential.get("enabled"):
            path = Path(sequential["soft_density_model_file"])
            sequential["soft_density_model_file"] = (
                str((config_file.parent / path).resolve()) if not path.is_absolute() else str(path)
            )
            if not Path(sequential["soft_density_model_file"]).is_file():
                raise FileNotFoundError(sequential["soft_density_model_file"])
    output.mkdir(parents=True)
    recipe = config["example"]
    for name in config["analyses"]:
        observations, emulators = _inputs(
            recipe["kind"], recipe.get("datasets", {}).get(name, []), recipe.get("data_seed", 418)
        )
        directory = output / f"{name}_exponential"
        data_IO.write_dict_to_h5(observations, directory, "observables.h5", verbose=False)
        for group, models in emulators.items():
            with (directory / f"emulation_group_{group}.pkl").open("wb") as stream:
                pickle.dump({"emulators": models}, stream)
    resolved = output / "resolved.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "template": str(config_file.relative_to(ROOT)) if config_file.is_relative_to(ROOT) else config_file.name,
        "template_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "data_seed": recipe.get("data_seed", 418),
        "inputs": {
            str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output.rglob("*")
            if path.is_file()
        },
    }
    (output / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sampler", choices=["emcee", "pocoMC"])
    args = parser.parse_args()
    print(prepare(args.config, args.output_dir, args.sampler))  # noqa: T201 - CLI output


if __name__ == "__main__":
    main()
