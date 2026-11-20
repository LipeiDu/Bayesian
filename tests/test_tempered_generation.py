"""Exercise generated config paths through the normal preparation command."""

import copy
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from bayesian import prepare_tempered_sequential_runs as preparation
from bayesian.sequential_inference import resolve_sequential_inference_config
from tests.sequential_examples import artifact_config


def test_tempered_command_rebases_artifacts_and_stages_zero_endpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    shutil.copyfile(artifact_config("gaussian_kde").soft_density_model_file, models / "soft.pkl")
    source = tmp_path / "configs" / "base.yaml"
    source.parent.mkdir()
    analysis = {
        "parameterizations": ["exponential"],
        "parameterization": {
            "exponential": {
                "names": ["AlphaS", "nucleon_width", "normalization"],
                "min": [0.1, 0.5, 10.0],
                "max": [0.5, 1.5, 20.0],
            }
        },
        "parameters": {
            "mcmc": {
                "sequential_inference": {
                    "enabled": True,
                    "soft_density_model_file": "../models/soft.pkl",
                    "soft_parameter_names": ["nucleon_width", "normalization"],
                }
            }
        },
    }
    original = {"plot": {}, "analyses": {"a": analysis}}
    source.write_text(yaml.safe_dump(original))
    inputs = tmp_path / "emulators"
    inputs.mkdir()
    for name in ["observables.h5", "emulation_group_linear.pkl"]:
        (inputs / name).touch()
    output = tmp_path / "generated" / "nested"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--base-config",
            str(source),
            "--betas",
            "0",
            "0.25",
            "--source-analysis-dir",
            str(inputs),
            "--output-config-dir",
            str(output),
        ],
    )
    preparation.main()
    for power in [0.0, 0.25]:
        config_file = output / f"reduced_soft_hard_sequential_beta_{preparation.beta_tag(power)}.yaml"
        generated = yaml.safe_load(config_file.read_text())
        target = generated["analyses"]["a"]
        specification = resolve_sequential_inference_config(
            mcmc_configuration=target["parameters"]["mcmc"],
            config_file=config_file,
            analysis_config=target,
            parameterization="exponential",
        )
        assert specification.soft_density_model_file == models / "soft.pkl"
        assert specification.hard_likelihood_power == power
        staged = Path(generated["output_dir"]) / "a_exponential"
        assert (staged / "observables.h5").resolve() == inputs / "observables.h5"
        assert (staged / "emulation_group_linear.pkl").resolve() == inputs / "emulation_group_linear.pkl"
    assert yaml.safe_load(source.read_text()) == original


def test_rebasing_shared_yaml_settings_uses_original_path_once(tmp_path) -> None:
    sequential = {"enabled": True, "soft_density_model_file": "../models/soft.pkl"}
    analysis = {"parameters": {"mcmc": {"sequential_inference": sequential}}}
    base = {"analyses": {"a": analysis, "b": analysis}}
    original = copy.deepcopy(base)
    result = preparation.build_tempered_config(
        base,
        beta=0.25,
        source_config_file=tmp_path / "configs" / "base.yaml",
        output_config_file=tmp_path / "nested" / "configs" / "tempered.yaml",
    )
    for target in result["analyses"].values():
        assert (
            target["parameters"]["mcmc"]["sequential_inference"]["soft_density_model_file"] == "../../models/soft.pkl"
        )
    assert base == original


def test_rebasing_requires_both_config_locations(tmp_path) -> None:
    with pytest.raises(ValueError, match="both"):
        preparation.build_tempered_config({}, beta=0.25, source_config_file=tmp_path / "base.yaml")
