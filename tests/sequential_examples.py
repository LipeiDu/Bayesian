"""Versioned soft-density artifacts with an exact, inexpensive hard likelihood."""

import pickle
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import yaml

from bayesian import mcmc
from bayesian.sequential.config import SequentialInferenceConfig
from tests.sampler_examples import build_gaussian_example

ROOT = Path(__file__).resolve().parents[1]
BOUNDS = np.array([[0.5, 1.5], [10.0, 20.0]])
MEASUREMENT = np.array([0.32, 1.08])
UNCERTAINTY = np.sqrt(np.array([0.025, 0.15]) ** 2 + np.array([0.015, 0.03]) ** 2)


def artifact_config(model_type, power=1.0):
    files = {
        "gaussian_kde": "projected_soft_kde_2d/kde_soft_posterior_2d.pkl",
        "torch_realnvp_2d": "projected_soft_flow_2d/flow_soft_posterior_2d.pt",
    }
    return SequentialInferenceConfig(
        soft_density_model_file=ROOT / "data" / "sequential_priors" / files[model_type],
        soft_density_model_type=model_type,
        soft_parameter_names=("nucleon_width", "normalization"),
        alpha_parameter_name="AlphaS",
        hard_likelihood_power=power,
    )


@lru_cache(maxsize=2)
def artifact_payload(model_type):
    path = artifact_config(model_type).soft_density_model_file
    if model_type == "gaussian_kde":
        with path.open("rb") as stream:
            return pickle.load(stream)
    import torch  # noqa: PLC0415

    return torch.load(path, map_location="cpu", weights_only=False)


def reference_soft_density(samples, model_type):
    """Evaluate payloads directly, including an independent flow inversion."""
    payload = artifact_payload(model_type)
    samples = np.atleast_2d(samples)
    unit = (samples - BOUNDS[:, 0]) / np.diff(BOUNDS, axis=1).ravel()
    z = np.log(unit) - np.log1p(-unit)
    jacobian = -np.sum(np.log(np.diff(BOUNDS, axis=1).ravel()) + np.log(unit) + np.log1p(-unit), axis=1)
    if model_type == "gaussian_kde":
        return payload["kde"].logpdf(z.T) + jacobian

    import torch  # noqa: PLC0415
    import torch.nn.functional as functional  # noqa: PLC0415

    if payload.get("standardize", False):
        z = (z - payload["z_mean"]) / payload["z_std"]
        jacobian -= np.log(payload["z_std"]).sum()
    x = torch.tensor(z, dtype=torch.float32)
    log_jacobian = torch.zeros(len(x))
    architecture = payload["architecture"]
    state = payload["state_dict"]
    with torch.no_grad():
        for index in reversed(range(architecture["n_coupling_layers"])):
            prefix = f"layers.{index}."
            mask = state[prefix + "mask"]
            hidden = x * mask
            for layer in range(architecture["hidden_layers"]):
                key = prefix + f"nn.net.{2 * layer}."
                hidden = functional.relu(functional.linear(hidden, state[key + "weight"], state[key + "bias"]))
            key = prefix + f"nn.net.{2 * architecture['hidden_layers']}."
            shift, raw_scale = functional.linear(hidden, state[key + "weight"], state[key + "bias"]).chunk(2, dim=-1)
            scale = torch.tanh(raw_scale) * architecture["scale_limit"] * (1 - mask)
            x = x * mask + (1 - mask) * (x - shift * (1 - mask)) * torch.exp(-scale)
            log_jacobian -= scale.sum(dim=-1)
        base_log_density = -0.5 * (x**2 + np.log(2 * np.pi)).sum(dim=-1)
    return (base_log_density + log_jacobian).numpy() + jacobian


class SequentialLinearEmulator:
    def predict(self, parameters, return_std):
        assert return_std
        mean = np.column_stack([parameters[:, 0], parameters[:, 1] + 0.03 * (parameters[:, 2] - 15.0)])
        return mean, np.tile([0.015, 0.03], (len(parameters), 1))


def hard_reference(samples):
    mean = np.column_stack([samples[:, 0], samples[:, 1] + 0.03 * (samples[:, 2] - 15.0)])
    return -0.5 * np.sum(((mean - MEASUREMENT) / UNCERTAINTY) ** 2, axis=1) - np.log(UNCERTAINTY).sum()


def build_sequential_example(root, sampler, model_type, power):
    initial = build_gaussian_example(root, sampler)
    raw = yaml.safe_load(initial.config_file.read_text())
    analysis = raw["analyses"]["gaussian"]
    analysis["parameterization"]["exponential"] = {
        "names": ["AlphaS", "nucleon_width", "normalization"],
        "min": [0.1, 0.5, 10.0],
        "max": [0.5, 1.5, 20.0],
    }
    settings = analysis["parameters"]["mcmc"]
    settings.update(n_walkers=12, n_burn_steps=80, n_sampling_steps=200, export_hard_likelihood_samples=False)
    settings["pocoMC"].update(n_effective=64, n_active=32, draw_n_prior_samples=128, n_total_samples=500)
    settings["sequential_inference"] = {
        "enabled": True,
        "soft_density_model_file": str(artifact_config(model_type).soft_density_model_file),
        "soft_parameter_names": ["nucleon_width", "normalization"],
        "hard_likelihood_power": power,
    }
    initial.config_file.write_text(yaml.safe_dump(raw))
    with h5py.File(initial.input_analysis_dir / "observables.h5", "r+") as stream:
        observable = next(iter(stream["Data"].values()))
        observable["y"][:] = MEASUREMENT
        observable["y_err"][:] = [0.025, 0.15]
    with (initial.input_analysis_dir / "emulation_group_linear.pkl").open("wb") as stream:
        pickle.dump({"emulators": [SequentialLinearEmulator()]}, stream)
    return mcmc.MCMCConfig(
        analysis_name="gaussian",
        parameterization="exponential",
        analysis_config=analysis,
        config_file=initial.config_file,
    )
