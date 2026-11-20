"""Small serialized emulator and analytic target shared by sampler tests and figures."""

import pickle
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import multivariate_normal

from bayesian import data_IO, mcmc

MATRIX = np.array([[1.0, 0.4], [-0.3, 1.0]])
DATA = np.array([0.35, 1.0])
ERROR = np.sqrt(np.array([0.2, 0.15]) ** 2 + np.array([0.04, 0.03]) ** 2)
PRIOR_MEAN = np.array([0.1, 0.8])
PRIOR_STD = np.array([0.5, 0.4])


class LinearEmulator:
    """Exact linear prediction with a fixed emulator uncertainty."""

    def predict(self, parameters, return_std):
        assert return_std
        # The fixed third parameter must reach the emulator in full model order.
        np.testing.assert_array_equal(parameters[:, 2], 14.0)
        mean = parameters[:, :2] @ MATRIX.T
        return mean, np.tile([0.04, 0.03], (len(parameters), 1))


def gaussian_reference(samples):
    likelihood = -0.5 * np.sum(((samples[:, :2] @ MATRIX.T - DATA) / ERROR) ** 2, axis=1)
    likelihood -= np.log(ERROR).sum()
    prior = multivariate_normal.logpdf(samples[:, :2], mean=PRIOR_MEAN, cov=np.diag(PRIOR_STD**2))
    precision = MATRIX.T @ np.diag(1 / ERROR**2) @ MATRIX + np.diag(1 / PRIOR_STD**2)
    covariance = np.linalg.inv(precision)
    mean = covariance @ (MATRIX.T @ (DATA / ERROR**2) + PRIOR_MEAN / PRIOR_STD**2)
    return likelihood, prior, mean, covariance


def build_gaussian_example(root: Path, sampler: str, *, seed: int = 314159):
    """Write isolated inputs and return the backend's normal MCMC configuration."""
    root.mkdir(parents=True, exist_ok=True)
    settings = {
        "mcmc_package": sampler,
        "random_seed": seed,
        "n_walkers": 20,
        "n_burn_steps": 120,
        "n_sampling_steps": 400,
        "n_logging_steps": 1000,
        "export_hard_likelihood_samples": True,
        "pocoMC": {
            "n_effective": 128,
            "n_active": 64,
            "draw_n_prior_samples": 256,
            "n_total_samples": 1000,
            "n_importance_samples_for_evidence": 0,
        },
    }
    analysis = {
        "parameterizations": ["exponential"],
        "parameterization": {
            "exponential": {
                "names": ["AlphaS", "nucleon_width", "normalization"],
                "min": [-2.0, -1.0, 10.0],
                "max": [2.0, 3.0, 20.0],
                "fixed_parameters": {"normalization": 14.0},
                "prior": {
                    "type": ["gaussian", "gaussian", "uniform"],
                    "mean": [0.1, 0.8, None],
                    "std": [0.5, 0.4, None],
                },
            }
        },
        "parameters": {
            "mcmc": settings,
            "emulators": {
                "linear": {
                    "emulator_name": "sk_learn",
                    "n_pc": None,
                    "observable_list": ["hadron__pt_"],
                }
            },
        },
    }
    raw = {
        "output_dir": str(root),
        "observable_table_dir": str(root),
        "observable_config_dir": str(root),
        "observables_filename": "observables.h5",
        "analyses": {"gaussian": analysis},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = root / "gaussian_exponential"
    observable = "5020__PbPb__hadron__pt_test____0-5"
    data_IO.write_dict_to_h5(
        {
            "Data": {
                observable: {
                    "y": DATA,
                    "y_err": np.array([0.2, 0.15]),
                    "xmin": np.array([1.0, 2.0]),
                    "xmax": np.array([2.0, 3.0]),
                }
            },
            "Prediction": {observable: {"y": np.zeros((2, 4))}},
        },
        output,
        "observables.h5",
        verbose=False,
    )
    with (output / "emulation_group_linear.pkl").open("wb") as stream:
        pickle.dump({"emulators": [LinearEmulator()]}, stream)
    return mcmc.MCMCConfig(
        analysis_name="gaussian", parameterization="exponential", analysis_config=analysis, config_file=path
    )
