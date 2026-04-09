from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ParameterizationInfo:
    full_names: list[str]
    full_min: list[float]
    full_max: list[float]
    fixed_parameters: dict[str, float]

    @property
    def sampled_names(self) -> list[str]:
        return [name for name in self.full_names if name not in self.fixed_parameters]

    @property
    def sampled_min(self) -> list[float]:
        return [
            lower
            for name, lower in zip(self.full_names, self.full_min, strict=True)
            if name not in self.fixed_parameters
        ]

    @property
    def sampled_max(self) -> list[float]:
        return [
            upper
            for name, upper in zip(self.full_names, self.full_max, strict=True)
            if name not in self.fixed_parameters
        ]

    def filter_prior_config(self, prior_config: dict[str, Any] | None) -> dict[str, Any] | None:
        if prior_config is None or "prior_source" in prior_config:
            return prior_config

        sampled_indices = [
            i for i, name in enumerate(self.full_names) if name not in self.fixed_parameters
        ]
        filtered_config = dict(prior_config)
        for key in ["type", "mean", "std"]:
            if key in filtered_config:
                filtered_config[key] = [filtered_config[key][i] for i in sampled_indices]
        return filtered_config

    def expand_sampled_to_full(
        self,
        X_sampled: npt.NDArray[np.float64],
        sampled_names: list[str] | None = None,
    ) -> npt.NDArray[np.float64]:
        X_sampled = np.array(X_sampled, dtype=float, ndmin=2)

        if sampled_names is None:
            sampled_names = self.sampled_names

        if X_sampled.shape[1] == len(self.full_names):
            return X_sampled

        expected_sampled = len(sampled_names)
        if X_sampled.shape[1] != expected_sampled:
            raise ValueError(
                f"Expected {expected_sampled} sampled parameters with names {sampled_names}, "
                f"but received array with shape {X_sampled.shape}."
            )

        sampled_name_to_index = {name: i for i, name in enumerate(sampled_names)}
        X_full = np.zeros((X_sampled.shape[0], len(self.full_names)))
        for i_full, name in enumerate(self.full_names):
            if name in sampled_name_to_index:
                X_full[:, i_full] = X_sampled[:, sampled_name_to_index[name]]
            else:
                X_full[:, i_full] = self.fixed_parameters[name]
        return X_full


def parameterization_info(
    analysis_config: dict[str, Any],
    parameterization: str,
) -> ParameterizationInfo:
    parameterization_config = analysis_config["parameterization"][parameterization]
    fixed_parameters = parameterization_config.get("fixed_parameters", {})
    return ParameterizationInfo(
        full_names=list(parameterization_config["names"]),
        full_min=list(parameterization_config["min"]),
        full_max=list(parameterization_config["max"]),
        fixed_parameters=dict(fixed_parameters),
    )
