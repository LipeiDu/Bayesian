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
