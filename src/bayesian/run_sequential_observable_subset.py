"""Compatibility imports and forwarding entry point for a study-owned command.

Numerical helpers remain in bayesian.diagnostics; the CLI recipe belongs to
STAT_soft_hard. See docs/development/ownership.md for installation and ownership.
"""

import warnings

warnings.warn(
    "bayesian.run_sequential_observable_subset is deprecated; use bayesian.diagnostics.run_sequential_observable_subset instead.",
    DeprecationWarning,
    stacklevel=2,
)

from importlib import import_module

from bayesian.diagnostics.run_sequential_observable_subset import (
    select_emulator_groups as select_emulator_groups,
    _default_input_dir as _default_input_dir,
    _default_output_dir as _default_output_dir,
)


def __getattr__(name):
    return getattr(import_module("bayesian.diagnostics.run_sequential_observable_subset"), name)


def main():
    try:
        command = import_module("stat_bayesian_tools.run_sequential_observable_subset")
    except ModuleNotFoundError as error:
        if error.name != "stat_bayesian_tools":
            raise
        message = ("This study CLI moved to STAT_soft_hard/supporting_workflows/bayesian_backend_tools. "
                   "Add that directory to PYTHONPATH; see docs/development/ownership.md.")
        raise SystemExit(message) from error
    command.main()


if __name__ == "__main__":
    main()
