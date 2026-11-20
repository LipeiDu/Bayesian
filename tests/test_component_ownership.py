"""Preserve numerical imports while forwarding study commands to their owning package."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import warnings
import subprocess
import sys

MODULES = [
    "hard_likelihood_diagnostics",
    "fixed_medium_alpha_scan",
    "sector_compatibility_assessment",
    "compare_sequential_observable_subsets",
    "run_sequential_observable_subset",
]


@pytest.mark.parametrize("name", MODULES)
def test_study_cli_forwards_without_copying_its_defaults_into_backend(monkeypatch, name):
    legacy = _legacy(name)
    calls = []
    monkeypatch.setattr(legacy, "import_module", lambda _name: SimpleNamespace(main=lambda: calls.append(_name)))
    legacy.main()
    assert calls == ["stat_bayesian_tools." + name]


@pytest.mark.parametrize("name", MODULES)
def test_reusable_library_does_not_embed_private_machine_paths(name):
    module = import_module("bayesian.diagnostics." + name)
    assert "/Users/" not in Path(module.__file__).read_text()
    legacy = _legacy(name)
    exports = {
        "hard_likelihood_diagnostics": "weighted_summary",
        "fixed_medium_alpha_scan": "summarize_density",
        "sector_compatibility_assessment": "predictive_moments",
        "compare_sequential_observable_subsets": "DensitySamples",
        "run_sequential_observable_subset": "select_emulator_groups",
    }
    assert getattr(legacy, exports[name]) is getattr(module, exports[name])


def _legacy(name):
    # These tests intentionally exercise the compatibility surface. All other
    # tests keep the suite's warning-as-error policy.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"bayesian\..* is deprecated; use bayesian\..* instead\.", category=DeprecationWarning)
        return import_module("bayesian." + name)


@pytest.mark.parametrize("name", MODULES + [
    "sequential_inference", "sequential_prior", "sequential_flow_model", "sequential_export",
])
def test_legacy_import_warns_in_a_fresh_process(name):
    script = """
import importlib, warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', DeprecationWarning)
    importlib.import_module('bayesian.' + NAME)
messages = [str(item.message) for item in caught if item.category is DeprecationWarning]
assert len(messages) == 1, messages
assert messages[0].startswith('bayesian.' + NAME + ' is deprecated; use bayesian.')
"""
    subprocess.run([sys.executable, "-c", "NAME = " + repr(name) + "\n" + script], check=True)
