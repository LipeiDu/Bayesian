"""Preserve numerical imports while forwarding study commands to their owning package."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULES = [
    "hard_likelihood_diagnostics",
    "fixed_medium_alpha_scan",
    "sector_compatibility_assessment",
    "compare_sequential_observable_subsets",
    "run_sequential_observable_subset",
]


@pytest.mark.parametrize("name", MODULES)
def test_study_cli_forwards_without_copying_its_defaults_into_backend(monkeypatch, name):
    legacy = import_module("bayesian." + name)
    calls = []
    monkeypatch.setattr(legacy, "import_module", lambda _name: SimpleNamespace(main=lambda: calls.append(_name)))
    legacy.main()
    assert calls == ["stat_bayesian_tools." + name]


@pytest.mark.parametrize("name", MODULES)
def test_reusable_library_does_not_embed_private_machine_paths(name):
    module = import_module("bayesian.diagnostics." + name)
    assert "/Users/" not in Path(module.__file__).read_text()
    legacy = import_module("bayesian." + name)
    exports = {
        "hard_likelihood_diagnostics": "weighted_summary",
        "fixed_medium_alpha_scan": "summarize_density",
        "sector_compatibility_assessment": "predictive_moments",
        "compare_sequential_observable_subsets": "DensitySamples",
        "run_sequential_observable_subset": "select_emulator_groups",
    }
    assert getattr(legacy, exports[name]) is getattr(module, exports[name])
