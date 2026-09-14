from __future__ import annotations

import pytest

from bayesian.run_sequential_observable_subset import select_emulator_groups


def _analysis_config() -> dict:
    return {
        "parameters": {
            "emulators": {
                "hadron_group": {"observable_list": ["hadron"]},
                "jet_group": {"observable_list": ["jet"]},
            }
        }
    }


def test_select_emulator_groups_keeps_only_requested_group() -> None:
    source = _analysis_config()

    selected = select_emulator_groups(source, ["jet_group"])

    assert list(selected["parameters"]["emulators"]) == ["jet_group"]
    assert list(source["parameters"]["emulators"]) == ["hadron_group", "jet_group"]


def test_select_emulator_groups_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="Unknown emulator groups"):
        select_emulator_groups(_analysis_config(), ["photon_group"])


def test_select_emulator_groups_requires_a_group() -> None:
    with pytest.raises(ValueError, match="At least one"):
        select_emulator_groups(_analysis_config(), [])
