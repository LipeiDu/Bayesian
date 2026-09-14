from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bayesian.emulation import base


def test_mapping_uses_only_observables_from_active_groups(monkeypatch) -> None:
    observables = {
        "Prediction": {
            "hadron_a": {"y": np.zeros((2, 4))},
            "jet_a": {"y": np.zeros((3, 4))},
        }
    }

    monkeypatch.setattr(base.data_IO, "read_dict_from_h5", lambda *args, **kwargs: observables)

    def sorted_observables(values, observable_filter=None):
        if observable_filter == "hadron_filter":
            return ["hadron_a"]
        return ["hadron_a", "jet_a"]

    monkeypatch.setattr(base.data_IO, "sorted_observable_list_from_dict", sorted_observables)
    config = SimpleNamespace(
        output_dir="unused",
        emulation_groups_config={
            "hadron_group": SimpleNamespace(observable_filter="hadron_filter"),
        },
    )

    mapping = base.SortEmulationGroupObservables.learn_mapping(config)

    assert list(mapping.emulation_group_to_observable_matrix) == ["hadron_a"]
    assert mapping.emulation_group_to_observable_matrix["hadron_a"] == (
        "hadron_group",
        slice(0, 2),
        slice(0, 2),
    )
    assert mapping.shape == (4, 2)
