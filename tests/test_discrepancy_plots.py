"""Check local kernel slices separately from global uncertainty-vector indices."""

import numpy as np

from bayesian.plot_discrepancy import extract_groupwise_diagonals, get_group_metadata
from tests.test_model_discrepancy import block_example


def test_discrepancy_plot_metadata_distinguishes_local_and_global_slices():
    config, _settings, _coords, _params = block_example()
    metadata = get_group_metadata(config)
    assert metadata["a"]["observable_slices"] == [slice(0, 2), slice(2, 4)]
    assert metadata["a"]["global_slices"] == [slice(0, 2), slice(3, 5)]
    extracted = extract_groupwise_diagonals(np.arange(5), metadata)
    np.testing.assert_array_equal(extracted["a"], [0, 1, 3, 4])
    np.testing.assert_array_equal(extracted["b"], [2])
