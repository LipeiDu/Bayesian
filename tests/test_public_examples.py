"""Run the published YAML recipes through the real native driver and verifier."""

from pathlib import Path

import pytest

from bayesian.steer_analysis import SteerAnalysis
from examples.inference.prepare import prepare
from examples.inference.verify import verify

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    "baseline",
    "discrepancy_fixed",
    "discrepancy_inferred",
    "sequential_kde",
    "sequential_flow",
    "multistep_gaussian",
    "multistep_kde",
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.slow
def test_public_native_example(tmp_path, monkeypatch, case):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    config = prepare(ROOT / "config/examples" / f"{case}.yaml", tmp_path / case)
    SteerAnalysis(config_file=str(config)).run_analysis()
    result = verify(config.parent)
    assert result["status"] == "passed"
    assert (config.parent / "validation.png").stat().st_size > 1000
    assert (config.parent / "inputs.json").is_file()


def test_example_preparation_refuses_existing_outputs(tmp_path):
    with pytest.raises(FileExistsError, match="new output"):
        prepare(ROOT / "config/examples/sequential_kde.yaml", tmp_path)
