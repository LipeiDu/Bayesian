# Reproducible inference components

This guide covers model discrepancy, projected-soft sequential inference, and
two-dataset posterior-to-prior inference. The runnable examples use native
Bayesian **YAML** configuration files and inexpensive analytic emulators. They
exercise the real driver, samplers, likelihood, prior loaders, saved outputs,
and component checks without running JETSCAPE simulations.

| Component | Guide | Native configurations |
| --- | --- | --- |
| Fixed and inferred covariance discrepancy | [Model discrepancy](model-discrepancy.md) | [fixed](../../config/examples/discrepancy_fixed.yaml), [inferred](../../config/examples/discrepancy_inferred.yaml) |
| Projected-soft posterior used as a prior | [Sequential inference](sequential-inference.md) | [KDE](../../config/examples/sequential_kde.yaml), [flow](../../config/examples/sequential_flow.yaml) |
| Two datasets fitted in two steps | [Multistep inference](multistep-inference.md) | [Gaussian source](../../config/examples/multistep_gaussian.yaml), [KDE source](../../config/examples/multistep_kde.yaml) |
| Numerical guarantees, outputs, evidence, and test commands | [Validation and outputs](validation.md) | [Automated native-recipe tests](../../tests/test_public_examples.py) |

## Environment

Use a source checkout and Python 3.10. The tested requirements are in
[examples/inference/requirements.txt](../../examples/inference/requirements.txt).
These demonstrations use the emulator prediction protocol and do not need
GPflow, PyMC, simulation outputs, or a private research checkout. The sequential
examples use the versioned projected-soft artifacts already under `data/`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/inference/requirements.txt
export PYTHONPATH="$PWD:$PWD/src"
export MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 SLURM_CPUS_PER_TASK=1
```

The tested environment is x86-64 macOS with Python 3.10.20. The pinned dependency
versions provide a reproduction target; bitwise-identical sampler output across
platforms and numerical libraries is not required by the checks.

## Run one example

Run from the checkout root. Choose a fresh output directory; preparation refuses
to overwrite an existing directory.

```bash
case=discrepancy_fixed
run="output/examples/$case"
python -m examples.inference.prepare \
  --config "config/examples/$case.yaml" --output-dir "$run"
python -m bayesian.steer_analysis -c "$run/resolved.yaml"
python -m examples.inference.verify --output-dir "$run"
```

Preparation writes locally generated analytic emulator objects, observations,
a resolved native YAML file, and input hashes. It resolves artifact paths before
writing the configuration. The optional `example` block is preparation metadata;
inference settings use the normal backend schema. The prepared files are input
to the regular driver; no alternative sampling implementation is involved.

The verifier checks the saved density fields and a known posterior reference,
then writes `validation.json` and `validation.png`. The diagrams below are
rendered from those example outputs. Their parameters are statistical test
fixtures, not calibrated JETSCAPE physics results.

## Run all six recipes

```bash
for case in discrepancy_fixed discrepancy_inferred sequential_kde sequential_flow \
            multistep_gaussian multistep_kde; do
  run="output/examples/$case"
  python -m examples.inference.prepare --config "config/examples/$case.yaml" --output-dir "$run"
  python -m bayesian.steer_analysis -c "$run/resolved.yaml"
  python -m examples.inference.verify --output-dir "$run"
done
```

Defaults exercise both emcee and pocoMC. `prepare --sampler emcee` or
`--sampler pocoMC` overrides a recipe's backend and records the resolved setting.
Examples are small correctness checks; production convergence and emulator
validation require their own study-specific analysis.

## Public tests

```bash
python -m pytest tests/test_public_examples.py -q
```

This prepares every published YAML file, calls `SteerAnalysis.run_analysis`, and
checks the outputs with the same public verifier. See the
[validation guide](validation.md) for independent kernel, density, sampler,
weighted-summary, and evidence regression commands.

Reusable implementation and its regression tests stay in this repository.
Study-specific command recipes and internal engineering records have separate
owners, described in [component ownership](../development/ownership.md).
