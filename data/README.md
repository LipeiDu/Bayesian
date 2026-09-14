# Versioned Analysis Inputs

This directory contains compact, versioned inputs required to reproduce the reduced soft-hard Bayesian analyses. Large generated products such as trained emulators, MCMC chains, and posterior files remain under `output/` and are intentionally ignored by Git.

- `emulator_input_tables/20260401/` contains the reduced-design experimental data, design points, and model predictions used to initialize the hard-observable emulators.
- `sequential_priors/` contains immutable copies of the validated projected-soft KDE and normalizing-flow artifacts consumed by the Bayesian sequential runs.

Treat these files as input snapshots. If an input changes, add a new dated or versioned directory rather than silently replacing an existing snapshot.
