# JETSCAPE Bayesian inference of QCD transport properties

This repository contains analysis code to implement Bayesian inference of one of the key emergent quantities of quantum chromodynamics (QCD), the jet transverse diffusion coefficient $\hat{q}$.
The most recent results, performing the first comprehensive analysis including hadron and jet observables, were published in [arXiv:2408.08247](https://inspirehep.net/literature/2818238).
The initial results were published in [arXiv:2102.11337](https://inspirehep.net/literature/1847995).
This codebase is a merge of a variety of Bayesian inference codebase, including the [JETSCAPE STAT](https://github.com/JETSCAPE/STAT) codebase, the [bayesian-inference](https://github.com/raymondEhlers/bayesian-Inference) codebase, etc...

The end-to-end workflow consists of:
- Simulating a physics model $f(\theta)$ at a collection of design points $\theta$ using the [JETSCAPE](https://github.com/JETSCAPE/JETSCAPE) framework – requiring $\mathcal{O}(10M)$ CPU-hours.
- Using PCA to reduce the dimensionality of the feature space.
- Fitting Gaussian Processes to emulate the physics model at any $\theta$.
- Sampling the posterior $P(\theta|D)$ using MCMC, with a Gaussian likelihood constructed by comparing the emulated physics model $f(\theta)$ to published experimental measurements $D$ from the Large Hadron Collider (LHC) and the Relativistic Heavy Ion Collider (RHIC).

This results in a constraint on the transverse diffusion coefficient $\hat{q}$ describing a jet with energy $E$ propagating through deconfined QCD matter with temperature $T$.
![image](https://github.com/jdmulligan/bayesian-inference/assets/16219745/faac0d39-39ad-4acf-a898-91ec51d57a31)

## Running the data pipeline

The data pipeline consists of the following optional steps:
1. Read in the design points, predictions, experimental data, and uncertainties
2. Perform PCA and fit GP emulators
3. Run MCMC
4. Plot results and validation

The analysis is steered by the script `steer_analysis.py`, where you can specify which parts of the pipeline you want to run, along with a config file (e.g. `jet_substructure.yaml`).

The config files will specify which steps to run along with input/output paths for each step, where applicable.

Compact, reproducibility-critical inputs for the reduced soft-hard study are versioned under `data/`. This includes the 20260401 emulator tables and the validated projected-soft density artifacts. Generated emulators, chains, and posteriors remain under the ignored `output/` directory.

### To run the analysis:

```bash
export PYTHONPATH="$PWD:$PWD/src"
python -m bayesian.steer_analysis -c ./config/jet_substructure.yaml
```

Configure the software as usual for python using a virtual environment.

## Reproducible components and tests

Start with the [public inference guide](docs/features/README.md) for runnable
YAML examples of model discrepancy, sequential KDE/flow inference, and two-step
posterior-to-prior inference. The examples prepare their own analytic emulator
inputs and include mathematical validation and figures.

The [validation guide](docs/features/validation.md) documents regression commands,
weighted output semantics, and evidence limitations. [Component ownership](docs/development/ownership.md)
separates the reusable backend from study command recipes and internal review
records. Backend tests remain in `tests/`.
