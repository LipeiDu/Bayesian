# Validation and output semantics

The published examples are executable checks of the native Bayesian workflow.
The full regression suite also tests independent mathematical references and
failure cases. It is possible to run these checks using only this checkout.

## Run the suite

With the [example environment](README.md) active:

```bash
export PYTHONPATH="$PWD:$PWD/src"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$(mktemp -d /tmp/bayesian-mpl.XXXXXX)"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 SLURM_CPUS_PER_TASK=1
python -m pytest -p no:cacheprovider --tb=short -q
```

Warnings are errors in the pytest configuration. The checks do not require
study posteriors, private command packages, a JETSCAPE installation, or a GPU.
They use the tracked table fixture and projected-soft artifacts where indicated.
The public recipes exercise both emcee and pocoMC.

The 2026-11-19 source-only validation passed **237 tests in 529.49 seconds**,
including all six native YAML recipes. The reference example metrics and figures
are retained in [example-validation.json](example-validation.json) and the
component guides. Source, configurations, examples and tests were compared
byte-for-byte with the clean copy used for that run.

| Area | Reference or failure check | Tests |
| --- | --- | --- |
| Gaussian likelihood | SciPy Gaussian densities, invalid Cholesky inputs, covariance placement and PCA batch invariance | `test_likelihood_covariance.py` |
| Sampler adapters | Known Gaussian posterior, fixed values, density decomposition and exports | `test_sampler_adapters.py` |
| Discrepancy | Independent elementwise kernels, global bins, parameter names, dense likelihood and posterior quadrature | `test_model_discrepancy.py`, `test_discrepancy_samplers.py`, `test_discrepancy_plots.py` |
| Sequential priors | Independent KDE/flow payload evaluations, support/Jacobians, repeated loading and tempered coupling marginals | `test_soft_density.py`, `test_sequential_samplers.py` |
| Two-step priors | Weighted named projections and exact Gaussian/KDE-mixture updates | `test_posterior_priors.py`, `test_multistep_workflow.py` |
| Posterior consumers | Particle mass, ties, zero weights, bounded draws, model projection and saved PDFs | `test_posterior_summaries.py`, `test_weighted_plot_consumers.py` |
| Fixed-medium integration | Exact uniform/triangular CDFs and moments on nonuniform grids | `test_fixed_medium_alpha_scan.py` |
| Evidence | Normalized analytic Gaussian evidence, box/Jacobian constants, unsupported estimators and boundaries | `test_evidence.py` |
| Published native recipes | YAML preparation → actual driver → independent verifier | `test_public_examples.py` |

A passing stochastic test means agreement within its stated sampling tolerance.
It is not a convergence certificate for arbitrary scientific inputs. The example
verifier checks density fields within `2e-9` for double-precision cases and
`1e-4` for the flow's float32 arithmetic. Posterior means, covariances and CDFs
have separate sampling tolerances in the verifier. The discrepancy quadrature
is a finite-grid approximation; its finer-resolution backend regression checks
convergence independently.

## HDF5 fields

`posterior.h5` contains named sampled parameters and optional particle weights.
Fixed model parameters are represented by configuration metadata and restored
before model prediction. `mcmc.h5` contains the sampler-shaped chain and the same
density components:

- `log_likelihood_hard`: raw Gaussian hard likelihood, including its covariance
  determinant and excluding the common `-n/2*log(2*pi)` constant;
- `log_likelihood_hard_tempered`: its contribution at the configured power;
- `log_prior_bayesian`: the active Bayesian prior;
- `log_target`: the sum of the powered hard likelihood and the active prior.

For pocoMC, legacy `logl` holds the practical target passed as its likelihood
callback. Its `logp` is the normalized uniform reference prior used by the
sampler. The explicit fields above avoid mixing these meanings.

Particle summaries use normalized weights. Quantiles interpolate between
cumulative-mass midpoints after pooling ties and removing zero mass; this is an
explicit convention rather than the inverse step empirical CDF. The legacy
`hpd` option is a shortest contiguous mass interval, not a disconnected
highest-density set. Marginal median parameters are labeled as representatives;
a saved MAP display requires the joint `log_target` values.

## Evidence

When `compute_evidence: true`, emcee defaults to a local-Hessian Laplace
approximation; pocoMC defaults to its native estimate. The `harmonic_mean` method
is retired because the old path used log posterior as log likelihood and the
estimator is unstable. Unsupported or failed estimates receive an explicit
unavailable status while valid posterior outputs are retained.

Let `f=L_* p_*` be the practical target and `M=integral_box p_* dtheta`. For the
untempered model, normalized evidence is

\[
\log Z=\log\!\int_{\mathcal B} f\,d\theta
-\frac{n}{2}\log(2\pi)-\log M.
\]

pocoMC's uniform-reference volume is restored before applying these constants.
Manual-prior normalization includes uniform widths, log ranges and Gaussian
truncation. Laplace optimizes the actual target and uses its local Hessian,
including the parameter-coordinate Jacobian. It rejects boundary-dominated or
non-positive-curvature approximations. A Gaussian regression is exact to its
numerical tolerance; non-Gaussian or multimodal study posteriors can make
Laplace inaccurate, and no standard error is invented for it.

An `evidence` group records the estimator and integration measure. Top-level
`logZ` is emitted only when normalization is established. For posterior-source
and projected-soft priors, the current interface can retain a physical-parameter
`log_target_integral` with `status: target_integral_only`, while withholding
absolute `logZ`. A Bayes-factor calculation needs the relevant normalizers or a
proved cancellation, along with consistent likelihood constants and measures.

```bash
python -m pytest tests/test_evidence.py -q
```

The real native-evidence regression uses 5,000 importance samples and checks
against an analytic normalized Gaussian model. A separate test covers the
Laplace calculation with an optimized mode and independent density context.

## Applying the components to a study

Replace the analytic example inputs with your validated observation tables and
trained emulator artifacts. Establish design coverage, predictive accuracy,
posterior convergence, prior normalization where needed, and the intended data
independence assumptions. The reduced soft-hard study's recipes and scientific
sensitivity studies have separate ownership; see
[component ownership](../development/ownership.md). Public examples and generic
regression tests remain maintained here.
