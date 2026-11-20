# Component ownership and maintenance boundaries

Library implementation details stay with the package they implement. Study
recipes, local input locations, and engineering review history have separate
owners. Public-facing support is defined by the native configurations and
[component guides](../features/README.md), with regression tests in this checkout.
An underscore-prefixed implementation helper is not a stable external API.

| Material | Owner and location | Tracking and support |
| --- | --- | --- |
| Likelihood, emulator organization, priors, samplers, discrepancy, sequential loading, output semantics | `Bayesian/src/bayesian/` | Public backend, tested here |
| Native driver, sequential self-check, tempered-config generator | `Bayesian/src/bayesian/` | Public commands |
| Reusable posterior, likelihood and scan diagnostics | `Bayesian/src/bayesian/diagnostics/` | Backend helper library; APIs may evolve with tests |
| Portable statistical demonstrations | `Bayesian/config/examples/`, `examples/inference/`, `docs/features/` | Public configs, prepared analytic inputs, and executable validation |
| Regression and integration tests | `Bayesian/tests/` | Stay beside the implementation, including generic tests motivated by a study |
| Reduced soft-hard production/reproduction recipes | `jetscape-bayesian-workflows` | Study workflow and artifact provenance |
| Study-only diagnostic CLI defaults and research paths | `STAT_soft_hard/supporting_workflows/bayesian_backend_tools/stat_bayesian_tools/` | Private research workspace, separately tracked |
| Scientific interpretation and sensitivity studies | `STAT_soft_hard` | Private project documentation and results |
| Detailed backend debugging history and engineering figures | `heavy-ion-research/infrastructure/bayesian-backend-review/2026-11-19/` | Private engineering record and full illustrated review |

The historical configs outside `config/examples/` retain their original role in
archived study reproduction and regression coverage. They are not portable
quickstarts. The companion owns the production recipe; the frozen table and
soft-density inputs remain versioned here until their distribution is managed
by a separate, versioned artifact release. Original history is preserved.

## Study command migration

The following command recipes moved to the study-owned `stat_bayesian_tools`
package. Their numerical calculations remain in `bayesian.diagnostics` with
unchanged function bodies. Existing numerical imports remain available through
compatibility modules.

| Previous command module | Study recipe | Reusable helper module |
| --- | --- | --- |
| `bayesian.hard_likelihood_diagnostics` | `stat_bayesian_tools.hard_likelihood_diagnostics` | `bayesian.diagnostics.hard_likelihood_diagnostics` |
| `bayesian.fixed_medium_alpha_scan` | `stat_bayesian_tools.fixed_medium_alpha_scan` | `bayesian.diagnostics.fixed_medium_alpha_scan` |
| `bayesian.sector_compatibility_assessment` | `stat_bayesian_tools.sector_compatibility_assessment` | `bayesian.diagnostics.sector_compatibility_assessment` |
| `bayesian.compare_sequential_observable_subsets` | `stat_bayesian_tools.compare_sequential_observable_subsets` | `bayesian.diagnostics.compare_sequential_observable_subsets` |
| `bayesian.run_sequential_observable_subset` | `stat_bayesian_tools.run_sequential_observable_subset` | `bayesian.diagnostics.run_sequential_observable_subset` |

For private study work, set these checkout paths explicitly:

```bash
export BAYESIAN=/path/to/Bayesian
export STAT_PROJECT=/path/to/STAT_soft_hard
export PYTHONPATH="$BAYESIAN:$BAYESIAN/src:$STAT_PROJECT/supporting_workflows/bayesian_backend_tools"
python -m stat_bayesian_tools.hard_likelihood_diagnostics --help
```

The previous `python -m bayesian.<command>` names forward to the study package
when it is available. Without that package, they explain where the recipe moved.
Public native examples and backend tests do not need this package or a private
checkout. No private command source is fetched automatically.

The complete module-role inventory is in
[component-inventory.json](component-inventory.json). The branch preservation
record is in [branch-archive.json](branch-archive.json). The public validation
guide contains the executable test commands; the private review records the
full debugging sequence, study verification, and internal discussion.

## Preserved branches

The inactive `bayesian_evidence`, `multistep_inference`, `model_discrepancy`,
`refactor_emulator`, and `soft-hard-study` tips have annotated preservation tags
under `archive/2026-11-19/`. The first, second, third and fifth tips are included
in the reviewed ancestry. `refactor_emulator` contains one distinct emulator
API/filename refactor, preserved without merging it into the tested code.
The machine-readable archive record gives the exact tips and retirement status.

Restore a branch for inspection with, for example:

```bash
git fetch origin --tags
git switch -c inspect-emulator-refactor archive/2026-11-19/refactor_emulator
```

The reviewed public branch is `codex/inference-components`, based on the
published study baseline `97ea901`. The default `main` branch remains unchanged.
