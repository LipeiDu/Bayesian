# Projected-Soft Density Artifacts

This directory archives the validated two-dimensional soft-posterior density artifacts used as priors by the Bayesian sequential analyses. They are immutable inference inputs copied from the `STAT_soft_hard` project, where fitting and validation remain authoritative.

- `projected_soft_kde_2d/` contains the production Gaussian KDE artifact and metadata.
- `projected_soft_flow_2d/` contains the normalizing-flow cross-check artifact and metadata.

The Bayesian YAML files refer to these copies with paths relative to their config files, removing the previous dependency on a user-specific absolute path. `SHA256SUMS` records the archived payload checksums. Verify them with:

```bash
cd data/sequential_priors
shasum -a 256 -c SHA256SUMS
```

When the project-side density fit changes, copy it into a new versioned directory, retain its metadata, update the relevant configs, and record the change rather than overwriting these inputs silently.
