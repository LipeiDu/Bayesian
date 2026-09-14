# Reduced Soft-Hard Emulator Tables: 20260401

This is the versioned input snapshot for the reduced `(AlphaS, nucleon_width, normalization)` hard-observable study. It was copied on 2026-09-14 from the previously unversioned local campaign directory:

`../STAT-outputs/20260401/tables`

The snapshot contains 271 text tables:

- 90 experimental-data tables under `tables/Data/`
- 1 reduced Latin-hypercube design table under `tables/Design/`
- 180 model-prediction and statistical-error tables under `tables/Prediction/`

The active reduced-study YAML configurations use `data/emulator_input_tables/20260401/tables` as `observable_table_dir`. Run the Bayesian commands from the repository root, as documented in the main README.

`SHA256SUMS` records an individual checksum for every table. Verify the snapshot with:

```bash
cd data/emulator_input_tables/20260401
shasum -a 256 -c SHA256SUMS
```

The tables are authoritative emulator inputs, not generated Bayesian outputs. Do not edit them in place; create a new versioned snapshot for a revised simulation campaign.
