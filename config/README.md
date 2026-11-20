# Configuration ownership

The maintained public quickstarts are in [`examples/`](examples/), with
[component guides](../docs/features/README.md) and a preparation command that
writes all analytic inputs before the native Bayesian driver runs.

Other YAML files are historical laboratory and reduced-study recipes. Their
input and output paths assume the corresponding data or archived emulator
layout. They remain here for archived reproduction and regression coverage;
production workflow ownership is described in
[component ownership](../docs/development/ownership.md).

The backend reads YAML. JETSCAPE simulation XML belongs to the upstream
simulation workflow and is not an inference configuration format.
