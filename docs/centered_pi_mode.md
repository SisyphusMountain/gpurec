# Centered Pi mode

Centered Pi/Pibar storage is selected explicitly through
`SolverOptions.pi_representation`:

```python
from gpurec import GeneReconModel, SolverOptions

model = GeneReconModel(
    species_tree,
    gene_trees,
    device="cuda",
    dtype=torch.float32,
    solver_options=SolverOptions(pi_representation="centered"),
)
```

The corresponding TOML and CLI forms are:

```toml
[solver]
pi_representation = "centered"
```

```text
gpurec reconcile ... --device cuda --dtype float32 --pi-representation centered
```

`"absolute"` remains the default. The former private
`GPUREC_CENTERED_PI_FORWARD` environment switch is no longer consulted, so the
representation is part of the model's validated and serializable solver
configuration.

## Current support boundary

Centered mode stores each Pi/Pibar row as a CUDA fp32 residual and a separate
fp64 row offset. The likelihood head promotes root rows, E, and origination
tensors to fp64 for both representations, so this small final reduction does
not discard the centered offset precision.

| operation | current status |
|---|---|
| single-batch loss | supported forward path |
| streamed and genewise multi-batch loss | supported; fp64 loss dtype is preserved across accumulation |
| single-batch `model().backward()` | supported by the native offset-aware first-order adjoint |
| streamed/genewise gradients and optimization closures | supported by the native offset-aware first-order adjoint |
| Pi/backward convergence diagnostics | supported in centered frames |
| reconciliation backtracking | supported by reconstructing only the selected family in fp64 at the native sampler boundary |
| JVP/tangent kernels | supported with native Pi/DTS frame corrections |
| second-order wave/DTS contractions | supported with native frame corrections |
| exact HVP, GGN, and curvature setup | supported through the saved centered sidecar |

The named fp64 backward bridge reconstructs the two complete Pi matrices and
runs the retained absolute adjoint coherently in fp64. It remains a test oracle
for the native offset-aware kernels, not the public execution path. Native
first- and second-order parity has passed focused uniform/weighted,
single/grouped-split, fp64-reference, and finite-difference fixtures. See
[`forward_precision_report.md`](forward_precision_report.md) for the
component-level fp32/fp64 trace and
[`centered_kernels_report.md`](centered_kernels_report.md) for the numerical,
memory, profile, and runtime evidence.

CPU centered evaluation and fp64 centered residual storage fail explicitly.
The storage contract and per-consumer frame rules are specified in
[`centered_state_contract.md`](centered_state_contract.md).
