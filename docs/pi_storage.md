# Pi storage

Pi/Pibar state on CUDA always uses centered storage. Each row is represented by
a residual in the model dtype (`torch.float32` or `torch.float64`) and a
separate fp64 row offset. This is an invariant of the solver rather than a
runtime option: Python, TOML, CLI, and environment configuration cannot select
an alternative storage path.

The likelihood head reconstructs root rows in fp64 and promotes E and
origination tensors before reduction. Streamed family and batch reductions also
remain fp64, so an fp32 model does not discard the offset precision at the
small final reduction.

## Support boundary

| operation | current status |
|---|---|
| CUDA fp32 and fp64 forward | centered residuals plus fp64 row offsets |
| single-batch loss | supported forward path |
| streamed and genewise multi-batch loss | supported; fp64 loss dtype is preserved across accumulation |
| single-batch `model().backward()` | supported by the native offset-aware first-order adjoint |
| streamed/genewise gradients and optimization closures | supported by the native offset-aware first-order adjoint |
| Pi/backward convergence diagnostics | supported in centered frames |
| reconciliation backtracking | reconstructs only the selected family in fp64 at the sampler boundary |
| JVP/tangent kernels | supported with native Pi/DTS frame corrections |
| second-order wave/DTS contractions | supported with native frame corrections |
| exact HVP, GGN, and curvature setup | supported through the saved centered sidecar |

An fp64 model uses the same canonical kernels and serves as the high-precision
oracle in focused tests; production execution does not fall back to a second
absolute-storage solver. Native first- and second-order parity has passed
focused uniform/weighted, single/grouped-split, fp64-reference, and
finite-difference fixtures. See
[`forward_precision_report.md`](forward_precision_report.md) for the
component-level fp32/fp64 trace and
[`centered_kernels_report.md`](centered_kernels_report.md) for the numerical,
memory, profile, and runtime evidence.

Centered kernels require CUDA. CPU reconciliation fails at the CUDA support
boundary rather than selecting another representation. The storage contract
and per-consumer frame rules are specified in
[`centered_state_contract.md`](centered_state_contract.md).
