# Pi storage

Pi/Pibar state on CUDA always uses centered storage. Each row is represented by
a residual in the model dtype (`torch.float32` or `torch.float64`) and a
separate row offset in the configured accumulator dtype. Centered storage is an
invariant of the solver rather than a runtime option; its two precisions are
configured through `[precision].model_dtype` and
`[precision].accumulator_dtype`.

The likelihood head reconstructs root rows in the accumulator dtype and
promotes E and origination tensors before reduction. Streamed family and batch
reductions retain that dtype, so the small final reduction does not discard row
offset precision.

Supported model/accumulator pairs are `float32/float32`, `float32/float64`, and
`float64/float64`. `float64/float32` is rejected: an accumulator may be wider
than the model state but never narrower. The shipped default is
`float32/float64`; it preserves the mixed-precision behavior measured in the
centered-kernel reports. An explicit Python `dtype=` or CLI `--dtype` overrides
only `model_dtype`, so it must still be compatible with the configured
accumulator.

## Support boundary

| operation | current status |
|---|---|
| CUDA fp32 and fp64 forward | centered residuals plus configured-accumulator row offsets |
| single-batch loss | supported forward path |
| streamed and genewise multi-batch loss | supported; configured accumulator dtype is preserved across accumulation |
| single-batch `model().backward()` | supported by the native offset-aware first-order adjoint |
| streamed/genewise gradients and optimization closures | supported by the native offset-aware first-order adjoint |
| Pi/backward convergence diagnostics | supported in centered frames |
| reconciliation backtracking | reconstructs only the selected family; the native sampler boundary converts to f64 because its host ABI requires f64 arrays |
| JVP/tangent kernels | supported with native Pi/DTS frame corrections |
| second-order wave/DTS contractions | supported with native frame corrections |
| exact HVP and GGN | supported through the saved centered sidecar |

An fp64/fp64 model uses the same canonical kernels and serves as the high-precision
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
