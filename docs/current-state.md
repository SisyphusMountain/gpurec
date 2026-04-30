# gpurec Current State

> Last updated: 2026-04-05. For architecture details see [architecture.md](architecture.md).

## What Works

- **Wave forward pass** (`Pi_wave_forward`): production solver, wave-ordered layout (v2), per-wave convergence.
- **Wave backward pass** (`Pi_wave_backward`): implicit differentiation via VJP, Neumann/CG/GMRES solvers.
- **Batched likelihood**: cross-family wave batching via `GeneDataset.compute_likelihood_batch()`.
- **Optimization**: `optimize_theta_wave` (SGD/Adam), `optimize_theta_genewise` (L-BFGS).
- **Pibar modes**: `dense` (cuBLAS TF32), `uniform` (O(W*S) formula), `topk` (compressed log-matmul).

## Performance (S=1999, 10 families)

| Path | Time | Speedup |
|------|------|---------|
| Wave v2 | 2.6s (~200ms/fam) | 18.6x vs FP |
| Fixed-point (legacy) | 48.0s (~4.8s/fam) | baseline |

At S=19999 (10K-leaf species tree): 0.26s/family with uniform pibar mode, 18 GB peak.

## Parameter Modes

| transfer_mode | genewise=False | genewise=True |
|--------------|----------------|---------------|
| uniform | tested | tested |
| specieswise | tested | tested |
| pairwise | forward only (no gradient) | not implemented |

## Known Gaps

- **Pairwise backward**: untested, no gradient support.
- **Batched backward at large S**: only validated at S<=199.
- **L-BFGS convergence**: `optimize_theta_genewise` built but never run to convergence on real data.
- **Small-S path**: wave solver slower than legacy fixed-point for small species trees.
- **Large-S memory**: ~18-20 GB peak for single family at S=20K.
