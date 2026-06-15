# optim diagnostics (research scripts)

Landscape / curvature research scripts carried over from kernel-bench `newton/`. They are kept for
reference and reproducing the basin/gauge/convergence/theta investigations documented in
`gpurec/docs/optim/`.

**Status: not yet adapted to gpurec.** They still import the kernel-bench `newton.*` / `kbench.*`
modules and use the kbench fixture loader. Adapt on demand (apply the rename contract in
`docs/optim/README.md`, re-point onto `gpurec.optim.*` and a `GeneReconModel`/capture) before running
— the same mechanical port already done for the core layer.

- `basin_search.py`, `basin_compare.py`, `basin_connectivity.py`, `basin_interp.py` — basin geometry
- `gauge_audit.py` — gauge/identifiability audit
- `convergence_audit.py` — solver-truncation convergence audit (pi_iters/neumann)
- `theta_diagnostics.py` — per-species rate diagnostics
