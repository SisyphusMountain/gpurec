# gpurec optimization / MAP / cross-validation layer

This is the kernel-bench research layer merged into gpurec (`gpurec/optim/`), so the
optimization, MAP, and cross-validation work runs on the full hogenom dataset rather than the
frozen 80-family `666x80` capture. See `MERGE_PLAN_kernel_bench_mapcv.md` for the merge design and
the other notes here for the underlying findings (basin geometry, truncation/convergence, HVP
performance, the Sanderson MAP+CV plan).

## Package map (`gpurec/optim/`)

| module | role |
|---|---|
| `value_and_grad.py` | `make_value_and_grad` — the single `(theta_vec)->(loss, grad)` contract, on `stream_batches` |
| `cg.py` | matrix-free CG / Lanczos (`steihaug_cg`, `cg_witness`, `lanczos_extremes`, `lanczos_min_eigpair`) |
| `optimize.py` | `first_order` (Adam + LR schedule), `ridge_anneal`, `newton_polish`, `optimize` |
| `baselines.py` | `lbfgs_scipy`, `gd` |
| `map_cv.py` | **k-fold-over-families CV with λ-homotopy** (the Sanderson MAP+CV goal) |
| `hvp_exact.py` | analytic exact-Hessian HVP (forward-over-reverse through the SO/tangent kernels) |
| `forward_tangent.py` | JVP (tangent forward sweep) feeding the HVP |
| `ggn.py` | Gauss-Newton / Fisher HVP + the parametrized production-backward VJP |
| `newton_cg.py` | Newton-CG / Lanczos outer loops (`newton_lanczos`, `_fd_hessian_hvp`) |
| `map_fit.py` | end-to-end Adam→L-BFGS→MAP-polish→**certified-PD** recipe (`fit`, `spectrum_min`) |
| `diagnostics/` | research scripts (basin/gauge/convergence/theta) — need import-adaptation to run |

Second-order Triton kernels live in `gpurec/core/kernels/`: `e_step_so`, `e_step_tangent`,
`wave_so`, `wave_tangent`, `dts_so`, `dts_tangent`.

## Terminology rename (kernel-bench → gpurec), applied throughout the port

`item↔family`, `col↔receiver`, `state↔species`, `node↔sp`,
`solve_e_pi↔solve_resident_e_pi`, `extract_parameters_weighted_cols↔..._weighted_receivers`,
`as_item_param↔as_family_param`, `as_item_state↔as_family_species`, `rate_item_idx↔rate_family_idx`,
`max_coupling↔max_transfer`, `col_log_probs↔receiver_log_probs`, `statewise↔specieswise`,
`itemwise↔genewise`. The 6 SO/tangent kernels are kept verbatim (kbench-internal names) and bound
to gpurec's renamed forward/backward kernels at the call sites.

## Verification (all reproducible)

| gate | script | result |
|---|---|---|
| value+grad ≡ kbench golden | `python -m gates._parity_kbench` | loss exact, grad rel_L2 **2e-7** (live-kbench 1.4e-5) |
| Adam→L-BFGS reaches the basin | `python -m gates._fit_kbench` | 170130 → **137686** (the known deep basin) |
| masking additivity + prior grad + fp64 FD | `python -m gates._verify_map` | 5e-8 / 3e-8 / **3.8e-4** |
| CV harness (live hogenom) | `python -m gpurec.optim.map_cv` | held-out NLL finite; **regularization helps** (λ=0 → λ\*>0) |
| exact HVP ≡ fp64 FD-of-grad | `python -m gates._verify_hvp` | rel **~1e-6**, symmetry 3e-7 |
| **golden regression (pytest)** | `pytest tests/test_optim_golden.py` | loss+grad at fixture **and checkpoint** vs pinned golden + **bit-exact** cross-parity vs live kbench — see `golden_test.md` |

The capture-based tests need the kernel-bench `666x80` capture (`whole.pt`); the live tests need the
hogenom trees. The fp64 HVP / spectrum at the full `666x80` scale OOMs on 24 GB consumer GPUs — run
those on the OIST saion A100s (`docs/optim/cluster_a100.md`).

## Provenance

kernel-bench (`../../../kernel-bench`) is a frozen faithful extract of gpurec's solver plus this
research layer; it is left intact as the source of truth. This merge ports the layer back; nothing
important is stranded.
