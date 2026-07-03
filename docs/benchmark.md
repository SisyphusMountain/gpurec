# AleRax fidelity & scale benchmark

## Why AleRax's shipped numbers are not the reference
Stock AleRax v1.3.0 omits the `−ln(2S−1)` origination branch-count term when forming L from
Π and E, so each shipped `per_fam_likelihoods.txt` value is off by `+ln(2S−1)` from gpurec's
likelihood (`test_trees_1`, S=8: shipped −2.56495 vs correct −5.27300; ln 15 = 2.70805). This
ln(2S−1) gap is a normalization difference, not convergence (flat across iterations at the
fixtures' low fitted rates). Convergence is nonetheless **rate-dependent** — at high rates AleRax's
default 4 iterations under-converge (measured 87 nats at fixed D/L/T=1.5/1.6/1.7 on test_mixed_200),
so the regen script pins `--fixed-point-iterations 64`. `benchmark/regen_alerax_refs.sh` rebuilds
`AleRax_fixed` (which computes the AleRax supplementary-formula likelihood gpurec shares) and
generates references under `tests/data/alerax/*/ref/`.

## In-repo fidelity gates (CI, GPU)
- `tests/test_fidelity_alerax.py` — the gate: `gpurec reconcile` at each fixture's AleRax_fixed
  rates matches the AleRax_fixed per-family logL ≤ 1e-3 nats (observed ~1e-5). The first
  checked-in proof that the base computes AleRax's likelihood — "prerequisite #1" — at toy scale.
- `tests/test_alerax_convention.py` (CPU) — documents that the shipped stock numbers differ from
  the AleRax_fixed references by exactly `ln(2S−1)` per family.

Parsers live in `gpurec/bench/alerax_io.py`; the harness in `gpurec/bench/fidelity.py`.

## Scale kit (cluster, not CI)
`benchmark/` reproduces the paper's large-scale fidelity/speed numbers on Saion. See
`benchmark/README.md`. Williams 2017 is a Zenodo download; archaea60 is local (sibling repo)
but ships only an ALEml reference, so it is a runtime/scale target unless you supply an
AleRax output directory.

## Fixtures
Vendored under `tests/data/alerax/<fixture>/` (text only — `sp.nwk`, `g.nwk`, `families.txt`,
shipped stock `per_fam_likelihoods.txt`/`model_parameters.txt`, and `ref/` with the AleRax_fixed
references). The 60-taxon archaea species tree is at
`tests/data/archaea60/reference_species_tree.newick`.
