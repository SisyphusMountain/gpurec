#!/usr/bin/env bash
# Generate AleRax_fixed reference likelihoods for the toy fixtures. AleRax_fixed computes
# gpurec's likelihood convention (includes the -ln(2S-1) origination term the stock v1.3.0
# shipped numbers omit). Copies + builds AleRax_fixed to a stable location OUTSIDE the tracked
# repo, runs each fixture, and vendors ref/{per_fam_likelihoods.txt,model_parameters.txt}.
#
# The build also applies a small full-precision patch to AleOptimizer.cpp (AleRax prints only
# ~6 significant figures by default, which truncates large-magnitude likelihoods/rates below the
# fidelity gate's resolution). With the patch, per_fam_likelihoods.txt and model_parameters.txt
# carry 17 significant digits so gpurec is compared at full precision.
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DATA="$REPO/tests/data/alerax"
ALERAX_SRC="${ALERAX_SRC:-}"
ALERAX_DST="${ALERAX_DST:-$REPO/.local-tools/AleRax_fixed}"
BIN="$ALERAX_DST/build/bin/alerax"

# 1) copy + build (idempotent; build lives OUTSIDE the tracked repo)
if [ ! -x "$BIN" ]; then
  [ -n "$ALERAX_SRC" ] || { echo "AleRax_fixed source not configured; set ALERAX_SRC"; exit 1; }
  mkdir -p "$ALERAX_DST"
  rsync -a --exclude build --exclude .git "$ALERAX_SRC/" "$ALERAX_DST/"
  # full-precision patch: emit 17 significant digits for per-family likelihoods + global rates
  # (default is ~6 sig figs, which truncates large values). Idempotent (grep-guarded).
  AO="$ALERAX_DST/src/ale/AleOptimizer.cpp"
  if [ -f "$AO" ] && ! grep -q "setprecision(17)" "$AO"; then
    sed -i \
      -e 's|#include <cassert>|#include <cassert>\n#include <iomanip>|' \
      -e 's|\(std::ofstream llOs(perFamilyLikelihoodPath);\)|\1\n    llOs << std::setprecision(17);|' \
      -e 's|\(ParallelOfstream ratesOs(globalRatesPath, true);\)|\1\n    ratesOs << std::setprecision(17);|' \
      "$AO"
  fi
  mkdir -p "$ALERAX_DST/build"
  ( cd "$ALERAX_DST/build" && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)" )
fi
# confirm this is AleRax_fixed (has the flag), not stock — we DO pass it (see below)
"$BIN" --help 2>&1 | grep -q "fixed-point-iterations" || { echo "not AleRax_fixed (no flag)"; exit 1; }

# 2) run each fixture at 64 fixed-point iterations (converged for any rate regime; the default 4
#    under-converges at high rates), vendor ref/
ITERS="${ITERS:-64}"
for f in test_trees_1 test_trees_2 test_trees_3 test_trees_200 test_mixed_200; do
  d="$DATA/$f"
  work="$(mktemp -d)"
  cp "$d/sp.nwk" "$d/g.nwk" "$d/families.txt" "$work/"
  ( cd "$work" && "$BIN" -f families.txt -s sp.nwk -p out --gene-tree-samples 0 \
      --species-tree-search SKIP --fixed-point-iterations "$ITERS" >alerax.log 2>&1 )
  mkdir -p "$d/ref"
  cp "$work/out/per_fam_likelihoods.txt" "$d/ref/"
  cp "$work/out/model_parameters/model_parameters.txt" "$d/ref/"
  echo "$f -> $(cat "$d/ref/per_fam_likelihoods.txt")"
  rm -rf "$work"
done
