# Release Readiness

This checklist records release requirements, manual release decisions, and the
packaging assumptions covered by current automated build steps.

## Required Before Redistribution

- The project license requirements are in place: top-level `LICENSE`, matching
  `pyproject.toml` license metadata, and an OSI MIT license classifier.
- Governance artifacts are present and current: `CHANGELOG.md`, `CITATION.cff`,
  `docs/release-notes.md`, `docs/support-policy.md`,
  `docs/versioning-policy.md`, `docs/publication-checklist.md`, and a
  CUDA-capable deployment recipe (`Dockerfile`).
- Offline installation policy is documented and current in
  `docs/platform-matrix.md`.
- Build source artifacts from a clean checkout and install them in a fresh
  environment with a PyTorch build that matches the target CUDA runtime.

## Preprocessing Native Extension Contract

The current release contract is explicit rather than unresolved:

- Source-only installation is the supported production path.
- Source archives include `crates/gpurec-preprocess/` and support the locked
  Cargo build fallback for the native extension. That fallback requires
  Rust/Cargo.
- `gpurec validate-config --check-preprocess`, `gpurec optimize`, `gpurec run`,
  and `gpurec preprocess-check` must fail with actionable diagnostics when the
  native preprocess artifact is unavailable or version-incompatible.

## Sampling Binary Distribution Contract

The current release contract is explicit rather than unresolved:

- Source-only installation is the supported production path.
- Source archives include `crates/gpurec-backtrack/` and support the locked
  `cargo run` fallback. That fallback requires Rust/Cargo and
  fetches the pinned `rustree` git dependency declared by the crate lockfile.
- Python backtracking helpers with native backend resolution may use a
  compatible external PyO3 library selected with `GPUREC_BACKTRACK_NATIVE_LIB`.
  This does not replace the CLI binary requirement for `gpurec sample`,
  `gpurec run`, or `gpurec backtrack-check`.
- `gpurec sample`, the sampling phase of `gpurec run`, and
  `gpurec backtrack-check` must fail with actionable diagnostics when the
  backtracking binary is unavailable or version-incompatible.

## Dependency Inventory And Vulnerability Scans

Release artifacts should include a reproducible dependency manifest before a
public artifact is published. Track it as `dist/dependency-inventory.json` when
building the wheel/source bundle:

```bash
python scripts/generate_dependency_inventory.py \
  --output dist/dependency-inventory.json \
  --check-git-dependency-pins
```

The inventory includes the Python dependency declaration block (`pyproject.toml`)
and locked Rust dependency entries from `crates/*/Cargo.toml` and
`crates/*/Cargo.lock`, including git source pins and resolved package entries.

Capture a dependency scan snapshot for publication audit history:

```bash
cat > dist/python-dependencies.txt <<'EOF'
torch>=2.0
triton>=2.1
EOF
python -m pip install pip-audit
pip-audit -r dist/python-dependencies.txt | tee dist/pip-audit.txt
```

If rustsec tooling is available in the build environment, include a locked lockfile
scan for both Rust crates:

```bash
if command -v cargo-audit >/dev/null 2>&1; then
  cargo-audit --file crates/gpurec-preprocess/Cargo.lock | tee dist/cargo-audit-preprocess.txt
  cargo-audit --file crates/gpurec-backtrack/Cargo.lock | tee dist/cargo-audit-backtrack.txt
else
  echo "cargo-audit unavailable; document the intentional gap in this release log"
fi
```

Keep these generated files as part of the internal release evidence directory for
post-incident comparison and reproducibility.

Generate and archive checksums for release artifacts:

```bash
sha256sum dist/* > dist/SHA256SUMS
```

Record release artifact provenance alongside checksums in the release evidence
bundle so maintainers can trace which clean checkout, toolchain, and commands
produced the published artifacts.

Before release handoff, add a quick artifact sanity pass over the final output
directory with the local validator script so structural regression in `summary.json`,
`history.jsonl`, checkpoint payloads, TSV exports, and `run_manifest.json` is
caught before packaging checks begin:

```bash
python scripts/validate_output_artifacts.py \
  --summary "${out_dir}/summary.json" \
  --history "${out_dir}/history.jsonl" \
  --checkpoint "${out_dir}/checkpoints/latest.pt" \
  --run-manifest "${out_dir}/run_manifest.json" \
  --tsv "${out_dir}/rates_final.tsv" \
  --tsv "${out_dir}/per_fam_likelihoods.tsv" \
  --json
```

For older one-off scripts or benchmark outputs, include only the `--tsv` and
`--summary` flags that match what is present for that workflow.

## Release Checklist Tiers

Use this checklist split before publication:

1. Quick PR checks:
   CPU unit gate, command-help smoke, parser/preflight smoke, and metadata
   checker in CI (`cpu-unit.yml`).
2. Nightly checks:
   scheduled GPU validation workflow (`gpu-validation.yml`) plus dependency
   inventory and audit snapshots.
3. Release-candidate checks:
   run `scripts/run_long_validation.py` on
   `docs/workflow-examples/end-to-end-tutorial/run.json`, archive the report,
   and verify it satisfies [validation-envelope.md](validation-envelope.md).
4. Final publication checks:
   clean-checkout build, source/wheel artifact checks, release notes update,
   changelog/citation/license confirmation, and attached release evidence bundle.

## Maintainer Build Path

GPU validation is enforced in `.github/workflows/gpu-validation.yml` for
`pull_request` and `push` events (protected branches `main` and `production`).
Those required runs must execute on a CUDA-capable self-hosted runner and fail
if CUDA is unavailable. The weekly scheduled run remains a non-blocking drift
check that can skip when no CUDA runtime is exposed. The required run also
enforces artifact-structure checks (`summary.json`, `history.jsonl`,
`run_manifest.json`, checkpoint, and TSV outputs) and numeric/time thresholds
for the realistic CUDA smoke fixture.

The CPU GitHub Actions workflow includes a packaging job that installs
`.[release]` on Python 3.10 and 3.12, builds source and wheel artifacts, runs
`twine check`, installs the built wheel with existing runtime dependencies,
checks the source archive for current documentation, packaged examples, and
Rust preprocessing and backtracking crate sources, smokes those crates from the
unpacked source archive, runs source-archive
`validate-config --require-mode-default-optimizer
--require-production-default-route --check-preprocess` on
`examples/minimal-run-config.json` and
`examples/specieswise-adagrad-restarts-config.json`, then confirms
`--require-cuda-backward-ready` fails both source-archive examples with
`cuda_backward_ready_reason=requires_s_gt_256` and no stdout, and smokes both
`gpurec --help` and `python -m gpurec.cli --help`. It also checks installed
`gpurec preprocess-check --help` and confirms a wheel-only environment without
`GPUREC_PREPROCESS_NATIVE_LIB` reports the native-extension requirement without a
traceback. It also smokes installed
`gpurec config-template --help`,
`gpurec config-template --mode specieswise`,
`gpurec config-template --mode global`, and
`gpurec validate-config --help` so the installed template/preflight surface
remains part of the command surface, including genewise Hessian-SGD route knobs,
the specieswise Adagrad restart schedule, and the global diagnostic template
boundary, and verifies the
`--require-cuda-backward-ready`, `--require-mode-default-optimizer`, and
`--require-production-default-route` preflight gates are exposed. These
mode-default optimizer gates remain separate from the stricter production-route
gate. It also generates genewise and specieswise configs from the installed
`gpurec config-template` command, points them at the checked tiny AleRax fixture
paths, and validates both with
`gpurec validate-config --require-mode-default-optimizer
--require-production-default-route --check-preprocess`, proving the installed
preflight can parse the referenced AleRax fixture files and reports
`preprocess_checked=true` with `cuda_backward_ready=false` for the tiny species
tree, then confirms `--require-cuda-backward-ready` fails those generated
configs with `cuda_backward_ready_reason=requires_s_gt_256` and no stdout. It
generates a global template too, then
checks that it passes `--require-mode-default-optimizer` as a mode-default
`adam` diagnostic and fails `--require-production-default-route` with a `mode`
mismatch. It also smokes
the installed-wheel preprocessing path by building the source-checkout
`crates/gpurec-preprocess` PyO3 extension and exporting
`GPUREC_PREPROCESS_NATIVE_LIB`, running `gpurec preprocess-check`, and only then
calling installed-wheel `--check-preprocess`, matching the wheel-only deployment
requirement for a compatible prebuilt native preprocessing extension. It also
smokes
`gpurec summary-info --help` and `gpurec checkpoint-info --help` so artifact
inspection stays available without CUDA model construction, including the
`--require-converged` and `--require-final-check-ok` summary gates for
automation, the mode-default optimizer audit gate, the production-route audit
gate, and the checkpoint final-check gate for direct artifact inspection.
The same package job checks `gpurec optimize --help` for the direct
optimization convergence, final-check, mode-default optimizer, and
production-route gates, including help text that names the objective,
likelihood/gradient route, rate parameterization, `final_check_iters_e`
evidence, and optimizer route. It checks `gpurec run --help` for the matching
pre-sampling convergence, final-check, mode-default optimizer, and
production-route gates, keeps
examples out of wheels while requiring them in the source
archive, verifies the minimal example config points to source-archive files,
checks source-archive preprocessing reports the tiny fixtures as
`cuda_backward_ready=false`, verifies the source-archive examples hard-fail
`--require-cuda-backward-ready` with empty stdout, and checks installed
`gpurec sample --help`,
`gpurec run --help`, and `gpurec backtrack-check` for external backtracking
binary guidance, direct sampling mode-default optimizer and production-route
gating, and the installed missing-binary diagnostic.

Install release tooling from the dedicated extra:

```bash
python -m pip install -e ".[release]"
```

Check metadata before building public artifacts:

```bash
python scripts/check_release_metadata.py
gpurec doctor
```

The checker validates required project metadata, required URLs and
classifiers, the `gpurec = "gpurec.cli:main"` console-script entry point,
declared README files, and declared license metadata.  License metadata may
point at a file or provide inline text.  With the blocker list requirements
met, this check should pass for redistribution with a complete license, project
URL, and script configuration before building public artifacts.

Build and inspect distribution artifacts from a clean checkout:

Before invoking the build module, preview ignored packaging artifacts and remove
only stale `build/`, `dist/`, or `*.egg-info/` directories that could shadow
release tooling or contaminate the artifact set:

```bash
git clean -Xdn -- build dist '*.egg-info'
```

Only after confirming the preview, remove those scoped ignored artifacts:

```bash
git clean -Xdf -- build dist '*.egg-info'
```

The repository intentionally ignores local caches, HOGENOM datasets, W&B runs,
generated profiling files, AleRax checkouts, and `rustree/`.

```bash
python -m build
python -m twine check dist/*
```

Smoke the installed wheel from outside the checkout so the import, public-export,
and command tests cannot accidentally use editable source files:

```bash
repo_root=$(git rev-parse --show-toplevel)
python -m pip uninstall -y gpurec
python -m pip install --no-deps dist/*.whl
smoke_dir=$(mktemp -d)
cd "$smoke_dir"
gpurec --help
python -m gpurec.cli --help
gpurec preprocess-check --help
unset GPUREC_PREPROCESS_NATIVE_LIB
set +e
gpurec preprocess-check > preprocess-check-missing.txt 2>&1
preprocess_check_status=$?
set -e
test "$preprocess_check_status" -eq 1
grep -q "GPUREC_PREPROCESS_NATIVE_LIB" preprocess-check-missing.txt
grep -q -- "--preprocess-native-lib" preprocess-check-missing.txt
gpurec config-template --mode genewise
gpurec config-template --mode specieswise
gpurec config-template --mode global
cargo build --release --locked --features python-extension \
  --manifest-path "$repo_root/crates/gpurec-preprocess/Cargo.toml"
export GPUREC_PREPROCESS_NATIVE_LIB="$repo_root/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so"
gpurec preprocess-check
gpurec config-template --mode genewise \
  --species-tree "$repo_root/examples/tiny/species.nwk" \
  --families-file "$repo_root/examples/tiny/families.txt" \
  --out-dir "$smoke_dir/generated-genewise-out" \
  --output generated-genewise-run.json
gpurec validate-config --config generated-genewise-run.json \
  --require-mode-default-optimizer \
  --require-production-default-route \
  --check-preprocess
set +e
gpurec validate-config --config generated-genewise-run.json \
  --require-mode-default-optimizer \
  --require-production-default-route \
  --check-preprocess \
  --require-cuda-backward-ready \
  > generated-genewise-cuda-ready.out \
  2> generated-genewise-cuda-ready.err
genewise_cuda_status=$?
set -e
test "$genewise_cuda_status" -eq 2
grep -q "cuda_backward_ready=false cuda_backward_ready_reason=requires_s_gt_256" \
  generated-genewise-cuda-ready.err
test ! -s generated-genewise-cuda-ready.out
gpurec config-template --mode specieswise \
  --species-tree "$repo_root/examples/tiny/species.nwk" \
  --families-file "$repo_root/examples/tiny/families.txt" \
  --out-dir "$smoke_dir/generated-specieswise-out" \
  --output generated-specieswise-run.json
gpurec validate-config --config generated-specieswise-run.json \
  --require-mode-default-optimizer \
  --require-production-default-route \
  --check-preprocess
set +e
gpurec validate-config --config generated-specieswise-run.json \
  --require-mode-default-optimizer \
  --require-production-default-route \
  --check-preprocess \
  --require-cuda-backward-ready \
  > generated-specieswise-cuda-ready.out \
  2> generated-specieswise-cuda-ready.err
specieswise_cuda_status=$?
set -e
test "$specieswise_cuda_status" -eq 2
grep -q "cuda_backward_ready=false cuda_backward_ready_reason=requires_s_gt_256" \
  generated-specieswise-cuda-ready.err
test ! -s generated-specieswise-cuda-ready.out
gpurec config-template --mode global \
  --species-tree "$repo_root/examples/tiny/species.nwk" \
  --families-file "$repo_root/examples/tiny/families.txt" \
  --out-dir "$smoke_dir/generated-global-out" \
  --output generated-global-run.json
gpurec validate-config --config generated-global-run.json \
  --require-mode-default-optimizer
set +e
gpurec validate-config --config generated-global-run.json \
  --require-production-default-route \
  > generated-global-production-route.out \
  2> generated-global-production-route.err
global_status=$?
set -e
test "$global_status" -eq 2
grep -q "config production default route fields differ for mode 'global': mode" \
  generated-global-production-route.err
test ! -s generated-global-production-route.out
gpurec summary-info --help
gpurec checkpoint-info --help
gpurec sample --help
gpurec run --help
gpurec backtrack-check --help
GPUREC_REPO_ROOT="$repo_root" python - <<'PY'
import os
from pathlib import Path
import gpurec
import gpurec.workflow as workflow

package_path = Path(gpurec.__file__).resolve()
repo_root = Path(os.environ["GPUREC_REPO_ROOT"]).resolve()
if package_path.is_relative_to(repo_root):
    raise SystemExit(f"imported gpurec from checkout: {package_path}")
if "site-packages" not in package_path.parts and "dist-packages" not in package_path.parts:
    raise SystemExit(f"gpurec import is not from installed packages: {package_path}")
for name in gpurec.__all__:
    getattr(gpurec, name)
if not set(gpurec.__all__) <= set(dir(gpurec)):
    raise SystemExit("top-level exports missing from dir(gpurec)")
for name in workflow.__all__:
    workflow_value = getattr(workflow, name)
    if name not in gpurec.__all__:
        raise SystemExit(f"workflow export missing from gpurec.__all__: {name}")
    if getattr(gpurec, name) is not workflow_value:
        raise SystemExit(f"top-level workflow export mismatch: {name}")
if not set(workflow.__all__) <= set(dir(workflow)):
    raise SystemExit("workflow exports missing from dir(gpurec.workflow)")
namespace = {}
exec("from gpurec.workflow import *", namespace)
exported = {name for name in namespace if not name.startswith("__")}
if exported != set(workflow.__all__):
    raise SystemExit(f"workflow wildcard mismatch: {sorted(exported ^ set(workflow.__all__))}")
print("exports_ok")
PY
```

Do not publish artifacts until the license and all command-surface checks in this
document are satisfied.  The sampling binary expectation above is the current
release contract, not a separate no-publish blocker.

## Lightweight Verification

Run these CPU-safe gates before release packaging:

```bash
CUDA_VISIBLE_DEVICES='' gpurec --help
CUDA_VISIBLE_DEVICES='' python -m gpurec.cli --help
CUDA_VISIBLE_DEVICES='' gpurec config-template --mode genewise
CUDA_VISIBLE_DEVICES='' gpurec config-template --mode specieswise
CUDA_VISIBLE_DEVICES='' gpurec preprocess-check --help
CUDA_VISIBLE_DEVICES='' gpurec optimize --help
CUDA_VISIBLE_DEVICES='' gpurec validate-config --config examples/minimal-run-config.json
CUDA_VISIBLE_DEVICES='' gpurec summary-info --help
CUDA_VISIBLE_DEVICES='' gpurec checkpoint-info --help
CUDA_VISIBLE_DEVICES='' gpurec run --help
CUDA_VISIBLE_DEVICES='' python - <<'PY'
import gpurec
import gpurec.workflow as workflow
for name in gpurec.__all__:
    getattr(gpurec, name)
for name in workflow.__all__:
    getattr(workflow, name)
namespace = {}
exec("from gpurec.workflow import *", namespace)
exported = {name for name in namespace if not name.startswith("__")}
if exported != set(workflow.__all__):
    raise SystemExit(f"workflow wildcard mismatch: {sorted(exported ^ set(workflow.__all__))}")
print("exports_ok")
PY
CUDA_VISIBLE_DEVICES='' gpurec backtrack-check --help
CUDA_VISIBLE_DEVICES='' pytest -q -m "unit and not gpu"
cargo test --locked --manifest-path crates/gpurec-backtrack/Cargo.toml
cargo run --locked --quiet --manifest-path crates/gpurec-backtrack/Cargo.toml -- --help
pytest -q tests/integration/test_rust_backtracking_fixture.py
```

GPU validation should start with a deliberately small family subset on a species
tree large enough for the current backward kernels (`S > 256`).  The tiny source
example is only a config/parser fixture, not an end-to-end optimizer smoke.  Run
memory-heavy HOGENOM or 1000-tree benchmark checks after that.

## Artifact Loading

Checkpoints and preprocessing caches are loaded through PyTorch's
`weights_only=True` path.  Preprocessing caches are also checked for nested
species and family payload structure.  Regenerate legacy preprocessing caches if
safe loading or cache validation rejects them.  Treat old pickle-only checkpoints
as trusted migration inputs rather than loading them through normal CLI
workflows.

Optimization directories also include `theta_final.pt` as a raw tensor export
for inspection and custom analysis.  Do not document it as a resumable or
sampling checkpoint: it does not carry run configuration, family ordering, or
species ordering metadata.  Use `checkpoints/best.pt` or
`checkpoints/latest.pt` for workflows that restore parameters into a model.
