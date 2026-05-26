# Release Readiness

This checklist records remaining release blockers, manual release decisions,
and the packaging assumptions covered by current automated build steps.

## Required Before Redistribution

- Choose and add a project license.  After a `LICENSE` file exists, add matching
  `pyproject.toml` license metadata and a license classifier.  Project URLs and
  non-license classifiers are already present.
- Build source and wheel artifacts from a clean checkout and install them in a
  fresh environment with a PyTorch build that matches the target CUDA runtime.

## Sampling Binary Distribution Contract

The current release contract is explicit rather than unresolved:

- Wheels intentionally do not ship the Rust backtracking binary or Rust crate
  sources.  Installed `gpurec sample`, the sampling phase of `gpurec run`, and
  `gpurec backtrack-check` require a compatible prebuilt binary for
  `gpurec-backtrack`, selected with `GPUREC_BACKTRACK_BIN` or
  `--backtrack-binary`.
- Source archives include `crates/gpurec-backtrack/` and support the locked
  source-archive `cargo run` fallback.  That fallback requires Rust/Cargo and
  fetches the pinned `rustree` git dependency declared by the crate lockfile.
- Python backtracking helpers with native backend resolution may use a
  compatible external PyO3 library selected with `GPUREC_BACKTRACK_NATIVE_LIB`.
  This does not replace the CLI binary requirement for `gpurec sample`,
  `gpurec run`, or `gpurec backtrack-check`.
- Release notes and deployment docs must state the wheel-only external-binary
  requirement unless a future packaging change ships and smokes a bundled
  binary.  Any such change must update the README, package-data checks,
  installed-wheel smoke, and source-archive smoke together.

## Maintainer Build Path

The CPU GitHub Actions workflow includes a packaging job that installs
`.[release]` on Python 3.10 and 3.12, builds source and wheel artifacts, runs
`twine check`, installs the built wheel with existing runtime dependencies,
checks the source archive for current documentation, packaged examples, and
Rust preprocessing and backtracking crate sources, smokes those crates from the
unpacked source archive, runs source-archive `validate-config --check-preprocess`
on `examples/minimal-run-config.json` and
`examples/specieswise-adagrad-restarts-config.json`, and smokes both
`gpurec --help` and `python -m gpurec.cli --help`.  It also smokes installed
`gpurec config-template --help`,
`gpurec config-template --mode specieswise`, and
`gpurec validate-config --help` so the installed template/preflight surface
remains part of the command surface, including genewise Hessian-SGD route knobs
and the specieswise Adagrad restart schedule, and verifies the
`--require-cuda-backward-ready`, `--require-mode-default-optimizer`, and
`--require-production-default-route` preflight gates are exposed. These
mode-default optimizer gates remain separate from the stricter production-route
gate. It also smokes
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
`cuda_backward_ready=false`, and checks installed `gpurec sample --help`,
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
```

The checker validates required project metadata, required URLs and
classifiers, declared README files, and declared license metadata.  License
metadata may point at a file or provide inline text, but redistribution still
requires the top-level license file called out in the blocker list.

This check is currently expected to fail on the unresolved license blockers
listed above.  Do not bypass it for redistribution; choose a license, add the
top-level `LICENSE` file, and add matching `pyproject.toml` license metadata and
classifier first.

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
gpurec config-template --mode genewise
gpurec config-template --mode specieswise
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

Do not publish artifacts until the license blocker above is resolved.  The
sampling binary expectation above is the current release contract, not a
separate no-publish blocker.

## Lightweight Verification

Run these CPU-safe gates before release packaging:

```bash
CUDA_VISIBLE_DEVICES='' gpurec --help
CUDA_VISIBLE_DEVICES='' python -m gpurec.cli --help
CUDA_VISIBLE_DEVICES='' gpurec config-template --mode genewise
CUDA_VISIBLE_DEVICES='' gpurec config-template --mode specieswise
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
