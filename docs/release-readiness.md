# Release Readiness

This checklist records release blockers and packaging assumptions that are not
yet encoded as automated build steps.

## Required Before Redistribution

- Choose and add a project license.  After a `LICENSE` file exists, add matching
  `pyproject.toml` license metadata and a license classifier.  Project URLs and
  non-license classifiers are already present.
- Decide the Rust backtracking binary distribution model.  The Python package
  supports a compiled sampler via `GPUREC_BACKTRACK_BIN` or
  `--backtrack-binary`; the source-tree `cargo run` fallback uses a locked
  Cargo build and the pinned `rustree` git dependency.
- Build source and wheel artifacts from a clean checkout and install them in a
  fresh environment with a PyTorch build that matches the target CUDA runtime.

## Maintainer Build Path

The CPU GitHub Actions workflow includes a packaging job that installs
`.[release]`, builds source and wheel artifacts, runs `twine check`, installs
the built wheel with existing runtime dependencies, checks the source archive
for packaged C++ preprocessing sources, and smokes both `gpurec --help` and
`python -m gpurec.cli --help`.

Install release tooling from the dedicated extra:

```bash
python -m pip install -e ".[release]"
```

Check metadata before building public artifacts:

```bash
python scripts/check_release_metadata.py
```

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

Do not publish artifacts until the license and binary distribution expectation
for sampling above are resolved.

## Lightweight Verification

Run these CPU-safe gates before release packaging:

```bash
CUDA_VISIBLE_DEVICES='' gpurec --help
CUDA_VISIBLE_DEVICES='' python -m gpurec.cli --help
CUDA_VISIBLE_DEVICES='' pytest -q -m "unit and not gpu"
cargo test --locked --manifest-path crates/gpurec-backtrack/Cargo.toml
cargo run --locked --quiet --manifest-path crates/gpurec-backtrack/Cargo.toml -- --help
pytest -q -m "integration and not gpu"
```

GPU validation should use a small species tree and a small family subset before
running memory-heavy HOGENOM or 1000-tree benchmark checks.

## Artifact Loading

Checkpoints and preprocessing caches are loaded through PyTorch's
`weights_only=True` path.  Preprocessing caches are also checked for nested
species and family payload structure.  Regenerate legacy preprocessing caches if
safe loading or cache validation rejects them.  Treat old pickle-only checkpoints
as trusted migration inputs rather than loading them through normal CLI
workflows.
