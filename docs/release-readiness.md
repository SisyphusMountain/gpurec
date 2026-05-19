# Release Readiness

This checklist records release blockers and packaging assumptions that are not
yet encoded as automated build steps.

## Required Before Redistribution

- Choose and add a project license.  After a `LICENSE` file exists, add matching
  `pyproject.toml` license metadata, classifiers, and project URLs.
- Decide the Rust backtracking release model.  The Python package currently
  expects a compiled sampler via `GPUREC_BACKTRACK_BIN` or `--backtrack-binary`;
  the source-tree `cargo run` fallback depends on a local `rustree/` checkout.
- Build source and wheel artifacts from a clean checkout and install them in a
  fresh environment with a PyTorch build that matches the target CUDA runtime.

## Maintainer Build Path

Install release tooling from the dedicated extra:

```bash
python -m pip install -e ".[release]"
```

Build and inspect distribution artifacts from a clean checkout:

```bash
python -m build
python -m twine check dist/*
```

Do not publish artifacts until the license and Rust backtracking release model
above are resolved.

## Source Checkout Hygiene

- Preview ignored generated files before building local archives or containers:

```bash
git clean -Xdn
```

- Remove ignored artifacts only after confirming the preview:

```bash
git clean -Xdf
```

The repository intentionally ignores local caches, HOGENOM datasets, W&B runs,
generated profiling files, AleRax checkouts, and `rustree/`.

## Lightweight Verification

Run these CPU-safe gates before release packaging:

```bash
python -m gpurec.cli --help
CUDA_VISIBLE_DEVICES='' pytest -q -m "unit and not gpu"
```

GPU validation should use a small species tree and a small family subset before
running memory-heavy HOGENOM or 1000-tree benchmark checks.

## Artifact Loading

Checkpoints and preprocessing caches are loaded through PyTorch's
`weights_only=True` path and checked for expected dictionary keys.  Regenerate
legacy preprocessing caches if safe loading rejects them.  Treat old pickle-only
checkpoints as trusted migration inputs rather than loading them through normal
CLI workflows.
