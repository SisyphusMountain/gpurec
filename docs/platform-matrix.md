# Production Platform Matrix

This page defines what this repository explicitly supports for production use and
what combinations are expected to work based on CI and release checks.

## Supported Runtime Matrix

| Component | Supported Versions / Constraints | Practical Note |
| --- | --- | --- |
| Linux OS | Linux (x86_64) | `Environment :: OS only: POSIX :: Linux` is the published package scope. |
| Python | `>= 3.10, < 3.13` | `3.10`, `3.11`, and `3.12` are the tested package CI matrix entries. |
| PyTorch | `torch >= 2.0` | Must match the local CUDA runtime installed in the target environment. |
| Triton | `triton >= 2.1` | Core dependency for CUDA execution. |
| Hardware | NVIDIA CUDA GPU | GPU is required for the production likelihood path; no full CPU fallback is published. |
| Native preprocess | `GPUREC_PREPROCESS_NATIVE_LIB` (wheels) or source crate build | Installed wheels never include compiled preprocessing artifacts. Source builds may use Cargo fallback from `crates/gpurec-preprocess`. |
| Native backtracking | `GPUREC_BACKTRACK_BIN` (wheels) or source crate build | Installed wheels never include the CLI artifact. Source builds may use Cargo fallback from `crates/gpurec-backtrack`. |
| Rust toolchain | stable toolchain available (for source builds/CLI artifacts) | Source checkout and source-archive workflows use `cargo` for Rust builds. |

## Installation Matrix

| Workflow | Linux | Python | Native requirements | Expected command path |
| --- | --- | --- | --- | --- |
| Source checkout / source archive | ✅ | `3.10` `3.11` `3.12` | `cargo` for native crate builds. | `pip install .` or `pip install -e ".[dev]"` then use `gpurec` CLI/`GPUREC_*` vars as needed. |
| Source-only install (data + GPU) | ✅ | `3.10` `3.11` `3.12` | native artifacts installed via env vars or built from crates | `pip install .` |
| Wheel install | ✅ | `3.10` `3.12` (package CI path) | set `GPUREC_PREPROCESS_NATIVE_LIB` and `GPUREC_BACKTRACK_BIN` before running runtime commands | `pip install --no-deps dist/*.whl` |
| Container image (`Dockerfile`) | ✅ | image-local Python pinned by image base | image prebuilds native artifacts from source | `docker build -t gpurec .` |

## Why these constraints exist

- The production likelihood and optimizer route is CUDA-first and GPU-only today.
- Native preprocess and backtracking are externalized for wheel installs by design.
- The parser and CUDA kernels are validated against the current production route and
  platform in repository release checks.

## Compatibility Gates (Recommended Before Long Runs)

- `gpurec doctor` (single readiness command)
- `gpurec preprocess-check` and `gpurec backtrack-check`
- `gpurec validate-config --config ... --check-preprocess --require-cuda-backward-ready`

Run these once before starting an optimization run so environment and artifact
assumptions fail fast.
