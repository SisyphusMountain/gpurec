# Release Notes Template

This file holds user-facing release notes, known limitations for each release, and
upgrade notes for downstream consumers.

## How to use this file

- Create one section per release tag at the top of the file.
- Include at least:
  - Summary of major user-visible behavior changes.
  - Notable bug fixes and validation gates added or changed.
  - Dependency and Python/Torch/CUDA support updates.
  - Tested platform matrix and validation hardware/software notes.
  - Benchmark evidence summary and scope disclaimer.
  - Known limitations carried into the release.
  - Migration notes for config/CLI surface changes.
  - Release artifact notes (wheel/source expectations and native binaries).

## Template

## [VERSION] - YYYY-MM-DD

- Added
- Changed
- Fixed
- Known limitations
- Migration notes

Keep this file updated whenever `CHANGELOG.md` is appended.

## 0.1.0 - 2026-05-27

- Added
  - Baseline CLI workflow with `validate-config`, `optimize`, `sample`, and
    `run` preflight gates.
  - Production-route audit flags and checkpoint/summmary metadata inspection.
  - Release governance assets for citation and governance documentation.
- Changed
  - Standardized machine-readable status output and error reporting for installed
    CLI command smoke paths.
- Fixed
  - Artifact gate edge cases for mode/route mismatches and structured status
    formatting.
- Tested platform matrix and validation hardware/software notes
  - Linux x86_64, Python 3.12, PyTorch 2.6.0+cu124, CUDA 12.4, Triton 3.2.
- Benchmark evidence summary and scope disclaimer
  - Release validation evidence is benchmark evidence for this tested matrix,
    not a guaranteed performance contract for all hardware or datasets.
- Known limitations
  - CUDA path is required for the retained optimizer path and `S > 256` backward
    gate.
  - `examples/` and native Rust crate sources are present for source-archive
    smoke workflows, but wheel users still require external native preprocessing
    and backtracking artifacts.
- Migration notes
  - New `gpurec validate-inputs` surface is available for structured input
    validation without model execution.
  - New release governance files introduced:
    `CHANGELOG.md`, `CITATION.cff`, `LICENSE`, `Dockerfile`, and
    `docs/release-notes.md`.
