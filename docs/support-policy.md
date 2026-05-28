# Support Policy

This document defines what the project maintainers currently support for
production use and how long fixes are expected for active releases.

## Scope

The supported user-facing surface is the contract documented in
[`api-contract.md`](api-contract.md): CLI commands, top-level Python workflow
exports, documented environment variables, and output artifact schemas.

Internal modules (especially `gpurec.core`) are not covered by compatibility
guarantees unless explicitly documented as public.

## Supported Environments

The supported environment matrix is versioned in
[`platform-matrix.md`](platform-matrix.md). Production support assumes:

- Linux x86_64 runtime.
- Python 3.10-3.12.
- CUDA-capable GPU runtime with compatible PyTorch and Triton.
- Source-based installation with Rust/Cargo available to build native artifacts.

## Support Window

The active support window tracks explicitly documented versions for:

- Python runtime versions.
- PyTorch runtime compatibility.
- CUDA runtime/toolkit compatibility.
- Native artifact versions (preprocessing and backtracking).

The latest release tag and [`platform-matrix.md`](platform-matrix.md) define
the current support window bounds for production triage.

## Release And Patch Policy

- `production` is the active stabilization branch.
- The latest release tag is the primary supported release line.
- Security, correctness, or reproducibility issues found in the latest release
  are fixed in `production` first, then included in the next release tag.
- Older tags may receive backports at maintainer discretion; they are not
  guaranteed.

## Support Expectations

- User-facing regressions in documented CLI/API behavior, output contracts, or
  release gates are treated as support issues.
- Requests outside the published platform matrix or outside the public contract
  are best-effort only.
- Known limitations are tracked in
  [`known-limitations.md`](known-limitations.md) and repeated in release notes.

## Evidence Required For Support

Bug reports should include:

- Installed `gpurec` version/tag and installation method.
- `gpurec doctor --json` output.
- Command line used and full stderr/stdout.
- Relevant artifacts (`run_config.json`, `summary.json`, `run_manifest.json`,
  and checkpoint metadata when applicable).

Without this evidence, maintainers may request additional diagnostics before
triage.
