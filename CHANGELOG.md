# Changelog

All notable project changes are recorded here.

## Unreleased

- Added a first-pass release governance layer: project-level citation and changelog
  files, container guidance, and explicit release policy references in release
  docs.
- Expanded release-readiness and CLI workflows to report structured validation,
  artifact evidence, and machine-readable preflight diagnostics.
- Added lightweight `validate-inputs` command and JSON output formats for validation
  and artifact-introspection paths.

## 0.1.0 - 2026-05-27

- Initial public-capability release for the CUDA-powered AleRax-style workflow.
- Public workflow now covers `validate-config`, `optimize`, `summary-info`,
  `checkpoint-info`, `sample`, `run`, and native preprocessing/backtracking checks.
- Added route metadata, optimizer-gate support, and stricter artifact contracts for
  automation.
- Added structured preflight checks for CUDA backward readiness and production route
  auditing.

