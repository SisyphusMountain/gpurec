# Versioning Policy

This project uses semantic versioning for published tags and package releases:
`MAJOR.MINOR.PATCH`.

## Version Semantics

- `MAJOR`: incompatible user-facing contract changes (CLI flags/behavior,
  documented Python API surface, or output artifact schema contracts).
- `MINOR`: backward-compatible feature additions or new documented workflow
  capabilities.
- `PATCH`: backward-compatible bug fixes, diagnostics improvements, and release
  process hardening that do not break the documented public contract.

## Compatibility Commitments

- The public contract is defined in [`api-contract.md`](api-contract.md).
- Deprecated user-facing behavior is announced in release notes before removal.
- Breaking changes require an explicit migration note in
  [`release-notes.md`](release-notes.md).

## Release Line Policy

- `production` is the active stabilization branch for the next release.
- The latest release tag is the primary supported line.
- Backports to older tags are best-effort and not guaranteed.

## Artifact And Metadata Consistency

Every release candidate must keep version identifiers aligned across:

- `pyproject.toml` package version.
- `gpurec.__version__`.
- release notes heading for the published tag.

Release checklists must include this consistency check before publication.
