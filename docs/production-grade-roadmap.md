# Production-Grade Roadmap For Bioinformatics Users

Date: 2026-05-27

This document summarizes what still has to be done before `gpurec` should be
treated as production-grade software for bioinformaticians. It is based on the
current public README, packaging metadata, workflow documentation, release notes,
CI configuration, native Rust crate layout, tests README, scripts ownership
notes, profiling ownership notes, and the repo-wide audit material already in
this repository.

The project is not starting from zero. It already has a supported CLI surface,
flat JSON workflow configs, CPU-safe preflight validation, checkpointed
optimization, output artifact contracts, RecPhyloXML sampling support, route
metadata, a CPU unit CI workflow, Rust backtracking tests, source archive
packaging checks, and substantial internal audit notes. The remaining work is
mostly about making those capabilities distributable, reliable outside this
checkout, understandable to biologists, and defensible under repeated use on real
datasets.

## Current Readiness Summary

| Area | Current state | Production gap |
| --- | --- | --- |
| Core workflow | `gpurec optimize`, `validate-config`, `sample`, `run`, `summary-info`, and `checkpoint-info` exist and are documented. | Needs complete end-to-end release validation on real-sized public fixtures, not only parser/config fixtures and local HOGENOM evidence. |
| Installation | Python metadata exists with Python 3.10-3.12 support and a console script. | License metadata is unresolved, wheels do not bundle native preprocessing or backtracking artifacts, and users must manually provide compatible binaries/libraries. |
| Native preprocessing | Rust preprocessing crate is present and source archives can build it. | Binary distribution, version compatibility, platform matrix, and user-facing diagnostics need to be productized. |
| Backtracking and RecPhyloXML | Rust backtracking crate and Python helpers exist; CPU-safe fixture tests exist. | Binary distribution and reproducible sampling examples need to be easier for installed users. |
| CUDA likelihood path | Optimized CUDA/Triton path is the production route. | No CPU fallback, no small-species production fallback, and no public GPU CI/performance gate visible from the inspected workflow. |
| Documentation | Strong operator docs exist for config, inputs, outputs, troubleshooting, and optimization. | Needs a shorter bioinformatician quickstart, installation decision tree, real dataset tutorial, format conversion guidance, and explicit limitation page. |
| Tests | CPU unit gate, packaging checks, Rust backtracking tests, and targeted CUDA smoke commands are documented. | Needs mandatory GPU CI or scheduled GPU validation, public benchmark fixtures, golden correctness comparisons, and performance regression thresholds. |
| Repository hygiene | Internal/public surface classification and legacy script ownership are documented. | Legacy HOGENOM scripts, notebooks, and profiling helpers still need migration, deletion, or hard separation from the user product. |
| Release governance | Release-readiness checklist and metadata checker exist. | Needs license, citation, changelog, versioning policy, release notes, binary provenance, and support policy. |

## Definition Of Done

For this repository to be production-grade for bioinformaticians, a new user
should be able to do the following without source-code help:

1. Install the package and required native artifacts on a supported Linux GPU
   machine.
2. Confirm the installation with one command that reports Python, PyTorch,
   CUDA, Triton, preprocessing, and backtracking readiness.
3. Prepare ordinary species and gene tree inputs using documented formats and
   receive actionable validation errors.
4. Run an optimization from a generated config on a realistic dataset.
5. Resume interrupted runs without guessing which checkpoint is valid.
6. Inspect convergence, final likelihood, rates, route metadata, and warnings
   from stable output files.
7. Sample reconciliations to RecPhyloXML with reproducible RNG settings.
8. Use the outputs in downstream scripts, workflow managers, or notebooks
   without depending on private module internals.
9. Compare a run against documented quality gates and known limitations.
10. Cite, version, and archive the exact toolchain used for a publication.

## P0 Blockers Before Public Redistribution

These items should block any public release or recommendation to external
bioinformatics users.

| Blocker | Why it matters | Acceptance gate |
| --- | --- | --- |
| Add a license | The current release-readiness notes identify license selection as unresolved. Without a license, downstream labs cannot safely redistribute, package, or cite the software. | Add top-level `LICENSE`, matching `pyproject.toml` license metadata, classifier, and make `scripts/check_release_metadata.py` pass. |
| Decide native artifact distribution | Installed wheels currently require external `GPUREC_PREPROCESS_NATIVE_LIB` and `GPUREC_BACKTRACK_BIN`. This is workable for maintainers but high-friction for users. | Choose one supported release model: bundled platform wheels, separately versioned native wheels/assets, or documented source-only installation. Test it from outside the checkout. |
| Publish a supported platform matrix | Users need to know which Python, PyTorch, CUDA, Triton, GPU, OS, Rust, and compiler combinations are supported. | Add a matrix to README/docs and test at least the advertised primary configuration. |
| Add installation verification | Users need one diagnostic command before spending GPU hours. | Provide `gpurec doctor` or extend `preprocess-check`/`backtrack-check` into a single readiness report covering Python package, PyTorch CUDA, Triton kernels, native preprocessing, backtracking, writable output directory, and version metadata. |
| Establish GPU validation | The inspected CI workflow is CPU-focused. Production correctness depends on CUDA kernels and GPU memory behavior. | Add mandatory or scheduled GPU CI for the retained likelihood/gradient path, small CUDA fixtures, and at least one realistic-size benchmark fixture or replay. |
| Resolve small-species behavior | The docs state the retained Pi backward path requires `S > 256`; tiny examples are parser/config fixtures, not optimizer smokes. This will surprise users with small species trees. | Either implement a CPU/Torch/Triton fallback for `S <= 256`, or make the limitation impossible to miss in install docs, quickstart, templates, validation errors, and publication guidance. |
| Ship a real end-to-end tutorial | Existing examples intentionally do not run optimization end to end. Bioinformaticians need one complete path. | Add a tracked or downloadable dataset that passes `validate-config --check-preprocess --require-cuda-backward-ready`, runs optimization, writes outputs, and samples RecPhyloXML. |
| Freeze public API and CLI contract | The repo has careful internal/public classification, but production users need stability. | Declare supported CLI commands, Python imports, config fields, output schemas, environment variables, and deprecation policy in one versioned API contract. |

## P1 Work Needed For A Usable Bioinformatics Tool

### Installation And Packaging

Production users should not have to reverse-engineer native library setup from
environment variables.

Required work:

- Provide installation paths for at least these cases: source checkout for
  developers, wheel install for users, container image for HPC clusters, and
  source archive build for reproducibility.
- Decide whether `gpurec-preprocess` and `gpurec-backtrack` are bundled,
  downloaded as release assets, or installed as separate packages.
- Add version handshake checks between Python, preprocessing native library,
  and backtracking binary.
- Make native artifact errors include expected version, discovered version,
  selected path, and exact remediation.
- Provide a minimal Dockerfile or Apptainer/Singularity recipe for CUDA
  deployments.
- Consider a conda package or documented conda environment because many
  bioinformatics labs standardize on conda/mamba.
- Pin or constrain PyTorch/Triton compatibility more explicitly than
  `torch>=2.0` and `triton>=2.1` if kernel behavior depends on specific ranges.
- Document whether offline installation is supported.

Acceptance gates:

- Fresh environment install from wheel succeeds outside the checkout.
- `gpurec doctor` succeeds on the supported GPU image.
- Wheel/source archive smoke tests cover missing and valid native artifacts.
- Release artifacts include checksums and clear provenance.

### Input Preparation And Validation

The existing `input-preparation.md` is useful, but a production bioinformatics
workflow needs stronger bridges from real data to `gpurec` inputs.

Required work:

- Add a concise "bring your own dataset" guide that starts from common species
  tree, gene tree, and mapping files.
- Provide examples for AleRax-style `[FAMILIES]` files with multiple families,
  multiple trees per family, and mapping files.
- Add conversion or validation helpers for common mapping conventions used by
  Treerecs, GeneRax, AleRax, OrthoFinder-style names, and simple
  `gene -> species` TSVs.
- Make Newick parser limitations prominent: unsupported quoted labels,
  comments, NHX/BEAST metadata, embedded delimiters, unary species nodes, and
  non-binary species trees should be listed in a single limitation section.
- Add a `validate-inputs` style command or expand `validate-config` output so
  users can request a structured JSON report of every family, missing mapping,
  duplicate name, rejected tree, and species coverage issue.
- Add examples of good and bad validation output with explanations.
- Provide guidance for large family sets: how to sample the first `N` families,
  how to estimate memory, and how to choose `clade_budget` and
  `family_chunk_size`.

Acceptance gates:

- A user can validate a dataset without constructing a CUDA model.
- Every common input failure includes file path, family name, affected label,
  expected format, and next action.
- A small public input fixture covers valid mapping, missing mapping, duplicate
  family names, malformed Newick, and unsupported species topology.

### User-Facing Workflow

The CLI is already the right production entry point. The remaining work is to
make it feel safe and predictable for non-developers.

Required work:

- Add a "first successful run" tutorial that uses only public commands.
- Add explicit run lifecycle docs: create config, validate, run, resume,
  inspect, sample, archive.
- Provide recommended defaults by user goal: exploratory run, production
  genewise run, production specieswise run, diagnostics-only global run.
- Add structured JSON output mode for `validate-config`, `summary-info`,
  `checkpoint-info`, and `doctor`, in addition to human-readable status lines.
- Document exit codes consistently for workflow managers.
- Provide Snakemake and Nextflow examples that use preflight, optimization,
  summary gates, and sampling.
- Add a run manifest artifact that records package version, native artifact
  versions, PyTorch version, CUDA availability, GPU name, CLI command, config,
  random seeds, and selected route.
- Make RNG behavior explicit for sampling and any stochastic optimizer steps.

Acceptance gates:

- A Snakemake or Nextflow example can fail fast on bad config, resume from a
  checkpoint, and reject non-converged outputs.
- A completed run directory contains enough metadata to reproduce or audit the
  run later.
- Users never need to pass a raw `theta_final.pt` where a checkpoint is required
  because docs and diagnostics make the distinction obvious.

### Documentation For Bioinformaticians

The current docs are strong for maintainers and operators, but still dense for
domain users.

Required work:

- Add a short README path: "Install", "Validate inputs", "Run optimization",
  "Inspect output", "Sample reconciliations".
- Move highly technical optimizer-route details behind an "advanced" section.
- Add a glossary for D, T, L, DTL, CCP, specieswise, genewise, global,
  RecPhyloXML, NLL, route, solver budget, and checkpoint.
- Add visual diagrams for the input/output flow and run directory structure.
- Add example output snippets for `summary.json`, `rates_final.tsv`,
  `per_fam_likelihoods.tsv`, and RecPhyloXML outputs.
- Add a known-limitations page with CUDA-only status, `S > 256`, Newick subset,
  wheel-native-artifact requirements, and experimental bf16 scope.
- Add a troubleshooting section by symptom rather than internal subsystem.
- Add citation and publication checklist docs once license and citation metadata
  are resolved.

Acceptance gates:

- A bioinformatician who knows tree reconciliation concepts but not this codebase
  can complete the tutorial without reading Python source.
- A lab engineer can install and validate the tool from docs alone.
- The docs distinguish stable user workflows from HOGENOM-only research scripts.

## P1 Correctness And Validation Work

### GPU Correctness

The production value of this project depends on CUDA/Triton kernels, implicit
gradients, batching, and solver behavior. CPU-only tests cannot fully validate
that.

Required work:

- Add GPU CI or scheduled GPU jobs for the retained forward/backward path.
- Test `GeneReconModel` forward/backward modes on CUDA across supported Python
  and PyTorch versions.
- Add numerical parity tests between resident batches, chunked/global mode, and
  reference small implementations where possible.
- Add golden likelihood and gradient fixtures for known tree sets.
- Compare final likelihoods and rates against AleRax or another trusted
  reference on at least one documented public dataset.
- Test deterministic behavior for fixed seeds and fixed hardware where
  deterministic behavior is claimed.
- Add stress tests for memory pressure, oversized families, batch planning,
  resume, failed final validation, and sampling checkpoint selection.
- Add direct coverage for kernel edge cases that are currently only covered
  through higher-level integration tests.

Acceptance gates:

- Every supported production route has a GPU test that exercises at least one
  full objective and gradient evaluation.
- Every release runs a documented GPU validation bundle before publication.
- Numerical tolerances are documented and enforced.

### Real Dataset Validation

Local HOGENOM evidence is useful but not enough for external users unless the
data and commands are reproducible.

Required work:

- Select one or more redistributable datasets or provide a deterministic
  generator for realistic tree/family scale.
- Archive benchmark inputs or document how to fetch them.
- Record expected family/species counts, runtime envelope, peak memory, final
  NLL range, convergence status, and sampling output shape.
- Provide a "long validation" workflow that maintainers run before releases.
- Keep HOGENOM-specific evidence as provenance, but avoid making it the only
  evidence users can reproduce.

Acceptance gates:

- A fresh machine can reproduce the documented validation run.
- The validation run fails on known-bad changes to likelihood, gradient, input
  parsing, output publication, or sampling.
- Public docs distinguish "benchmark evidence" from "guaranteed performance".

### Parser And Format Robustness

Bioinformatics inputs are messy. The parser can remain intentionally narrow, but
the validation layer needs to protect users.

Required work:

- Add fuzz or property tests for accepted and rejected Newick subsets.
- Add tests for mapping files with duplicate genes, duplicate species mappings,
  unknown species, empty files, whitespace variants, and large family manifests.
- Add tests for path resolution from config files, stdin configs, and CLI flags.
- Add schema-style validation for `summary.json`, checkpoint metadata,
  `history.jsonl`, and TSV outputs.
- Add compatibility checks for RecPhyloXML output with downstream validators if
  such tools are available.

Acceptance gates:

- Bad input fails before GPU allocation.
- Error messages are stable enough to support workflow-manager triage.
- Output files are parseable by simple downstream scripts without special-case
  handling for `NaN`, `Infinity`, or malformed rows.

## P2 Engineering Quality Work

### Public API And Internal Surface

The project already documents that `gpurec.core` is internal. Production quality
requires enforcing that boundary.

Required work:

- Keep the top-level `gpurec` exports small and documented.
- Add a compatibility policy for config fields, CLI flags, Python imports, and
  output files.
- Avoid promoting internal helpers just because tests import them.
- Add deprecation warnings and migration notes before removing supported
  surfaces.
- Move reusable behavior out of legacy HOGENOM scripts into `gpurec.workflow`
  before changing or deleting those scripts.
- Continue reducing duplicate evaluator, scheduler, optimizer, and parameter
  shape paths as described in the existing simplification notes.

Acceptance gates:

- Public API docs match `gpurec.__all__`, `gpurec.workflow.__all__`, CLI help,
  and README examples.
- Direct imports from `gpurec.core` are never required in user tutorials.
- Legacy scripts have explicit keep, migrate, or delete decisions.

### Type Safety, Linting, And Static Checks

The inspected metadata includes pytest and release tooling, but no visible
static type or linting gate.

Required work:

- Add a formatter and linter policy, for example Ruff plus a minimal configured
  rule set.
- Add type-checking for public configuration, workflow, and artifact parsing
  code, even if kernel-heavy internals are exempt initially.
- Add import hygiene checks so optional heavy dependencies do not load during
  `gpurec --help` or metadata inspection.
- Add docstring requirements for public APIs and user-facing exceptions.
- Add a generated CLI help snapshot or parser-level assertions for important
  flags and exit behavior.

Acceptance gates:

- CI rejects obvious lint errors, unused imports, broken type annotations in
  public modules, and accidental public-surface drift.
- Help and import smoke tests remain CPU-safe.

### Release Process

The release-readiness document is detailed. The next step is to convert it into
a repeatable release procedure.

Required work:

- Add `CHANGELOG.md`.
- Add `CITATION.cff` or equivalent citation metadata.
- Define semantic versioning expectations.
- Define support window for Python, PyTorch, CUDA, and native artifact versions.
- Add signed or checksummed release artifacts.
- Add release notes template with known limitations and migration notes.
- Add a maintainer checklist that separates quick PR checks, nightly checks,
  release-candidate checks, and final publication.

Acceptance gates:

- A release candidate can be built from a clean checkout by following one
  checklist.
- The package metadata checker passes.
- Release notes state exactly how native preprocessing and backtracking are
  provided.

### Dependency And Supply Chain Hygiene

The Rust crates pin `rustree` by git revision, and Python dependencies are broad.
Production users need reproducibility.

Required work:

- Decide whether git dependencies are acceptable for releases or should be
  vendored, published, or mirrored.
- Document how Cargo lockfiles and Python dependency ranges are updated.
- Add dependency vulnerability scanning if releases are public.
- Add an SBOM or at least a dependency inventory for Python and Rust artifacts.
- Ensure source archives and wheels do not include build artifacts such as
  `target/`, caches, or local data.

Acceptance gates:

- A release can be rebuilt from source archive without relying on unpinned
  moving targets.
- Binary artifacts can be traced to source revisions and build settings.

## P2 Reliability And Operations

### Performance Regression Control

The repo contains profiling tools and performance logs, but production needs
defined performance gates.

Required work:

- Define benchmark tiers: quick smoke, PR benchmark, nightly benchmark, release
  benchmark.
- Track wall time, GPU memory, objective/gradient time, preprocessing time,
  sampling time, and output publication time.
- Add baseline JSON outputs for benchmark scripts so regressions can be
  detected automatically.
- Define acceptable variance and fail thresholds.
- Keep HOGENOM-specific benchmarks separate from portable benchmarks.
- Include OOM and fallback behavior in benchmarks, not just successful fast
  paths.

Acceptance gates:

- Release candidates cannot regress key benchmark medians beyond a documented
  threshold without an explicit note.
- Performance documentation lists hardware, software versions, dataset, and
  command.

### Failure Recovery

Checkpointing and staged publication already exist. The next step is to make
failure recovery obvious and exhaustively tested.

Required work:

- Test interrupted optimization, resume with unchanged config, resume with
  incompatible config, and resume after partial final artifact publication.
- Document exactly which files are authoritative after failure.
- Add CLI commands for run-directory inspection and repair if needed.
- Make all failure states machine-readable in summaries and checkpoints.
- Add examples for nonconvergence, final-check failure, nonfinite objective,
  missing native binary, and insufficient CUDA memory.

Acceptance gates:

- A user can recover or safely discard a failed run without inspecting Python
  source.
- Workflow managers can distinguish retryable failures from input contract
  failures.

### HPC And Multi-User Environments

Bioinformatics production often runs on clusters.

Required work:

- Document GPU memory requirements and how to estimate them before launch.
- Add Slurm examples with environment modules, CUDA visibility, and output
  paths.
- Support or document thread controls for preprocessing and PyTorch.
- Add guidance for local scratch directories versus shared network storage.
- Make cache and temporary-file locations configurable where relevant.
- Ensure logs are useful when stdout/stderr are captured by schedulers.

Acceptance gates:

- A Slurm example can validate inputs, run optimization, resume, and sample.
- The docs explain what to collect when asking for support.

## P3 Bioinformatics Product Features

These are not necessarily release blockers, but they would materially reduce
friction for biological users.

### Format Interoperability

Recommended work:

- Add import helpers for common family manifest and mapping formats.
- Add optional exporters for rate tables in formats expected by common
  phylogenetic tooling.
- Add RecPhyloXML validation and summary commands.
- Add species/gene label normalization reports.
- Add warnings for suspicious biological inputs, such as many unmapped genes,
  singleton-heavy families, extreme tree imbalance, duplicate labels, or
  species not represented by any gene.

### Analysis Utilities

Recommended work:

- Add a stable command to summarize rates by family or species.
- Add plots or CSV summaries for convergence and rate distributions.
- Add commands to compare two run directories.
- Add commands to compare `gpurec` rates or likelihoods against AleRax outputs
  when those outputs are available.
- Add an example notebook that consumes stable artifacts only, not private
  source-checkout internals.

### UX Improvements

Recommended work:

- Add progress estimates during preprocessing and optimization.
- Add more readable terminal summaries for long runs.
- Add suggestions in errors, not just assertions.
- Add `--dry-run` mode that estimates route, inputs, counts, and memory without
  optimization.
- Add `--explain-config` to show effective defaults and why they were chosen.

## P3 Algorithmic And Runtime Improvements

These items improve robustness and broaden applicability but should be scheduled
behind the release blockers.

Recommended work:

- Implement or restore a small-species backward fallback for `S <= 256`.
- Decide whether a CPU likelihood path is needed for education, smoke tests, or
  small datasets, even if it is not performance competitive.
- Revisit the experimental bf16 direct API and either promote it with tests or
  keep it clearly out of workflow configs.
- Continue simplifying duplicated global/specieswise/genewise evaluator paths.
- Consolidate chunked/global uniform evaluation with the shared evaluator where
  practical.
- Keep atomics and Triton/CUDA kernel optimization work behind correctness and
  benchmark gates.
- Add stronger numerical diagnostics around E-adjoint and Pi-adjoint convergence
  so users know when solver approximations are suspect.

## P4 Cleanup And Repository Shape

This work reduces long-term maintenance cost.

Recommended work:

- Delete committed bytecode caches and ensure they remain ignored.
- Audit `tests/data` for local HOGENOM data, generated trees, tarballs, and
  outputs. Keep only redistributable fixtures or documented generators.
- Move historical logs that are no longer actionable into an archive section or
  mark them clearly as historical.
- Convert retained experiment scripts into supported CLI features or archive
  them.
- Keep profiling artifacts ignored and move durable findings into docs.
- Split maintainer-only notes from user-facing docs.

Acceptance gates:

- `docs/README.md` points users to current material first and marks historical
  material clearly.
- A source archive contains only intentional source, docs, fixtures, and crates.
- User tutorials do not mention local HOGENOM paths unless explicitly presented
  as maintainer benchmark provenance.

## Suggested Milestones

### Milestone 1: Release Legality And Installability

Goal: a user can install the tool and verify the environment.

Deliverables:

- License and metadata fixed.
- Native artifact distribution decision implemented.
- `gpurec doctor` or equivalent readiness command.
- Installation matrix and container recipe.
- Clean wheel/source archive smoke outside checkout.

### Milestone 2: First Real User Workflow

Goal: a bioinformatician can run one documented dataset end to end.

Deliverables:

- Realistic public fixture or deterministic generator.
- Complete tutorial from input validation to RecPhyloXML sampling.
- Structured JSON status for validation and summaries.
- Known-limitations page.
- Snakemake or Nextflow example.

### Milestone 3: Correctness And Performance Confidence

Goal: maintainers can reject broken or slower releases.

Deliverables:

- GPU CI or scheduled GPU validation.
- Golden likelihood/gradient comparisons.
- Release benchmark thresholds.
- Stress tests for memory, resume, and failure recovery.
- Public validation report template.

### Milestone 4: Stable Product Surface

Goal: downstream users can depend on the CLI, Python API, and artifacts.

Deliverables:

- API/config/output compatibility policy.
- Changelog and citation metadata.
- Deprecation process.
- Public artifact schemas.
- Legacy script migration/deletion decisions.

## Practical Priority Order

If only one month of work is available, do this:

1. Resolve license and release metadata.
2. Choose and implement native artifact distribution.
3. Add a single environment readiness command.
4. Add one realistic end-to-end tutorial dataset or generator.
5. Add GPU validation for the retained production route.
6. Add a known-limitations page that prominently covers CUDA-only operation,
   `S > 256`, Newick subset limits, and external native artifacts.
7. Add structured JSON output for preflight and summary inspection.
8. Add a minimal Snakemake or Nextflow example.

If two more months are available, add:

1. Public benchmark gates.
2. Better input conversion and validation helpers.
3. Changelog, citation metadata, and release checklist.
4. Small-species fallback or a deliberately supported rejection workflow.
5. Cleanup or archival of legacy HOGENOM scripts and local data assumptions.

## Release Gate Checklist

A release should not be recommended to external bioinformatics users until all
of these are true:

- License and citation metadata are present.
- Installation docs cover native preprocessing and backtracking artifacts.
- A fresh install can run a readiness command successfully.
- A realistic dataset can be validated, optimized, inspected, resumed, and
  sampled using documented commands.
- GPU validation has passed for the production route.
- Output artifacts have stable schemas or documented compatibility rules.
- Known limitations are prominent and accurate.
- CLI errors are actionable for common input, native binary, CUDA, and
  convergence failures.
- Release artifacts are built from a clean checkout and smoke-tested outside the
  repository.
- The release notes state the exact tested platform matrix and benchmark
  evidence.

## References In This Repository

- Main user entry point: `../README.md`
- Documentation map: `README.md`
- Release checklist: `release-readiness.md`
- Input guide: `input-preparation.md`
- Run config reference: `run-config-reference.md`
- Output artifact contract: `output-artifacts.md`
- Troubleshooting guide: `troubleshooting.md`
- Tests overview: `../tests/README.md`
- Source examples: `../examples/README.md`
- Config ownership: `../configs/README.md`
- Script ownership: `../scripts/README.md`
- Profiling ownership: `../profiling/README.md`
- Repo-wide audit: `repo-wide-audit-2026-05-21.md`
