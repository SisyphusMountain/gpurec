# gpurec consolidation record

Implementation date: 2026-09-01

This file records the consolidation of the gpurec code, documentation, papers,
benchmarks, datasets, and recovery material. It supersedes the pre-migration
inventory that described the same content while it was distributed among sibling
directories.

## Canonical layout

| Content | Canonical location | Git policy |
|---|---|---|
| gpurec package and coupled experiments | repository root, `gpurec/`, `experiments/` | tracked |
| User, design, and mathematical documentation | `docs/` | tracked sources; generated PDFs ignored |
| GPU reconciliation paper | `papers/gpu-reconciliation/` | source, figure inputs, generators, and compact evidence tracked |
| Ghost-lineage research | `papers/ghost-lineages/` | independent local Git repository; ignored by the gpurec parent repository |
| Stable HOGENOM CPU/GPU benchmark | `benchmarks/hogenom-cpu-vs-gpu/` | scripts and compact results tracked |
| Large datasets | `data/external/` | ignored payload, documented by `data/README.md` |
| Historical outputs and recovery material | `archive/storage/` | ignored payload, documented by `archive/README.md` |
| Selected historical documents and audits | `archive/` outside `storage/` | tracked when small and useful |

The separate `These` thesis repository remains in its original sibling location.
Its early ALE-GPU draft is also preserved here as a read-only historical snapshot.

## Maintained and retained documents

| Document | Location | Status |
|---|---|---|
| GPU reconciliation paper | `papers/gpu-reconciliation/main.tex` | canonical paper; HOGENOM and archaeal performance results, including RTX 4090 measurements |
| Kernel mathematical reference | `docs/mathematics/kernel-mathematics/main.tex` | maintained documentation; clean 11-page build verified |
| Active-subclade pruning note | `docs/mathematics/active-subclade-pruning/main.tex` | maintained documentation; clean 7-page build verified |
| Ghost-lineage rerouting report | `papers/ghost-lineages/gpurec-experiment/report/rerouting_report.tex` | separate repository; portable 13-page build verified with six local figures |
| Main-branch review | `archive/internal-reviews/2026-08-11-main-358a5b80/` | dated audit, not maintained product documentation |
| Early ALE-GPU draft | `archive/papers/ale-gpu-early-draft/main.tex` | historical snapshot from `These` commit `24958191da5e` |

The useful parts of the main-branch review were extracted into
`docs/glossary.md`, `docs/input-contract.md`, and
`docs/internal-reviews/main-358a5b80-backlog.md`. The review's claims about
missing tests, missing docs, and dead code that no longer apply to `dev` were not
promoted.

## GPU paper and benchmark provenance

The paper was migrated from the former `kernel-bench` repository. Its preserved
history is available in:

- branch `rescue/pre-doc-consolidation-2026-09-01` in the retired checkout;
- tag `archive/pre-doc-consolidation-2026-09-01` at its pre-rescue head;
- complete bundle
  `archive/storage/repository-bundles/kernel-bench-all-refs-2026-09-01.bundle`.

The canonical paper figures live in `papers/gpu-reconciliation/figures/`. Their
generators and evidence are documented in that paper's README. The HOGENOM speed
plot is generated from
`benchmarks/hogenom-cpu-vs-gpu/results/headline.csv`. Validation found that the
old included speed plot used stale timing values; the canonical regenerated plot
now matches the manuscript's 119 s, 219 s, and 758 s GPU timings and its CPU
comparison.

The retained publication-input figures are:

| Figure | Generator or compact evidence |
|---|---|
| `fig_speed.pdf` | `benchmarks/hogenom-cpu-vs-gpu/plots/make_fig_speed.py` and `results/headline.csv` |
| `fig_cv.pdf` | `experiments/sanderson_cv/plot_fig_cv.py` and `reproduction/cv/*.pt` |
| `fig_profile.pdf` | `papers/gpu-reconciliation/scripts/replot_profile_sym.py` and `reproduction/profile/fig_profile.pt` |
| `fig_s53_spectrum.pdf` | `experiments/sanderson_cv/fisher_information_s53.py` |
| `fig_s53_dl_confounding.pdf` | `experiments/sanderson_cv/fisher_information_s53.py` |
| `fig_s53_se.pdf` | `experiments/sanderson_cv/fisher_information_s53.py` |

## Ghost-lineage repository

The former non-versioned `ghost-induced-transfers` directory is now the independent
repository mounted at `papers/ghost-lineages/`. Its initial consolidated commit is
`15a1fca`. A complete recovery bundle is stored at
`archive/storage/repository-bundles/ghost-lineages-all-refs-2026-09-01.bundle`.
The historical `SisyphusMountain/ghost_experiments` repository is configured as
`reference-origin`, not as the publication target for the combined project.

The report's absolute graphics path was replaced by `report/figures/`; all six
inputs are versioned in the independent repository. Core experiment entry points
use `workspace_paths.py`, with `GPUREC_ROOT`, `GPUREC_DATA_ROOT`,
`GHOST_DATA_ROOT`, `GHOST_RESULTS_ROOT`, `GHOST_SCRATCH_ROOT`, and
`GHOST_CYANO_ROOT` as portable overrides. Historical one-off recipes remain for
provenance and may still require their original external programs and datasets.

## Dataset and archive moves

| Former sibling | New location or disposition |
|---|---|
| `gpurec-data/` | `data/external/` |
| `gpurec-archive/` | `archive/storage/` |
| `gergely_version/` | recoverable copy under `archive/storage/retired-workspace/` |
| `comparison_report/` | recoverable copy under `archive/storage/retired-workspace/` |
| rescue/worktree directories | `archive/storage/recovery/` |
| centered-kernel worktree | source committed to branch `rescue/centered-kernels-2026-09-01`; ignored output archived |

The ghost-lineage input directory is at `data/external/ghost-lineages/`. Historical
generated results and third-party tool builds remain under
`archive/storage/consolidate-release-untracked/experiments/ghost_lineages/`.

## Build and tracking policy

- Track LaTeX sources, human-authored Markdown, publication-input figures,
  figure generators, and compact result summaries.
- Ignore LaTeX intermediates and compiled maintained-document PDFs.
- Keep large datasets, checkpoints, run directories, caches, third-party builds,
  repository bundles, and retired workspaces under the ignored data/archive stores.
- Use repository-relative paths or documented environment variables in maintained
  sources.
- Treat dated reviews and superseded drafts as archives, not parallel editable
  documentation.

## Verification completed

- GPU paper: clean LaTeX build, 20 pages, resolved references.
- Kernel mathematics: clean LaTeX build, 11 pages, resolved references.
- Active-subclade pruning: clean LaTeX build, 7 pages, resolved references.
- Ghost-lineage report: clean LaTeX build, 13 pages, resolved references and all
  six local figures.
- Migrated Python scripts: syntax compilation passed.
- Migrated shell scripts: `bash -n` passed.
- HOGENOM speed, cross-validation, and profile figure generators ran from their
  canonical inputs; the CV/profile extracted text and page geometry matched their
  prior publication versions.

## Remaining external step

The ghost-lineage project has a complete local Git history and bundle but no remote
for the combined repository. Creating or choosing that remote, pushing commit
`15a1fca`, and optionally replacing the ignored mount with a formal submodule is a
separate external publication decision.
