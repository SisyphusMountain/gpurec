# gpurec versus AleRax genewise performance

This benchmark supports the performance section of the differentiable DTL
reconciliation paper. It compares genewise DTL-rate fitting on the identical
HOGENOM family set:

| Tool | Hardware | Families | Fit time |
|---|---|---:|---:|
| gpurec | one RTX 4090 | 10,869 | 758 s |
| AleRax 1.4.0 | 24 CPU cores | 10,869 | 17,682 s |

The measured ratio is approximately 23.3x. The complete compact values used by
the plot are in `results/headline.csv`.

## Layout

- `scripts/`: dataset builders and gpurec/AleRax benchmark runners;
- `results/`: timing summaries and likelihood cross-check evidence;
- `plots/make_fig_speed.py`: regenerates the manuscript speed figure from the
  versioned CSV.

The historical optimization investigation is archived at
`../../archive/internal-audits/hogenom-performance-optimization-plan.md`.
Complete `kernel-bench` Git history, including the original logs and worktree,
is retained in the ignored local archive store.

## Data and environment

By default, scripts use the repository root and `data/external/`. Override the
following when necessary:

- `GPUREC_ROOT`: gpurec source checkout;
- `GPUREC_DATA_ROOT`: local dataset store;
- `PREPROCESS_SO`: built Rust preprocessing extension;
- `HOGENOM_ROOT` or `ARCHAEA_ROOT`: dataset-specific overrides;
- `PYTHON`: Python executable.

Build the native preprocessor before running gpurec:

```sh
cargo build --release --manifest-path crates/gpurec-preprocess/Cargo.toml
```

## Reproduce

From the repository root:

```sh
benchmarks/hogenom-cpu-vs-gpu/scripts/reproduce.sh
```

The GPU portion takes minutes. The AleRax CPU baseline takes hours and is
launched separately as described by the script.

Regenerate the publication plot with:

```sh
python benchmarks/hogenom-cpu-vs-gpu/plots/make_fig_speed.py
```

The default output is
`papers/gpu-reconciliation/figures/fig_speed.pdf`.

## Evidence

- `results/hogenom_subsets_timing.summary.txt`: gpurec measurements at three
  HOGENOM scales.
- `results/alerax_chunked_timing.txt`: eight memory-safe AleRax chunks whose
  wall times sum to 17,682 seconds.
- `results/xcheck_headline_fullset.txt`: likelihood comparison on all 10,869
  matched families.
- `results/alerax_hogenom_combined_likelihoods.txt` and
  `results/per_fam_likelihoods.txt`: per-family likelihood inputs used by the
  cross-check.

These measurements were made on 2026-06-24. Machine wall-clock values should
not be interpreted as hardware-independent performance guarantees.
