# gpurec

`gpurec` provides GPU-accelerated PyTorch reconciliation models and workflow
tooling for AleRax-style family inputs, checkpointed optimization, and
stochastic RecPhyloXML sampling.

## Runtime Surface

- `GeneReconModel` for `mode="global"`, `mode="specieswise"`, and
  `mode="genewise"`.
- `UniformChunkedReconModel` for large global/uniform datasets.
- `gpurec.workflow` production runners for AleRax-style family inputs,
  checkpointed optimization, convergence diagnostics, and stochastic
  backtracking.
- `gpurec` CLI entry point with `optimize`, `sample`, and `run` commands.
- Standard PyTorch optimizers over `model.theta`, including `torch.optim.Adam`.
- `gpurec.optimization.BatchedLBFGS` for row-wise genewise polishing.
- The optimized uniform CUDA forward/backward kernels used by the 1000-tree
  benchmark.

Performance-pruning history and retained benchmark context live in
[`docs/lean-fast-path.md`](docs/lean-fast-path.md); the README focuses on the
supported runtime surface.

## Installation

For a source checkout:

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

The CUDA kernels import Triton directly, so Triton is a core dependency rather
than an optional extra.  Install a PyTorch build that matches the local CUDA
runtime before installing `gpurec`.  The preprocessing extension is compiled at
runtime through PyTorch's C++ extension loader; source installs therefore need
a working C++ compiler, OpenMP support, and the normal PyTorch extension build
tooling available in the Python environment.

For the checkout-local HOGENOM experiment scripts:

```bash
pip install -e ".[hogenom,dev]"
```

## Basic Optimization

```python
import torch
from gpurec import GeneReconModel

model = GeneReconModel.from_trees(
    species_tree="sp.nwk",
    gene_trees=["g_0.nwk", "g_1.nwk"],
    mode="genewise",          # also: "global", "specieswise"
    device="cuda",
    dtype=torch.float32,
    fixed_iters_Pi=6,
    neumann_terms=6,
)

opt = torch.optim.Adam([model.theta], lr=0.03)
for _ in range(20):
    opt.zero_grad(set_to_none=True)
    loss = model()
    loss.backward()
    opt.step()
    model.clamp_theta_(min_rate=1e-10, max_rate=2.0)
```

For genewise row-wise polishing:

```python
from gpurec.optimization import BatchedLBFGS

opt = BatchedLBFGS([model.theta], lr=1.0)
```

## Production AleRax-Style Workflow

The production workflow accepts an AleRax `[FAMILIES]` file and a species tree.
It defaults to genewise D/T/L parameters, writes resumable checkpoints, logs
optimization diagnostics, and can sample RecPhyloXML reconciliation scenarios.
The optimized likelihood path currently requires CUDA.
The `--config` option accepts a flat JSON `RunConfig`; Hydra-style YAML
configs should be converted to JSON or passed as explicit CLI flags.

```bash
gpurec optimize \
  --species-tree S.tree \
  --families-file families.txt \
  --out-dir output_gpurec \
  --mode genewise \
  --device cuda
```

Main outputs include:

- `checkpoints/latest.pt` and `checkpoints/best.pt`
- `optimization_history.csv` and `history.jsonl`
- `rates_final.tsv`, `theta_final.pt`, `summary.json`
- `per_fam_likelihoods.tsv` for genewise runs

To sample scenarios from the best checkpoint:

```bash
gpurec sample \
  --checkpoint output_gpurec/checkpoints/best.pt \
  --samples 100
```

Sampling writes RecPhyloXML files and AleRax-style summaries under
`output_gpurec/reconciliations/`, including per-sample event counts,
`totalSpeciesEventCounts.txt`, and `totalTransfers.txt`.
Sampling uses the Rust backtracking binary.  Installed environments should
provide a compiled binary through `GPUREC_BACKTRACK_BIN` or
`--backtrack-binary`.  The source-checkout `cargo run` fallback also requires a
Rust toolchain and the local `rustree/` checkout expected by
`crates/gpurec-backtrack/Cargo.toml`; otherwise use a prebuilt binary.

## HOGENOM Workflows

The installed `gpurec` CLI is the supported general workflow.  The HOGENOM
scripts under `scripts/` are legacy checkout-local experiment launchers and
diagnostics for a local, untracked HOGENOM benchmark layout.  The HOGENOM data
under `tests/data/HOGENOM/...` is intentionally not distributed with the package.
The optimization launchers default to those local paths; pass explicit
`--species-tree`, `--families-file`, `--preprocess-cache`, and `--out-dir`
values to those launchers for other datasets.  The one-pass profiler is tied to
that local HOGENOM layout and exposes profiling and batch-control flags instead
of dataset path overrides.

| Task | Command | Notes |
| --- | --- | --- |
| General installed workflow | `gpurec optimize`, `gpurec sample`, `gpurec run` | Uses flat JSON for `--config`; Hydra YAML is not accepted by the main CLI. |
| Curated HOGENOM W&B run | `python scripts/optimize_hogenom_ccp_wandb.py` | Uses argparse flags, checkpoints, plots, and optional W&B logging. |
| Hydra HOGENOM run | `python scripts/optimize_hogenom_ccp_hydra.py` | Uses `configs/hogenom_ccp_wandb.yaml` and Hydra override syntax. |
| Checkpoint rate export | `python scripts/export_hogenom_rates_from_checkpoint.py` | Exports D/T/L rates from a HOGENOM optimization checkpoint. |
| One-pass profiling | `python scripts/profile_hogenom_ccp_pass.py` | Profiles full, streamed, active-batch, or largest-batch forward/backward passes on the local HOGENOM layout. |

Optional script dependencies are intentionally separate from the core package
and are grouped under the `hogenom` extra.  The R plotting helper also needs
its CRAN/Bioconductor plotting packages.

Minimal HOGENOM smoke when CUDA memory allows:

```bash
python scripts/optimize_hogenom_ccp_wandb.py \
  --max-families 1 \
  --steps 1 \
  --wandb-mode disabled \
  --no-timestamped-out-dir
```

## Source-Checkout Performance Check

The full-dataset benchmark harness lives under `profiling/` and is intended
for source checkouts, not installed wheels:

```bash
python profiling/bench_uniform_forward_backward_pipeline.py \
  --dataset /path/to/test_trees_1000 \
  --fams 1000 \
  --family-chunk-size auto \
  --max-wave-size auto \
  --fixed-iters 6 \
  --neumann-terms 3 \
  --warmups 1 \
  --reps 3 \
  --strict-optimized-kernels
```

Expected on the measured RTX 4090 setup is roughly:

```text
forward_median_ms  ~2445
backward_median_ms ~3532
total_median_ms    ~5979
generic_self_loop_calls 0
strict_optimized_verdict pass
```

## Documentation

See `docs/README.md` for the current documentation map.  It separates current
operating notes from historical performance and research logs.
