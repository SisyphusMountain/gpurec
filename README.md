# gpurec

`gpurec` provides GPU-accelerated PyTorch reconciliation models and workflow
tooling for AleRax-style family inputs, checkpointed optimization, and
stochastic RecPhyloXML sampling.

## Runtime Surface

- `GeneReconModel` for `mode="global"`, `mode="specieswise"`, and
  `mode="genewise"`.
- `UniformChunkedReconModel` for large global/uniform datasets, with public
  chunk metadata for inspecting resident chunk planning.
- `gpurec.workflow` production runners for AleRax-style family inputs,
  checkpointed optimization, convergence diagnostics, and stochastic
  backtracking.
- `gpurec` CLI entry point with `optimize`, `sample`, `run`, and
  `backtrack-check` commands.
- Standard PyTorch optimizers over `model.theta`, including `torch.optim.Adam`.
- `gpurec.optimization.BatchedLBFGS` for row-wise genewise polishing.
- The optimized uniform CUDA forward/backward kernels used by the 1000-tree
  benchmark.

Performance-pruning history and retained benchmark context live in
[`docs/lean-fast-path.md`](docs/lean-fast-path.md); the README focuses on the
supported runtime surface.

## Installation

Supported Python versions are Python 3.10-3.12, matching
`requires-python = ">=3.10,<3.13"` in the project metadata.

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

For direct `from_trees` inputs, gene-tree leaf labels are mapped to species by
the legacy prefix fallback: `Species_gene` maps to species `Species`, and a leaf
without `_` maps to the full leaf label.  Use AleRax family files with `mapping`
entries, `UniformChunkedReconModel(..., leaf_species_maps=...)`, or
`GeneDataset(..., leaf_species_maps=...)` when gene labels do not follow that
prefix convention.

The retained C++ preprocessing parser supports a deliberately small Newick
subset: nested trees with unquoted labels, optional internal labels, optional
numeric branch lengths, and ordinary whitespace.  Branch lengths are ignored.
When a branch length is present, the numeric text must immediately follow `:`.
The final semicolon is optional for a single tree or final gene-tree record.
Labels cannot rely on quotes, escaping, comments, NHX or BEAST-style metadata,
or embedded `:`, `,`, `(`, `)`, or `;` delimiters.  A species-tree file must
contain exactly one rooted binary tree.  Each gene-tree file may contain one or
more semicolon-delimited records; all records supplied for one family are
amalgamated into that family's CCP.  Gene multifurcations are right-binarized,
while unary gene nodes and non-binary species nodes are rejected.

`model.nll_per_family()` and `model.full_nll_per_family()` are genewise-only:
they return one independent NLL per family and are the public surface for
row-wise optimizers.  In `global` or `specieswise` mode, use
`model(reduce="per_family")` under `torch.no_grad()` only as a diagnostic
shared-theta breakdown; independent per-family gradients are not defined there.

For large global/uniform datasets, `UniformChunkedReconModel.loss_and_grad()`
returns `(loss, grad, stats)` for direct stochastic or sampled-chunk optimizers.
The stats dictionary includes selected chunk/family counts, timing fields,
`grad_norm`, reduction metadata, and E-adjoint solve telemetry:
`e_adjoint_method`, `e_adjoint_iterations`, `e_adjoint_rel_res`, and
`e_adjoint_success`.

With lazy preprocessing or resident-batch prefetching,
`model.configure_solver_iterations()` updates the model defaults and resident
batch static states that are already built.  It does not cancel or rewrite
pending background prefetch work.  Configure solver iteration controls before
scheduling lazy prefetch, or call `model.materialize_batches()` and configure
again when all resident batches should share the new controls.

For genewise row-wise polishing:

```python
from gpurec.optimization import BatchedLBFGS

opt = BatchedLBFGS([model.theta], lr=1.0)
```

## Python Workflow API

The CLI is the supported entry point for production runs, and the same workflow
objects are also available as top-level Python shortcuts:

```python
from gpurec import RunConfig, SamplingConfig, optimize, sample

run = RunConfig.from_json("run.json")
result = optimize(run)

sampling = SamplingConfig(checkpoint=result.sampling_checkpoint, samples=100)
sample_result = sample(sampling)
```

For direct imports from the workflow package, `gpurec.workflow` exports the same
`RunConfig`, `SamplingConfig`, `OptimizationRunner`, `SamplingRunner`,
`OptimizationResult`, `SamplingResult`, and `optimize`/`sample` functions.

Top-level backtracking helpers are also available for lower-level sampling and
validation workflows:

```python
from gpurec import (
    ensure_backtracking_available,
    export_backtracking_input,
    recphyloxml_event_counts,
    sample_recphyloxml,
    sample_recphyloxmls,
)
```

These helpers use the same Rust backtracking binary configuration documented
below for `gpurec sample` and `gpurec run`.

## Production AleRax-Style Workflow

The production workflow accepts an AleRax `[FAMILIES]` file and a species tree.
It defaults to genewise D/T/L parameters, writes resumable checkpoints, logs
optimization diagnostics, and can sample RecPhyloXML reconciliation scenarios.
The optimized likelihood path currently requires CUDA.
The `--config` option accepts a flat JSON `RunConfig`; Hydra-style YAML
configs should be converted to JSON or passed as explicit CLI flags.  Relative
paths in JSON configs are resolved from the config file's directory; relative
paths passed as explicit CLI flags are resolved from the current working
directory.
For a source checkout or source archive, a checked minimal JSON config and tiny
AleRax-style fixture live under `examples/`.  Inspect or adapt:

```bash
gpurec optimize --config examples/minimal-run-config.json
```

The checked config is a source-tree smoke for the retained optimized path and
sets `"device": "cuda"`.  The tiny tree files are portable, but the current
optimized likelihood implementation is not a CPU fallback.

Installed wheels do not install the `examples/` directory as runtime package
data; copy or adapt a flat JSON config alongside your own tree files instead:

```json
{
  "species_tree": "S.tree",
  "families_file": "families.txt",
  "out_dir": "output_gpurec",
  "mode": "genewise",
  "device": "cuda"
}
```

Workflow config aliases are shared between JSON configs and CLI flags.  `dtype`
accepts `float32`/`float64` plus aliases such as `fp32`, `single`, `fp64`,
`double`, `torch.float32`, and `torch.float64`.  `family_chunk_size` accepts a
non-negative integer, with `0`, `all`, `none`, or JSON `null` meaning one
resident batch.  `batch_packing` accepts `sequential`, `clade_first_fit`, and
`depth_first_fit`; aliases include `contiguous`/`input_order`,
`first_fit_decreasing`/`ffd`/`clade_ffd`, and
`depth_ffd`/`critical_path_first_fit`/`wave_first_fit`, with hyphenated forms
accepted by the CLI.  Non-sequential packing requires `clade_budget`.

The workflow CLI intentionally supports only float32 and float64.  The direct
`UniformChunkedReconModel` constructor also accepts `torch.bfloat16` as an
experimental CUDA-only path for memory-constrained forward/NLL probes, but
bf16 is not a supported workflow configuration dtype and is not supported by
the retained Pi backward/gradient path.  Do not use bf16 for release smokes,
optimizer checkpoints, or Hessian/second-order diagnostics.

Optimizer modes are selected with `optimizer` in JSON or `--optimizer` on the
CLI:

| Mode | Behavior | Notes |
| --- | --- | --- |
| `adam` | Default Adam optimizer for all configured steps. | Uses `lr` and ordinary PyTorch Adam state. |
| `adagrad` | Adagrad optimizer for all configured steps. | Uses `lr`; retained for long-running comparison runs. |
| `lbfgs` | PyTorch LBFGS for all configured steps. | Uses `lbfgs_lr`, `lbfgs_history_size`, `lbfgs_max_iter`, and `lbfgs_line_search`.  `lbfgs_line_search` is `none` or `strong_wolfe`; LBFGS runtime errors stop the run with a failed status. |
| `adam-lbfgs` | Adam warmup, then LBFGS polishing. | `adam_warmup_steps` controls the phase switch; incompatible resumed optimizer state is discarded when the checkpoint phase differs from the current phase. |

```bash
gpurec optimize \
  --species-tree S.tree \
  --families-file families.txt \
  --out-dir output_gpurec \
  --preprocess-cache output_gpurec/preprocess_cache \
  --mode genewise \
  --device cuda
```

`--preprocess-cache` stores reusable CPU preprocessing artifacts for unchanged
species and gene trees.  If cache loading fails with safe-loading or cache-shape
validation errors after a code upgrade, rerun with `--refresh-preprocess-cache`
to regenerate those entries from the original tree inputs.

Main outputs include:

- `checkpoints/latest.pt` and `checkpoints/best.pt`, metadata-bearing
  checkpoints for resume, sampling, and restore workflows
- `optimization_history.csv` and `history.jsonl`
- `rates_final.tsv`, `theta_final.pt`, `summary.json`
- `per_fam_likelihoods.tsv` for genewise runs

History rows include aggregate `solver/*` telemetry when the model reports
solver statistics.  E-adjoint nonconvergence is diagnostic-only: the retained
BiCGSTAB solve returns its best iterate with `success=False`, optimization
continues unless the objective or gradient becomes nonfinite, and history rows
surface `solver/e_adjoint_failed_batches` plus relative-residual and iteration
summaries for monitoring.

`theta_final.pt` is a raw tensor export for inspection or custom analysis.  It
does not carry run configuration, family ordering, or species ordering metadata;
use `checkpoints/best.pt` or `checkpoints/latest.pt` whenever a workflow needs
to restore parameters into a model or sample reconciliation scenarios.

Resume starts from the checkpoint `next_step`.  If `next_step` already equals
the configured `steps`, `gpurec optimize --resume-from ...` performs only the
final evaluation/artifact refresh, writes a fresh `latest.pt`, and returns the
same `not_converged`/`max_steps` status used by ordinary max-step exhaustion.
Increase `steps` beyond the checkpoint `next_step` to run additional optimizer
steps.

To sample scenarios from the best checkpoint:

```bash
gpurec sample \
  --checkpoint output_gpurec/checkpoints/best.pt \
  --samples 100 \
  --sample-out-dir output_gpurec
```

Sampling writes per-sample RecPhyloXML files and event-count files under
`output_gpurec/reconciliations/all/`.  Aggregate summaries live under
`output_gpurec/reconciliations/`, including `event_counts.tsv`,
`totalSpeciesEventCounts.txt`, and `totalTransfers.txt`.
`event_counts.tsv` is tab-separated.  `totalSpeciesEventCounts.txt` is the
AleRax-compatible comma-space text format with one species label followed by
event totals.  `totalTransfers.txt` uses whitespace-separated source species,
destination species, and average transfer count.  Aggregate values are averaged
over the requested sample count for each retained family, not over all families
in the original checkpoint.
The sampled RecPhyloXML subset expected by gpurec contains `recGeneTree` blocks
with `clade` nodes, `eventsRec` event containers, and the event tags
`speciation`, `duplication`, `branchingOut`, `transferBack`, `loss`, and
`leaf`.  gpurec-generated sample XML files are expected to contain one
`recGeneTree` per file; the shared event-count traversal can still read
multiple `recGeneTree` blocks in compatibility inputs.
Use `--family-start` and `--sample-max-families` to sample a family window,
`--seed` for reproducible stochastic backtracking, and `--max-events` to cap
pathological samples.
Successful sampling reruns replace prior gpurec-generated reconciliation
artifacts in the target output directory, including generated files outside a
requested window; use a separate `--sample-out-dir` to keep multiple windows.

To optimize and sample in one supported CLI workflow, run `gpurec run` with the
same optimization config plus sampling options:

```bash
gpurec run \
  --config run.json \
  --samples 100 \
  --sample-out-dir output_gpurec
```

Sampling flags such as `--samples`, `--family-start`, `--sample-max-families`,
`--max-events`, and `--backtrack-binary` are accepted on `gpurec run`.
`gpurec run` does not accept `--checkpoint`; it samples from the checkpoint
reported by the optimizer, falling back to `checkpoints/best.pt` or
`checkpoints/latest.pt` when needed, and exits without sampling if optimization
fails.  Use `gpurec sample --checkpoint ...` to sample an existing run.

### Sampling Binary Setup

`gpurec sample` and the sampling phase of `gpurec run` use the Rust
backtracking binary.  Wheels currently do not ship that binary or the Rust
crate sources, so installed environments should provide a compiled binary
through `GPUREC_BACKTRACK_BIN` or `--backtrack-binary`.  In a wheel-only
install, point gpurec at a prebuilt binary produced by your deployment or build
process:

```bash
export GPUREC_BACKTRACK_BIN="/opt/gpurec/bin/gpurec-backtrack"
gpurec backtrack-check
gpurec sample --checkpoint output_gpurec/checkpoints/best.pt --samples 100
```

For a source checkout or unpacked source archive, build that binary with Cargo:

```bash
cargo build --locked --release --manifest-path crates/gpurec-backtrack/Cargo.toml
export GPUREC_BACKTRACK_BIN="$PWD/crates/gpurec-backtrack/target/release/gpurec-backtrack"
gpurec backtrack-check
gpurec sample --checkpoint output_gpurec/checkpoints/best.pt --samples 100
```

The same `GPUREC_BACKTRACK_BIN` environment variable or `--backtrack-binary`
flag applies to `gpurec run` when it samples after optimization.
Use `gpurec backtrack-check` to validate the binary or source-tree Cargo
fallback by running its `--help` path without loading a checkpoint.

The automatic `cargo run` fallback works from a source checkout or unpacked
source archive.  It requires a Rust toolchain and fetches the pinned `rustree`
git dependency declared by `crates/gpurec-backtrack/Cargo.toml`; otherwise use a
prebuilt binary.

### Runtime Environment Flags

Most production settings should be expressed through JSON config files or CLI
flags.  The `GPUREC_*` environment variables are retained for binary discovery,
compatibility guards, memory-policy margins, and kernel diagnostics.
Boolean flags treat empty, `0`, `false`, `no`, and `off` as false.  CUDA
prototype mode flags also accept `auto`, with `1`, `true`, `yes`, `on`,
`force`, or `required` making a selected prototype path required instead of
best-effort.  These native CUDA prototype flags are experimental diagnostics:
`auto` and `enabled` fall back to the retained Triton paths when a selected
prototype is unavailable, while required modes re-raise failures only after the
prototype's dtype/device/layout eligibility checks select that path.
`GPUREC_CUDA_PIBAR_FROM_UD_STRICT=1` makes the Pibar prototype required after
selection even when `GPUREC_CUDA_PIBAR_FROM_UD=auto`.  Current fallback behavior
is intentionally documented before consolidation: self-loop prototype fallback
is silent and catches only optional import/runtime/validation failures, Pibar
prototype fallback is silent in `auto` but warns once in `enabled`, and the
self-loop loader preloads wheel NVRTC builtins while the Pibar loader currently
does not.
Both native CUDA prototypes compute dynamic shared-memory requirements from the
requested species count and launch tuning, set
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`, and pass that same byte count
to `cuLaunchKernel`.  They preflight the requested scratch size against the
device shared-memory limit before launch; later attribute or launch failures
follow the best-effort or required-mode fallback policy above.

| Variable | Scope |
| --- | --- |
| `GPUREC_BACKTRACK_BIN` | Path to the Rust backtracking binary used by `gpurec sample`, `gpurec run`, and `gpurec backtrack-check`. |
| `GPUREC_ALERAX_COMPAT` | Compatibility guard; differentiable model optimization supports only unset or `0`. |
| `GPUREC_MEMORY_POLICY_FRACTION`, `GPUREC_MEMORY_POLICY_RESERVE_GIB` | GPU memory-budget margins used by uniform chunk planning. |
| `GPUREC_FUSE_FINAL_PIBAR`, `GPUREC_SPECIALIZE_NONLEAF_LEAF_TERM` | Forward/backward fast-path selectors for retained uniform kernels. |
| `GPUREC_BACKWARD_NO_CPU_PRUNING`, `GPUREC_DTS_SKIP_INACTIVE_PIBAR_ZERO` | Backward pass pruning and inactive-output zero-fill controls. |
| `GPUREC_CUDA_SELF_LOOP_NOSPLIT`, `GPUREC_CUDA_SELF_LOOP_SPLIT`, `GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION` | Native CUDA self-loop prototype selectors and correction mode. |
| `GPUREC_CUDA_PIBAR_FROM_UD`, `GPUREC_CUDA_PIBAR_FROM_UD_STRICT` | Native CUDA Pibar-from-`u_d` prototype selector and strict-failure mode. |
| `GPUREC_WAVE_STEP_BLOCK_S`, `GPUREC_WAVE_STEP_NUM_WARPS` | Triton forward wave-step launch tuning. |
| `GPUREC_DTS_BLOCK_S`, `GPUREC_DTS_NUM_WARPS`, `GPUREC_DTS_GRAD_MT_TILE_SPLITS` | Triton cross-DTS backward launch tuning. |
| `GPUREC_DTS_PARENT_BLOCK_S`, `GPUREC_DTS_PARENT_NUM_WARPS`, `GPUREC_DTS_PARENT_TILE_SPLITS` | Triton parent-reduced DTS forward launch tuning. |
| `GPUREC_PIBAR_UD_BLOCK_S`, `GPUREC_PIBAR_UD_NUM_WARPS` | Triton Pibar-from-`u_d` launch tuning. |
| `GPUREC_SELF_LOOP_2D_BLOCK_W`, `GPUREC_SELF_LOOP_2D_BLOCK_NODES`, `GPUREC_SELF_LOOP_2D_NUM_WARPS`, `GPUREC_SELF_LOOP_2D_JT_NUM_WARPS`, `GPUREC_SELF_LOOP_2D_SKIP_INACTIVE_SCRATCH_ZERO` | Triton 2D backward self-loop launch tuning and scratch-zero behavior. |
| `GPUREC_CUDA_SELF_LOOP_BLOCK`, `GPUREC_CUDA_SELF_LOOP_CHILD_EDGE_WEIGHT` | Native CUDA self-loop launch tuning. |
| `GPUREC_CUDA_PIBAR_FROM_UD_BLOCK`, `GPUREC_CUDA_PIBAR_FROM_UD_PAD_SHARED` | Native CUDA Pibar-from-`u_d` launch tuning. |

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
| Sampling binary preflight | `gpurec backtrack-check` | Validates `GPUREC_BACKTRACK_BIN`, `--backtrack-binary`, or the source-tree Cargo fallback without loading a checkpoint. |
| Legacy HOGENOM W&B wrapper | `python scripts/optimize_hogenom_ccp_wandb.py` | Checkout-local compatibility wrapper with argparse flags, checkpoints, plots, and optional W&B logging. |
| Hydra HOGENOM run | `python scripts/optimize_hogenom_ccp_hydra.py` | Uses `configs/hogenom_ccp_wandb.yaml`, a checkout-local full experiment config, and Hydra override syntax. |
| Checkpoint rate export | `python scripts/export_hogenom_rates_from_checkpoint.py` | Exports D/T/L rates from a HOGENOM optimization checkpoint. |
| One-pass profiling | `python scripts/profile_hogenom_ccp_pass.py` | Profiles full, streamed, active-batch, or largest-batch forward/backward passes on the local HOGENOM layout. |
| Historical notebooks | `notebooks/` | Checkout-local HOGENOM analyses; see `notebooks/README.md` before using or migrating them. |

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
