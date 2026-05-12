# gpurec lean fast path

This branch keeps the measured high-performance uniform-transfer path and
removes the research/proposal code that was not part of that path.

Retained runtime surface:

- `GeneReconModel` for `mode="global"`, `mode="specieswise"`, and
  `mode="genewise"`.
- `UniformChunkedReconModel` for large global/uniform datasets.
- Standard PyTorch optimizers over `model.theta`, including `torch.optim.Adam`.
- `gpurec.optimization.BatchedLBFGS` for row-wise genewise polishing.
- The optimized uniform CUDA forward/backward kernels used by the 1000-tree
  benchmark.

Removed from this branch:

- Legacy full-matrix Pi fixed-point baselines.
- CLI and AleRax sampling glue.
- Old optimizer facades (`optimize_theta_wave`, `optimize_theta_genewise`,
  global L-BFGS wrappers).
- Failed/prototype forward kernel variants and proposal benchmark scripts.
- Historical docs, generated result tables, and vendored `rustree` sources.

## Installation

```bash
pip install -e ".[triton,dev]"
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

## Performance Check

The retained full-dataset harness is:

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
