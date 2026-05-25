# test_trees_1000 Resident Likelihood Timing

Date: 2026-05-25

This records the current fast path for full-dataset resident likelihood
evaluation on `tests/data/test_trees_1000` in specieswise mode.  The generated
tree dataset has a different shape from HOGENOM: `1999` species, `1000`
families, about `6.4M` clades, and `21` resident batches at a `315000` clade
budget.

For end-to-end timing on this dataset, the fastest route found so far is the
retained Rust preprocessing path with `clade_first_fit`.  It reuses the native
chunked wave-layout builder directly, avoiding the older Python scheduler
summary and JSON/tensor roundtrips during resident batch construction.  The
resident path also keeps only compact family summaries during construction; full
family tensors are materialized later only if diagnostic family input is
requested.  For this generated tree shape, the retained `clade_first_fit` path
uses the scheduler's forward nonleaf policy rather than the exhaustive
fewest-wave policy; it keeps the measured max wave count here while avoiding
extra native scheduling passes during construction.  CUDA warmup is started
while Rust preprocessing runs and exercises cuBLAS plus common elementwise math,
so more of the first E-solve setup is hidden under CPU/native work.  During
model construction, retained Rust layout generation is also started while CUDA
species-helper tensors are built.

Cold end-to-end command:

```bash
env PYTHONDONTWRITEBYTECODE=1 GPUREC_MEMORY_POLICY_RESERVE_GIB=0 \
  python profiling/bench_resident_likelihood.py \
  --dataset tests/data/test_trees_1000 \
  --mode specieswise \
  --fixed-iters 4 \
  --measure loss-only \
  --warmups 0 \
  --reps 1 \
  --family-chunk-size 500 \
  --clade-budget 315000 \
  --batch-packing clade_first_fit \
  --max-wave-size 8192 \
  --materialize-batches none \
  --prefetch-batches all
```

Cold result:

| Stage | Time |
|---|---:|
| model init / first resident batch | `1.0128701945068315s` |
| first fixed4 likelihood pass plus lazy remaining batches | `2.147080860013375s` |
| total to first fixed4 likelihood | `3.1598341505450662s` |

Cold first-pass fidelity samples with the same construction path:

| Pi/E/Neumann budget | total to first likelihood | loss bits | delta vs fixed128 |
|---:|---:|---:|---:|
| 4 | `3.1771413069800474s` | `2156427.0` | `670.25` |
| 6 | `3.6668981159455143s` | `2157095.0` | `2.25` |
| 8 | `4.169835773995146s` | `2157097.25` | `0.0` |

Before warming cuBLAS and the first elementwise CUDA math during preprocessing,
the same retained-layout overlap path took about `3.176s` to the first fixed4
likelihood.  Before overlapping retained Rust layout generation with CUDA
species-helper setup, the CUDA-warmed retained clade-first path took about
`3.19s`.  Before CUDA warmup overlap, the forward-scheduled retained clade-first
path took about `3.24s`.  Before the forward scheduler policy, the compact
retained-layout clade-first lazy path took about `3.46s`.  Before compact family
summaries, it took about `3.86s`.  Before retaining the Rust chunked layouts,
the same clade-first lazy path took `13.062108609999996s`, and the HOGENOM-style
`depth_first_fit` path took `17.548588357982226s` in a manual timing split.  The
construction path is therefore the main end-to-end win for this generated
dataset.

Steady-state command:

```bash
env PYTHONDONTWRITEBYTECODE=1 GPUREC_MEMORY_POLICY_RESERVE_GIB=0 \
  python profiling/bench_resident_likelihood.py \
  --dataset tests/data/test_trees_1000 \
  --mode specieswise \
  --fixed-iters 4,6,8,128 \
  --measure loss-only \
  --warmups 1 \
  --reps 3 \
  --family-chunk-size 500 \
  --clade-budget 315000 \
  --batch-packing clade_first_fit \
  --max-wave-size 8192 \
  --materialize-batches all
```

| Pi/E/Neumann budget | loss-only median | loss bits | delta vs fixed128 |
|---:|---:|---:|---:|
| 4 | `1.6164941460010596s` | `2156427.0` | `670.25` |
| 6 | `2.1500701049808413s` | `2157095.0` | `2.25` |
| 8 | `2.6858214950188994s` | `2157097.25` | `0.0` |
| 128 | `35.36583194194827s` | `2157097.25` | `0.0` |

With `--materialize-batches all`, the clade-first resident build split was
`1.0175689089810476s` for model init plus `0.0647850509849377s` for full
materialization in the fixed4 run.

Gradient timing for the same resident layout:

```bash
env PYTHONDONTWRITEBYTECODE=1 GPUREC_MEMORY_POLICY_RESERVE_GIB=0 \
  python profiling/bench_resident_likelihood.py \
  --dataset tests/data/test_trees_1000 \
  --mode specieswise \
  --fixed-iters 4,6,8 \
  --measure loss-grad \
  --warmups 1 \
  --reps 3 \
  --family-chunk-size 300 \
  --clade-budget 315000 \
  --batch-packing clade_first_fit \
  --max-wave-size 8192 \
  --materialize-batches all
```

| Pi/E/Neumann budget | loss+backward median | loss bits | grad inf |
|---:|---:|---:|---:|
| 4 | `6.295372458000202s` | `2156427.0` | `311.9596252441406` |
| 6 | `7.550051988975611s` | `2157095.0` | `312.9272766113281` |
| 8 | `8.658673683996312s` | `2157097.25` | `312.9301452636719` |

The first clade-first loss+backward warmup in this process took
`43.41080819600029s` because it compiled additional backward kernel
specializations.  The table above is steady-state after that compilation.

Interpretation:

- Starting at `4` Pi iterations is a good warm phase.  It is about `40%`
  cheaper than `8` for likelihood-only and about `27%` cheaper than `8` for
  loss+backward.
- Do not treat `4` as final fidelity on this dataset.  It is still `670` bits
  away from the fixed128 reference, in the optimistic direction.  Promote to at
  least `6` once a cheap phase stops making progress; `6` is within `2` bits of
  fixed128 here.
- `8`, `32`, and `128` agree at the printed precision, so higher fixed budgets
  are useful mainly as periodic validation points, not every-step work.

Rejected follow-ups:

- Increasing lazy prefetch workers from one to two made the fixed4 loss pass a
  few milliseconds faster in isolation, but the four-process cold total stayed
  around `3.178s`; three and four workers were slower.
- Putting the small tail batch first made the cold path worse (`3.406573s` in
  one sample) and increased reserved memory to about `21.4 GiB`.  The allocator
  behaves better when the first resident Pi batch is full sized.
- Pre-reserving `0.5` to `4.0 GiB` in the CUDA warmup thread did not beat the
  current path; the best samples were around `3.163s` and used more reserved
  memory.
- `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` reduced reserved memory
  dramatically, but the sampled total was about `3.196s`, slower than the
  default caching allocator for this benchmark.
- Decoupling E below the Pi budget was not a valid shortcut.  With `Pi=4`,
  `E=1/2/3` gave losses `1896378.75`, `2092414.125`, and `2148211.25` bits.
  With `Pi=6`, `E=4` still lost about `672` bits relative to `E=6`.
- Adaptive fixed-point checks were slower than fixed budgets here.  Accurate
  `max=8`, `check_interval=4` took about `3.903s`, and lower-overhead settings
  either hit the maximum or added check overhead.

Differences from HOGENOM:

- HOGENOM's successful counts-free specieswise route used a `8 -> 16 -> 32`
  optimizer budget ladder with a `128` validation check.  On `test_trees_1000`,
  the first useful fidelity point is lower: `4` is cheap enough to use as the
  startup phase, then `6` is already nearly at the high-budget likelihood.
- HOGENOM worked best with `depth_first_fit`.  On `test_trees_1000`, depth-first
  originally gave similar steady likelihood timing but paid an extra Python
  construction prepass.  Retaining the native Rust layouts removes that Python
  prepass; with the current timings, `clade_first_fit` remains the best measured
  default for the generated tree benchmark.
- HOGENOM's depth-first path should keep the exhaustive scheduler policy because
  wave compaction mattered there.  On `test_trees_1000`, the retained
  `clade_first_fit` chunks had the same measured max wave count with the
  forward scheduler policy, so skipping the extra candidate schedulers saves
  about `0.2s` of cold construction.
- HOGENOM needed full family tensors during the accepted route.  On
  `test_trees_1000`, the resident Rust layout already contains the tensors used
  for likelihood, so compact family summaries remove about `0.3s` from cold
  construction without changing likelihood values.
- HOGENOM's older end-to-end route paid CUDA setup before the heavy CPU/native
  work.  On this generated benchmark, starting CUDA warmup before retained Rust
  preprocessing hides most of that setup in the construction phase.  Warming
  cuBLAS and elementwise math is useful here because the first E solve is part
  of the measured cold likelihood pass.
- The generated benchmark also has independent CUDA species-helper setup and
  retained Rust layout generation during resident model construction; overlapping
  those saves a small additional amount.  This is less relevant to HOGENOM's
  accepted route because its depth-first layout planning was the dominant setup
  decision.
- HOGENOM resident batching had fewer batches under the accepted policy.  This
  dataset splits into `21` batches, so reusing a single global/specieswise
  resident E solve across no-grad batches removes repeated E work and is worth
  about two percent on likelihood-only timing.
- Larger clade budgets and larger wave caps hurt the first likelihood here.
  `250000`, `280000`, `300000`, `310000`, `312000`, `314000`, `316000`,
  `318000`, `320000`, `330000`, `350000`, `400000`, `500000`, and non-`8192`
  max-wave samples all hit much slower first-pass timings.  The
  `clade_first_fit`, `315000` clade-budget, `8192` max-wave policy keeps peak
  allocated memory near `5.15 GiB` for likelihood-only while avoiding those
  shape cliffs.
