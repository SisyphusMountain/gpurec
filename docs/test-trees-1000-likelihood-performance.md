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
extra native scheduling passes during construction.  The first Pi self-loop
kernel now synthesizes the all-`-inf`/leaf initial state directly, and non-leaf
waves without a leaf term copy the already-reduced DTS contribution instead of
launching the full first wave-step.  The second local iteration reads that DTS
tensor directly, avoiding a device-to-device copy into the global ping-pong
buffer.  CUDA warmup is started while Rust preprocessing runs and exercises
cuBLAS plus common elementwise math, so more of the first E-solve setup is
hidden under CPU/native work.  During model construction, retained Rust layout
generation is also overlapped with CUDA species-helper setup and a tiny resident
uniform E/Pi/DTS kernel warmup, hiding a large part of first-use Triton/CUDA
module loading under native layout construction.  Split-wave DTS output now
skips the full `-inf` initialization when the scheduler metadata proves every
row in the wave has a split contribution; on this generated layout all `973`
split waves are fully covered.  The first leaf-state Pi iteration now uses a
specialized kernel with species subtree intervals instead of paying a uniform
Pibar parent-chain walk from a one-hot leaf state, and the DTS species tile cap
is `512` for this large-`S` shape.  The no-callback Pi path also avoids the
per-wave progress shim calls used only by diagnostic tracing.  On the local
32-core host, the cold
benchmark also pins native preprocessing and retained layout generation to `16`
CPU threads; that is faster for this generated dataset than the default global
Rayon pool.

Cold end-to-end command:

```bash
env PYTHONDONTWRITEBYTECODE=1 GPUREC_MEMORY_POLICY_RESERVE_GIB=0 \
  python profiling/bench_resident_likelihood.py \
  --dataset tests/data/test_trees_1000 \
  --mode specieswise \
  --fixed-iters 4 \
  --preprocess-cpu-cores 16 \
  --measure loss-only \
  --warmups 0 \
  --reps 1 \
  --family-chunk-size 500 \
  --clade-budget 315000 \
  --batch-packing clade_first_fit \
  --max-wave-size 8192 \
  --max-root-wave-size none \
  --max-dts-partial-rows none \
  --materialize-batches none \
  --prefetch-batches all
```

Cold result:

| Stage | Time |
|---|---:|
| model init / first resident batch | `0.9730085899936967s` |
| first fixed4 likelihood pass plus lazy remaining batches | `1.577356451016385s` |
| total to first fixed4 likelihood | `2.5503650410100818s` |

Cold first-pass fidelity samples with the same construction path:

| Pi/E/Neumann budget | total to first likelihood | loss bits | delta vs fixed128 |
|---:|---:|---:|---:|
| 4 | `2.5503650410100818s` | `2156427.0` | `670.25` |
| 6 | `3.051072600996122s` | `2157095.0` | `2.25` |
| 8 | `3.5452640849980526s` | `2157097.25` | `0.0` |

Before pinning native preprocessing and retained layout generation to `16` CPU
threads on this host, the first-leaf-specialized route took about `2.678s` to
the first fixed4 likelihood.  Before specializing the first leaf-state Pi
iteration and raising the DTS tile cap for this shape, the DTS-fill-skip path
took about `2.702s` to the first fixed4 likelihood.  Before skipping full DTS
output initialization for fully covered split waves, the resident-kernel-warmed
path took about `2.745s` to the first fixed4 likelihood.  Before overlapping
the resident uniform kernel warmup with retained Rust layout generation, the
local-DTS input path took about `2.925s` to the first fixed4 likelihood.  Before
reading the first non-leaf DTS tensor directly on iteration 1, the copy-based
initial non-leaf fast path took about `2.994s` to the first fixed4 likelihood.
Before copying the first non-leaf DTS state directly, deriving the initial Pi
state inside the wave kernel took about `3.045s`.  Before deriving that initial
state inside the wave kernel, the same retained layout and CUDA math warmup path
took about `3.160s`.  Before warming cuBLAS and the first elementwise CUDA math
during preprocessing, the retained-layout overlap path took about `3.176s`.
Before overlapping retained Rust layout generation with CUDA species-helper
setup, the CUDA-warmed retained clade-first path took about `3.19s`.  Before
CUDA warmup overlap, the forward-scheduled retained clade-first path took about
`3.24s`.  Before the forward scheduler policy, the compact retained-layout
clade-first lazy path took about `3.46s`.  Before compact family summaries, it
took about `3.86s`.
Before retaining the Rust
chunked layouts, the same clade-first lazy path took `13.062108609999996s`, and
the HOGENOM-style `depth_first_fit` path took `17.548588357982226s` in a manual
timing split.  The construction path and per-batch Pi initialization are
therefore the main end-to-end wins for this generated dataset.

Skipping no-op progress callback dispatch and precomputing the root-only waves
used for final-Pibar skipping are small Python-side cleanups rather than large
kernel wins.  Fixed4 steady-state seven-repetition samples measured
`1.282079865981359s` and `1.2819753359653987s` medians with unchanged loss.
Cold first-likelihood samples remained noisy, so the cold route is still
reported with the established `2.55s` total sample above.

Steady-state command:

```bash
env PYTHONDONTWRITEBYTECODE=1 GPUREC_MEMORY_POLICY_RESERVE_GIB=0 \
  python profiling/bench_resident_likelihood.py \
  --dataset tests/data/test_trees_1000 \
  --mode specieswise \
  --fixed-iters 4,6,8,128 \
  --preprocess-cpu-cores 16 \
  --measure loss-only \
  --warmups 1 \
  --reps 3 \
  --family-chunk-size 500 \
  --clade-budget 315000 \
  --batch-packing clade_first_fit \
  --max-wave-size 8192 \
  --max-root-wave-size none \
  --max-dts-partial-rows none \
  --materialize-batches all \
  --prefetch-batches all
```

| Pi/E/Neumann budget | loss-only time | loss bits | delta vs fixed128 |
|---:|---:|---:|---:|
| 4 | `1.2834076849976555s` | `2156427.0` | `670.25` |
| 6 | `1.816627103020437s` | `2157095.0` | `2.25` |
| 8 | `2.3491738659795374s` | `2157097.25` | `0.0` |
| 128 | `35.236629224033095s` | `2157097.25` | `0.0` |

With `--materialize-batches all`, the clade-first resident build split was
`0.9815053479978815s` for model init plus `0.06590698298532516s` for full
materialization in the fixed4 steady-state run.  The `4/6/8` rows above are
three-repetition medians after one warmup; the `128` row is a single validation
sample.

Current fixed4 bottleneck profile on the same resident layout:

| Component | Time |
|---|---:|
| model init | `0.9855209130328149s` |
| materialize all resident batches | `0.06756154203321785s` |
| shared E solve | `0.003427101008128375s` |
| all Pi batches | `1.574903720000293s` |
| all root reductions | `0.012738829071167856s` |

The Pi pass remains the dominant cost.  Inside the instrumented fixed4 pass,
the largest kernel buckets were DTS phase 2 (`0.5598659516554342s`), final
Pibar/storage waves (`0.44713026658073274s`), local-DTS iteration-2 waves
(`0.19817568136751587s`), regular waves with DTS
(`0.19584079997241566s`), regular waves without DTS
(`0.12040464065969005s`), and DTS phase 3 (`0.023367967963218692s`).
All `973` split waves were fully covered; `922` were pure single-child split
waves, `30` were mixed, and `21` were pure multi-child split waves.

Gradient timing for the same resident layout:

```bash
env PYTHONDONTWRITEBYTECODE=1 GPUREC_MEMORY_POLICY_RESERVE_GIB=0 \
  python profiling/bench_resident_likelihood.py \
  --dataset tests/data/test_trees_1000 \
  --mode specieswise \
  --fixed-iters 4,6,8 \
  --preprocess-cpu-cores 16 \
  --measure loss-grad \
  --warmups 1 \
  --reps 3 \
  --family-chunk-size 500 \
  --clade-budget 315000 \
  --batch-packing clade_first_fit \
  --max-wave-size 8192 \
  --max-root-wave-size none \
  --max-dts-partial-rows none \
  --materialize-batches all \
  --prefetch-batches all
```

| Pi/E/Neumann budget | loss+backward median | loss bits | grad inf |
|---:|---:|---:|---:|
| 4 | `5.927877182024531s` | `2156427.0` | `311.95965576171875` |
| 6 | `7.102879976038821s` | `2157095.0` | `312.92724609375` |
| 8 | `8.261407713987865s` | `2157097.25` | `312.93011474609375` |

The first clade-first loss+backward warmup in this post-edit process paid
Triton recompilation and took `44.34427898400463s` for fixed4.  The table above
is steady-state after that warmup.

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
- Finite lazy prefetch depths also lost to `--prefetch-batches all` on the cold
  fixed4 route.  Single samples with depths `4`, `8`, and `12` had first
  likelihood passes of `1.6143241939716972s`, `1.6010761460056528s`, and
  `1.5955134850228205s`; the documented `all` route keeps the pass near
  `1.57s` while hiding remaining batch construction.
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
- The resident uniform kernel warmup trades memory for cold latency.  The cold
  fixed4 route now reserves about `16.7 GiB`, up from about `14.5 GiB` before
  the extra resident-kernel warmup.
- Deferring the resident warmup wait until after prefetch scheduling did not
  improve the median.  Caching E-derived Pi constants did not move the measured
  path, and `torch.inference_mode()` is not valid here because sparse uniform
  ancestor matmul attempts to update inference tensor version counters.
- After the DTS-fill change, nearby clade budgets `310000`, `318000`, and
  `320000` remained roughly tied or slower in repeated cold samples.  `305000`
  and `325000` still hit severe shape cliffs.
- DTS species tile caps of `128` and `2048` lost on the steady fixed4 path.
  `2048` also paid an unacceptable first compile.  The `512` cap was the best
  measured option for this `S=1999` generated benchmark.
- Sweeping the multi-child DTS parent-reduction `tile_splits` setting across
  `32`, `64`, `96`, `128`, and `256` did not produce a reliable improvement.
  Steady fixed4 medians ranged from `1.2849044169997796s` to
  `1.2880120140034705s`, within noise of the current `1.2834076849976555s`
  route, so the code keeps the existing `64` split tile.
- Raising the DTS species block cap from `512` to `1024` was not a stable win.
  One fixed4 steady run reached `1.2805625679902732s`, but a repeat regressed
  to `1.2851436759810895s`, and the cold first likelihood sample was slower at
  `1.5901524130022153s` with higher reserved memory.
- Bucketing the multi-child DTS `MAX_TILES` compile shape to multiples of eight
  lowered the post-compile fixed4 sample to `1.2795667869504541s`, but a fresh
  Triton-cache run exposed unacceptable first-use cost: model init rose to
  `3.424378103984054s` and the first fixed4 likelihood took
  `38.11984705296345s`.
- Reusing a single per-pass DTS scratch buffer lowered peak allocated memory
  from about `5.13 GiB` to about `5.09 GiB`, but timing was not reliable enough
  to keep for a speed-focused route.  Fixed4 medians ranged from
  `1.282040969002992s` to `1.2945115490001626s`, and the cold first-likelihood
  pass stayed around `1.577s`.
- Accumulating the final-Pibar row max/sum while computing the final Pi tile
  avoided one extra read/reduction pass in the wave-step kernel, but the extra
  register pressure erased the theoretical win.  One fixed4 steady run reached
  `1.281840581970755s`, but a repeat regressed to `1.285640132962726s`, and
  cold first-likelihood samples stayed around `1.576s` to `1.581s`.
- Warming additional final-wave-step flag combinations during resident model
  construction did not lower the measured first likelihood.  The first run paid
  `1.403340880991891s` of model init while compiling the extra variants, and
  the fixed4 first likelihood remained around `1.578s` to `1.581s`.
- Rechecking the uniform wave-step species tile cap after the leaf-state
  specialization did not improve the route: raising the generic uniform cap
  from `256` to `512` moved steady fixed4 to about `1.292s`, slower than the
  current `1.283s` measurement.
- A custom Triton row-logsumexp kernel for the already-gathered root rows saved
  only about `1ms` in the instrumented root-likelihood split while adding
  another cold compile/cache surface, so the PyTorch root reduction remains in
  place.
- Computing the shared root denominator once outside the batch loop was also
  slower in a direct timing split: the existing path measured
  `1.2793854749761522s` while the denominator-once loop measured
  `1.2841742759919725s`, and it changed the final fp32 sum by `0.25` bits from
  accumulation order.
- Disabling lazy prefetch/resident kernel warmup still loses.  It lowers model
  construction to about `1.01s`, but the first likelihood pass grows to about
  `1.90s`; the current prefetch/warmup path keeps first likelihood near
  `1.58s`.
- The post-leaf-specialization clade-budget resweep reconfirmed the sharp shape
  cliffs around the current route.  Single cold fixed4 samples at `300000`,
  `310000`, `318000`, `320000`, and `325000` all paid first likelihood times
  from about `14s` to `34s`, while `315000` stayed near `1.58s`.
- Rechecking `max_wave_size` after the current Pi fast paths reconfirmed that
  `8192` is the only good measured shape in the local sweep.  Cold fixed4 first
  likelihood samples were `14.915235649968963s` at `4096`,
  `11.24351201299578s` at `6144`, `1.5818899900186807s` at `8192`,
  `32.38246097502997s` at `12288`, and `20.292580714041833s` at `16384`.
- `max_root_wave_size` likewise stayed uncapped.  Cold fixed4 first-likelihood
  samples were `1.5791340339928865s` uncapped, `1.8603418220300227s` at `1`,
  `2.3634141320362687s` at `2`, `1.8267846549861133s` at `4`,
  `1.5795229229843244s` at `8`, `1.579538435966242s` at `16`, and
  `1.5842193069984205s` at `32`.
- Thread-count pinning is host- and shape-sensitive.  On this 32-core host, an
  isolated generated-dataset retained preprocess/layout timing was best at `16`
  threads, about `0.91s` total, while `12`, `20`, `24`, `32`, and the default
  global pool were slower.
- A narrower cold end-to-end thread-count resweep around `16` did not identify
  a more stable setting.  Single fixed4 cold totals were about `2.525s` at
  `14`, `2.600s` at `15`, `2.548s` at `16`, `2.519s` at `17`, and `2.559s`
  at `18`, but repeats regressed to about `2.560s` at `17` and `2.603s` at
  `14`.  The documented command therefore keeps `--preprocess-cpu-cores 16`.
- Family chunk size was not an active lever under the current `315000` clade
  budget.  Sweeping `250`, `333`, `400`, `500`, `625`, `750`, and `1000` all
  produced `21` batches and similar fixed4 likelihood times.  A paired cold
  comparison of `250` and `500` kept `500` as the documented route: the `250`
  totals were about `2.565s`, `2.523s`, and `2.601s`, while `500` was about
  `2.540s`, `2.553s`, and `2.582s`.
- Rechecking batch packing on the current checkout kept `clade_first_fit`.
  The `sequential` and `depth_first_fit` first samples paid fresh compile/path
  costs (`21.8037438269821s` and `17.070824889990035s` first likelihoods),
  and repeats were still worse end-to-end.  A depth-first sample with a
  temporarily forced forward nonleaf scheduler reduced model init to
  `1.7240403910400346s` and measured likelihood to `1.567154006974306s`, but
  its total remained about `3.29s`, far slower than clade-first.
- `max_dts_partial_rows` did not produce a stable end-to-end win.  Single
  samples at `25000`, `50000`, `75000`, and `100000` sometimes moved the
  first likelihood a few milliseconds, with the best single total
  `2.5357451439485885s` at `75000`, but paired uncapped versus `75000` samples
  were not better overall.  The paired totals were uncapped `2.583690342027694s`,
  `2.5344259899575263s`, `2.5741731580346823s` versus capped
  `2.633321139961481s`, `2.549399826035369s`, `2.553070565976668s`.
- Hoisting fixed-loop convergence and trace booleans out of the common no-grad
  Pi loop did not improve the measured path.  A seven-repetition fixed4
  steady-state run measured `1.2837643179809675s`, slightly slower than the
  current fixed4 baseline, so the change was reverted.
- Scheduling lazy prefetch before constructing resident batch 0 did not produce
  a stable cold win.  Four fixed4 cold samples measured totals of
  `2.608110374014359s`, `2.5429647190030665s`, `2.55199750198517s`, and
  `2.5878938979585654s`; the one low sample overlapped existing noise, while
  the median was worse than the documented route.
- Disabling the fused final-Pibar path and falling back to a separate parent-walk
  kernel was worse on this shape.  A fixed4 steady-state sample measured
  `1.2893211269984022s` median after a `1.7043520050356165s` warmup, so the
  fused final-Pibar store remains enabled.
- Forcing the forward parent-reduced DTS kernels to `num_warps=8` did not help
  this large-`S` generated shape.  The first warmup paid `31.28786089399364s`
  to compile the new Triton variants, and the post-compile fixed4 steady-state
  median was `1.2863251420203596s`, slower than the current route.
- The current `clade_first_fit` batches are already balanced for fixed4 Pi
  timing.  A post-warm per-batch split measured `20` full batches between
  about `0.061s` and `0.063s`, plus a tail batch at `0.03352210501907393s`;
  the summed batch time was `1.2789311109809205s`.  That rules out another
  simple batch-ordering win for the steady path.
- Replacing root-row advanced indexing with `index_select` did not improve the
  loss-only root gather path.  Two fixed4 steady-state medians were
  `1.2822566300164908s` and `1.2869528399896808s`, so the existing root gather
  remains in place.

Differences from HOGENOM:

- HOGENOM's successful counts-free specieswise route used a `8 -> 16 -> 32`
  optimizer budget ladder with a `128` validation check.  On `test_trees_1000`,
  the first useful fidelity point is lower: start the exploratory phase at `4`
  Pi iterations rather than going directly to `8`, then promote once the cheap
  phase stalls.  `6` is already nearly at the high-budget likelihood here.
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
  those with a resident uniform kernel warmup saves a larger additional amount.
  This is less relevant to HOGENOM's accepted route because it has fewer resident
  batches and the optimizer/gradient work dominates more of the run.
- HOGENOM resident batching had fewer batches under the accepted policy.  This
  dataset splits into `21` batches, so reusing a single global/specieswise
  resident E solve across no-grad batches removes repeated E work and is worth
  about two percent on likelihood-only timing.
- Both HOGENOM and `test_trees_1000` have fully covered split waves under the
  measured layouts, so the DTS-fill skip is structurally applicable to both.
  The generated benchmark has many more split rows in its likelihood pass
  (`4.81M` rows across `973` split waves versus HOGENOM's `0.94M` rows across
  `244` split waves in the accepted `depth_first_fit` layout), so the cold
  likelihood win is more visible here.
- The first-leaf-state specialization also applies to HOGENOM, but it is more
  important on `test_trees_1000`: the generated benchmark has `1.61M` leaf rows
  across `206` leaf waves, while the local HOGENOM depth-first fixture measured
  here has only `86k` leaf rows across `13` leaf waves.  HOGENOM's accepted
  route remains more optimizer/gradient dominated.
- Pinning native preprocessing and retained layout generation to `16` CPU
  threads is likewise more useful here than on the local HOGENOM fixture.  For
  `test_trees_1000`, default retained preprocessing and layout generation each
  cost about half a second; for the local HOGENOM depth-first fixture, the same
  work totals only about `0.17s`, and the default pool was essentially tied or
  slightly faster than a `16`-thread pin.
- This dataset also pays initial Pi setup once per resident batch.  Deriving the
  initial state in the wave kernel, using the first non-leaf DTS state directly,
  and letting iteration 1 read that local DTS tensor saves about `0.24s` versus
  the CUDA-math-warmed retained path (`3.160s` to `2.925s`), including about
  `0.07s` beyond the copy-based first non-leaf path.  The same optimization
  should help HOGENOM too, but its accepted end-to-end route has fewer resident
  batches and is dominated more by optimizer/gradient work.
- Larger clade budgets and larger wave caps hurt the first likelihood here.
  `250000`, `280000`, `300000`, `310000`, `312000`, `320000`, `330000`,
  `350000`, `400000`, `500000`, and non-`8192` max-wave samples all hit much
  slower first-pass timings in earlier sweeps.  After the local-DTS input
  change, `318000` was roughly tied with `315000` for fixed4 and used less
  reserved memory, but it did not clearly improve the fixed8 cold sample.  The
  `clade_first_fit`, `315000` clade-budget, `8192` max-wave policy remains the
  main measured route while avoiding those shape cliffs.
