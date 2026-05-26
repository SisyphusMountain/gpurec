# test_trees_1000 Resident Likelihood Timing

Date: 2026-05-26

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
is `512` for this large-`S` shape.  The DTS forward kernels no longer specialize
on wave-specific eq1 split counts or ge2 group/tile counts that are only used as
launch sizes or pointer strides, removing many identical first-use Triton
variants from the first likelihood pass.  The no-callback Pi path also avoids
the per-wave progress shim calls used only by diagnostic tracing.  Shared-theta
loss-only streaming now reuses one max-sized Pi/Pibar scratch pair across the
resident batches; this keeps the fixed4 timing band unchanged while avoiding the
large CUDA allocator reserve previously caused by cycling those work tensors
through `21` batches.  The forward pass also prepares the DTS `log_pD`/`log_pS`
addressing metadata once per Pi pass instead of revalidating the same specieswise
parameter layout for every split wave.  Lazy resident-batch prefetch now uses
three background static-layout workers instead of one, hiding more of the
remaining per-batch layout-to-device work under the first likelihood pass.
Shared-theta no-grad streaming also precomputes the root-origination
denominator from the single shared E solve instead of recomputing it for every
resident batch.  It also prepares the shared Pi forward constants and DTS
parameter metadata once after that E solve and reuses them across the `21`
resident batch Pi passes.  Non-genewise retained layouts now also skip the
per-clade `family_idx` tensor that only genewise rate layouts need, reducing
construction traffic and the retained batch memory footprint without changing
the shared/global/specieswise likelihood path.  On the local 32-core host, the cold
benchmark also pins native preprocessing and retained layout generation to `16`
CPU threads; that is faster for this generated dataset than the default global
Rayon pool.

Near-reference cold end-to-end command:

```bash
env PYTHONDONTWRITEBYTECODE=1 GPUREC_MEMORY_POLICY_RESERVE_GIB=0 \
  python profiling/bench_resident_likelihood.py \
  --dataset tests/data/test_trees_1000 \
  --mode specieswise \
  --fixed-iters-e 7 \
  --fixed-iters-pi 4 \
  --neumann-terms 4 \
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

Near-reference cold result:

| Stage | Time |
|---|---:|
| model init / first resident batch | `0.9268650940502994s` |
| first `E=7, Pi=4` likelihood pass plus lazy remaining batches | `1.3175705749890767s` |
| total to first near-reference likelihood | `2.244435669039376s` |

The tied fixed4 route is retained only as a lower-fidelity timing reference: it
is about as fast, but sits `670.25` bits below the fixed8/fixed128 likelihood.
The three-worker lazy-prefetch route produced fixed4 cold totals of
`2.2904225840466097s`, `2.266941985988524s`,
`2.2779467049986124s`, `2.262116832949687s`,
`2.2608266090392135s`, `2.2833276209421456s`,
`2.279989818984177s`, and `2.2708818310056813s`.  After precomputing the
shared origination denominator, three same-route cold samples measured
`2.278459662979003s`, `2.2939429280231707s`, and the current low
`2.257182637054939s`.
The new low remains best-observed rather than a stable median, but the measured
first-pass component moved from the previous one-worker `~1.334s` band to about
`1.321s` to `1.325s` in the three-worker samples.  Immediate same-checkout
one-worker A/B samples measured `2.3445855720201507s` and
`2.316118259972427s`, with measured passes `1.3370230720029213s` and
`1.336572587955743s`.  After omitting unused retained `family_idx`, a serial
same-route `E=8, Pi=4, Neumann=4` sample measured `2.254093322029803s` total
with build `0.9380271580303088s`, measured pass `1.3160661639994942s`,
loss `2157098.25`, peak allocated `5.085336208343506 GiB`, and peak reserved
`5.125 GiB`.  This is not a new timing low, but it keeps the route in-band
while lowering the previous `~5.17 GiB` reserved-memory band.  Peak reserved
CUDA memory remains well below the older `21.36 GiB` high-memory all-prefetch
route.

Cold first-pass tied-budget fidelity samples with the same construction path:

| Pi/E/Neumann budget | total to first likelihood | loss bits | delta vs fixed128 |
|---:|---:|---:|---:|
| 4 | `2.257182637054939s` | `2156427.0` | `670.25` |
| 6 | `2.788982933969237s` | `2157095.0` | `2.25` |
| 8 | `3.2904011249775067s`, `3.3188985750311986s` | `2157097.25` | `0.0` |

Splitting the E and Pi budgets gives a better near-reference point for this
generated dataset.  Keeping Pi at `4` but raising E removes almost all of the
fixed4 likelihood gap without paying the tied fixed6/fixed8 Pi loop cost:

| E budget | Pi budget | cold total samples | loss bits | signed loss delta vs fixed8 |
|---:|---:|---:|---:|---:|
| 6 | 4 | `2.373896275938023s`, `2.318044687039219s`, `2.357071833044756s`, `2.280870435992256s` | `2157096.0` | `-1.25` |
| 7 | 4 | `2.278900177916512s`, `2.287128569034394s`, `2.2731798990280367s`, `2.244435669039376s` | `2157098.25` | `+1.0` |
| 8 | 4 | `2.3150818049907684s`, `2.3092537610209547s`, `2.3501645749202s`, `2.2569857829948887s`, `2.3006827870267443s`, `2.270271884975955s`, `2.253481016959995s` | `2157098.25` | `+1.0` |

The `E=7, Pi=4` route now has a best-observed cold total of
`2.244435669039376s` on the default chunk-500 route, faster than the previous
`E=8, Pi=4` low while preserving the same `2157098.25`-bit printed loss.  It
is effectively tied with the raw fixed4 low but within one bit of the
fixed8/fixed128 likelihood.  The `E=6, Pi=4` route is slightly more optimistic
than fixed8, but still much closer than tied fixed4 and much faster than tied
fixed6.
Earlier current-code rechecks kept `E=8, Pi=4` as the near-reference route:
`E=7, Pi=4` had matched the `2157098.25`-bit loss but measured only
`2.278900177916512s` and `2.287128569034394s`, while `E=6, Pi=4` measured
`2.2726288129924797s` and `2.2791547380620614s` with the slightly optimistic
`2157096.0`-bit loss.  After the retained `family_idx` omission, fresh serial
`E=7, Pi=4` samples measured `2.2731798990280367s` and the new
`2.244435669039376s` low, so the documented near-reference route moved from
E8 to E7.
After adding production split-schedule support, a same-route `E=8, Pi=4,
Neumann=4` cold recheck measured `2.3047162730363198s` total with loss
`2157098.25` bits; it is a normal in-band sample, not a new best.
After hoisting shared Pi forward constants across the resident no-grad stream,
the same route measured `2.253481016959995s` total: `0.9326847809716128s`
model construction and `1.3207962359883823s` for the likelihood pass.  A same-code
direct fixed8 comparison measured `3.2904011249775067s` total with
`2157097.25`-bit loss, so the Pi4-start point was `1.0369201080175117s`
faster while sitting `+1.0` bit above fixed8.
With the retained `family_idx` omission present, a fresh serial fixed8 sample
measured `3.3445263609755784s` total with loss `2157097.25`, so the best
E7/Pi4 sample above was `1.1000906919362024s` faster while sitting `+1.0` bit
from fixed8.
An explicit promotion sequence starting with `E=8, Pi=4` and then evaluating
tied fixed8 measured `2.2692654849379323s` to the Pi4 result, then another
`2.3189413659856655s` for the fixed8 pass, for `4.588206850923598s`
cumulative.  Starting with Pi4 is therefore useful when the approximate result
or first optimizer phase is useful by itself; it is not a free prelude if the
caller immediately requires a tied fixed8 likelihood.
Starting with tied fixed4 and then promoting to `E=8, Pi=4` measured
`3.5742293469957076s` cumulative, so the best near-reference single result
still starts directly at the split Pi4 route rather than from tied fixed4.
Changing `family_chunk_size` did not change the 21-batch retained layout; a
single `666` sample reached `2.2480324909556657s`, but repeats moved back into
the `2.27s` to `2.32s` band.  Rechecking `666` after the route moved to
`E=7, Pi=4` measured `2.2692324600648135s`, `2.255797177029308s`,
`2.274380306014791s`, and `2.2722940339590423s`, so this remains construction
noise rather than a route change.
On the older `E=8, Pi=4` route, setting `CUDA_DEVICE_MAX_CONNECTIONS=1`
produced a then-best observed near-reference sample of `2.247412250027992s`,
but repeats measured `2.302840727963485s`, `2.2829688700148836s`, and
`2.2658810860011727s`.  A fresh `E=7, Pi=4` check with the same environment
measured `2.2591210909886286s`, `2.2517540119588375s`,
`2.2623116039903834s`, and `2.2573027559556067s`, so this is still rejected as
a stable default-route change.

There is also a distinct fast-approximate route below fixed4.  The old tied
`E=2, Pi=2` point is unusable, but increasing only E makes `Pi=2` useful:

| E budget | Pi budget | materialized pass | loss bits | signed loss delta vs fixed8 |
|---:|---:|---:|---:|---:|
| 2 | 2 | `0.7900940429535694s` | `2098963.0` | `-58134.25` |
| 3 | 2 | `0.7688322209869511s` | `2149260.0` | `-7837.25` |
| 4 | 2 | `0.768705825030338s` | `2156863.0` | `-234.25` |
| 5 | 2 | `0.769144629011862s` | `2157447.25` | `+350.0` |
| 6 | 2 | `0.7699056839919649s` | `2157479.75` | `+382.5` |
| 7 | 2 | `0.7702949329977855s` | `2157481.25` | `+384.0` |
| 8 | 2 | `0.7709092769655399s` | `2157481.25` | `+384.0` |

The best fast-approximate point is `E=4, Pi=2`: it is much faster than tied
fixed4 and its absolute loss error is smaller than tied fixed4's `670.25`-bit
gap.  Five cold `E=4, Pi=2` samples measured `1.789836187963374s`,
`1.7681905870558694s`, `1.8027284189593047s`, `1.801696198002901s`, and
`1.7690634109312668s`, all with loss `2156863.0`.

The same fast-approximate route keeps the general `315000` clade budget and
all-batch prefetch settings.  An `E=4, Pi=2` clade-budget sweep measured
`250000`: `1.779803994053509s`, `315000`: `1.791229895025026s`, `400000`:
`1.793410616053734s`, `500000`: `1.8211529290420003s`, and `650000`:
`1.8660963249858469s`; the lower-memory `250000` sample did not beat the best
`315000` repeats.  Prefetch depths measured `none`: `2.071594687993638s`,
`2`: `1.8023037790553644s`, `4`: `1.806620042945724s`, and `all`:
`1.795652323984541s`, so the current all-prefetch route remains best.
A current-code fast-route clade-budget recheck also failed to move the
best-observed sample: `250000` measured `1.779081451066304s` with `26` batches,
while `400000` measured `1.8268427449511364s` with `17` batches and higher
memory.

For `E=4, Pi=2`, a larger `12288` wave cap is a tiny timing near-tie rather
than a clear route change.  Single cold samples measured `4096`:
`1.8095041380729526s`, `8192`: `1.7904685469693504s`, `12288`:
`1.770083504030481s`, and `16384`: `1.838730920047965s`; paired repeats then
measured `8192`: `1.8086025349912234s` and `1.76793152699247s` versus
`12288`: `1.7686329080024734s` and a best-observed
`1.7664462180109695s`.  Materialized steady medians were
`0.7687285720021464s` at `8192` and `0.7682073380565271s` at `12288`.  The
general documented route keeps `8192` for lower memory and consistency with the
near-reference route; `12288` is the current best-observed fast-approximate
sample.
An additional wave/chunk check moved the best-observed fast approximate sample
to `1.7475068190251477s` at `E=4, Pi=2`, `max_wave_size=12288`, and
`family_chunk_size=666`, with the same `2156863.0`-bit loss.  Current-code
repeats after rejecting the early-prefetch prototype measured
`1.7857683220063336s` and `1.77085414895555s`, so `12288` remains the
best-observed fast setting but not a stable median improvement over the lower
memory `8192` route.  A same-code post-hoist `12288`/chunk-`666` recheck
measured `1.7595705519779585s` total with `0.9342424949863926s` construction
and `0.825328056991566s` measured likelihood time, so it stayed in the existing
fast-approximate band rather than setting a new low.
`CUDA_DEVICE_MAX_CONNECTIONS=1` did not improve the fast-approximate route
either: three `E=4, Pi=2`, `12288`, chunk-`666` samples measured
`1.7939132550382055s`, `1.759844618034549s`, and `1.7760014350060374s`.

A materialized split-budget sweep with Pi fixed at `4` measured `E=5` at
`1.2753524020081386s` and `2157060.5` bits, `E=6` at
`1.2751596000161953s` and `2157096.0` bits, `E=7` at
`1.2767304159933701s` and `2157098.25` bits, and `E=8` at
`1.2778630289831199s` and `2157098.25` bits.  `E=5` therefore leaves about
`36.75` bits on the table with no useful speed advantage over `E=6`; `E=6` and
`E=7` are the practical split-budget choices depending on whether a slight
optimistic or slight pessimistic one-bit error is preferable.  A later
materialized five-repetition recheck after the retained `family_idx` omission
measured `E=7, Pi=4` at `1.2762305510113947s` median and
`1.2732166629866697s` minimum, versus `E=8, Pi=4` at
`1.2793222990003414s` median and `1.2765266370261088s` minimum, with the same
`2157098.25` printed loss.
Older cold `E=7, Pi=4` samples also matched the `E=8, Pi=4` loss at
`2157098.25` bits, but did not beat the older best `E=8` cold sample: totals were
`2.3259418689995073s`, `2.314654977992177s`,
`2.267007304006256s`, and `2.2859832789981738s`.
A current-code recheck after the shared-constant hoist was worse still at
`2.3588287950260565s`; the latest retained-layout recheck reversed that route
choice and made `E=7, Pi=4` the documented near-reference default.

A split-budget clade-budget sweep kept the same `315000` route decision as the
tied fixed4 path.  With `E=8, Pi=4`, single cold samples measured `250000`:
`2.283307211997453s` at `4.21 GiB` reserved, `315000`:
`2.2804238710086793s` at `5.23 GiB`, `400000`:
`2.2799866489949636s` at `6.46 GiB`, `500000`:
`2.2937483189743944s` at `7.95 GiB`, and `650000`:
`2.355606388999149s` at `10.27 GiB`.  The small `400000` tie does not justify
the higher memory and larger wave envelope, and the previous fixed4 repeats
already showed this region does not hold a stable cold win.  Rechecking the new
`E=7, Pi=4` route likewise rejected the neighboring clade budgets: `250000`
measured `2.2842069599428214s` with `26` batches and `4.126953125 GiB`
reserved, while `400000` measured `2.296551337989513s` with `17` batches and
`6.419921875 GiB` reserved.

The split-budget route also keeps the documented `8192` max-wave cap.  One
`E=8, Pi=4` cold sweep measured `4096`: `2.3022571059991606s`, `8192`:
`2.2856174019398168s`, `12288`: `2.2725106599973515s`, and `16384`:
`2.267264521040488s`, but paired repeats did not establish a stable cold win.
A materialized steady check measured `8192` at `1.2848945879959501s` median and
`1.2801758240093477s` minimum, while `16384` measured
`1.2845551520003937s` median and `1.2798394349520095s` minimum with higher
memory.  The near-tie is not enough to move the default away from the lower
memory `8192` shape.  A fresh `E=7, Pi=4` sweep kept the same decision:
`4096` measured `2.295460112974979s` and `12288` measured
`2.2948793239775114s`, both slower than the documented `8192` route.

Split-budget rechecks also kept the root-wave and DTS partial-row caps
disabled.  With `E=8, Pi=4`, root caps at `8`, `16`, and `32` measured cold
totals of `2.270672336977441s`, `2.273461366945412s`, and
`2.2887684240122326s`, but their measured passes stayed at `1.329s` to
`1.332s`; the apparent total movement was construction noise.  DTS partial-row
caps at `50000`, `75000`, and `100000` measured `2.2849829280166887s`,
`2.2872178590041585s`, and `2.275073692027945s`, all slower than the
same-sweep uncapped `2.2581794050056487s` sample and with no loss change.

The split-budget route also still wants all-batch lazy prefetch.  With
`E=8, Pi=4`, `--prefetch-batches none` lowered model construction to
`0.8873169249854982s` but raised the first pass to `1.6338533840025775s`, for
`2.5211703089880757s` total.  Finite depths `2` and `4` measured
`2.308664307987783s` and `2.304723095963709s`; `all` measured
`2.262025747972075s` in the same sweep.
The split-budget route also keeps the three-worker resident prefetch pool.
Temporarily lowering the worker count to `2` gave `E=8, Pi=4` cold totals of
`2.3399722679168917s` and `2.2740334490081295s`; raising it to `4` gave
`2.3272328549646772s` and `2.3391964139882475s`.  Neither setting beat the
documented three-worker samples.
Current-code CPU and packing rechecks also kept the route unchanged.  With
`E=8, Pi=4`, `preprocess_cpu_cores` values `8`, `16`, `24`, and `32` measured
`2.399679420981556s`, `2.2792265199823305s`, `2.317675110010896s`, and
`2.3776359270559624s`, keeping `16` as the local 32-core host setting.
`depth_first_fit`, the HOGENOM-style packing, measured `3.5548714509932324s`
because construction was much slower on this generated shape; `sequential`
measured `2.360776627960149s`.  `clade_first_fit` remains the generated-tree
default.
Runtime environment rechecks did not establish a route change.  Explicit
`CUDA_MODULE_LOADING=LAZY` matched the base route at `2.270426261005923s`;
`CUDA_MODULE_LOADING=EAGER` was rejected after raising construction to
`7.578426808002405s` and total time to `8.904745513980743s`.
`CUDA_DEVICE_MAX_CONNECTIONS=1` gave the best single sample on the older E8/Pi4
route but did not repeat as a stable improvement and did not beat the newer
E7/Pi4 low.  `CUDA_DEVICE_MAX_CONNECTIONS=8` measured `2.2815428160247393s`.
Setting `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` measured `2.2856818459695205s`.

The production workflow now also supports opt-in early phase promotion for
specieswise `adagrad-restarts`: `adagrad_restart_phase_loss_patience > 0`
advances to the next scheduled budget when the current phase's loss stalls,
while the schedule's `steps` values remain hard caps.  A short validation run
on this dataset used `8/4:1.0:3,8:0.5:3` with phase patience `1`, fixed phase
caps, and fixed8 final validation.  Process wall time was `63.42s`; optimizer
summary elapsed was `61.245588069956284s`.  The first phase solver stats
reported Pi/Neumann `4`, the second reported Pi/Neumann `8`, and final
fixed8 validation ended at `1761134.0` bits with projected gradient infinity
norm `11.61174201965332`.  This run validates the Pi4-start workflow path; it
is a six-step smoke, not an optimum search.

## End-To-End Specieswise Optimization From 0.05

The current outer-optimization benchmark now starts from uniform
`theta_init_d = theta_init_l = theta_init_t = 0.05` and uses the production
`gpurec optimize` workflow, not the likelihood-only harness.  Unless noted,
runs used specieswise rates, `family_chunk_size=500`, `clade_budget=315000`,
`batch_packing=clade_first_fit`, `max_wave_size=8192`,
`preprocess_cpu_cores=16`, `GPUREC_MEMORY_POLICY_RESERVE_GIB=0`, and fixed8
final validation.

Measured routes:

| Route | Schedule / optimizer | Wall time | Final NLL bits | Projected grad inf | Notes |
|---|---|---:|---:|---:|---|
| A | `7/4:1.0:12,8:0.5:8,16/8:0.5:8`, `adagrad-restarts`, phase patience `1` | `170.31107250403147s` | `1707816.5` | `3.9513802528381348` | E16/Pi8 at `lr=0.5` was too aggressive and tripped the phase-loss stop after two repair evaluations. |
| B | `7/4:1.0:12,8:0.5:40`, `adagrad-restarts`, phase patience `1` | `407.7515940640005s` | `1704378.5` | `0.9977045059204102` | Current best short Pi4-start route. Fixed8, fixed16, and fixed32 validation all returned `1704378.5` bits. |
| C | `8:1.0:12,8:0.5:40`, direct fixed8 `adagrad-restarts`, phase patience `1` | `430.63535784796113s` | `1704378.625` | `0.9977216720581055` | Same trajectory as B within fp32 noise, but about `22.88s` slower because the first 12 steps use tied fixed8 instead of E7/Pi4. |
| 12-step Pi4 + L-BFGS-B tail | `7/4:1.0:12`, then `lbfgsb`, `lr=0.1`, 20 L-BFGS-B steps | `469.3958375949878s` combined | `1703574.875` | `28.24788475036621` | Too early for the L-BFGS-B switch; the tail spends most of its work catching up from a high NLL. |
| A + L-BFGS-B tail | Resume A, `lbfgsb`, `lr=0.1`, 20 L-BFGS-B steps | `547.463271428016s` combined | `1700309.75` | `28.042160034179688` | Better time-to-NLL than switching at step 52. |
| A + L-BFGS-B tail, larger step | Resume A, `lbfgsb`, `lr=0.2`, 20 L-BFGS-B steps | `548.0136427210527s` combined | `1699621.5` validated | `12.487090110778809` validated | Same wall-time band as `lr=0.1`, much better objective and residual. Fixed8, fixed16, and fixed32 validation all returned `1699621.5` bits. |
| A + L-BFGS-B tail, `lr=0.3` | Resume A, `lbfgsb`, `lr=0.3`, 20 L-BFGS-B steps | `548.8858391030226s` combined | `1699565.875` validated | `3.1161506175994873` validated | First route in the current best objective band from uniform `0.05`. Fixed8, fixed16, and fixed32 validation all returned `1699565.875` bits. |
| A + L-BFGS-B tail, `lr=0.4` | Resume A, `lbfgsb`, `lr=0.4`, 20 L-BFGS-B steps | `550.6247605990502s` combined | `1699564.25` validated | `2.950817108154297` validated | Lower residual than `lr=0.5`, but `4.375` bits higher and `2.33s` slower. Fixed8, fixed16, and fixed32 validation all returned `1699564.25` bits. |
| A + L-BFGS-B tail, `lr=0.5` | Resume A, `lbfgsb`, `lr=0.5`, 20 L-BFGS-B steps | `548.2928214290296s` combined | `1699559.875` validated | `4.012872695922852` validated | The workflow summary reported `1699560.0` bits, while manual fixed8, fixed16, and fixed32 validation all returned `1699559.875` bits. |
| A + L-BFGS-B tail, fast best | Resume A, `lbfgsb`, `lr=0.6`, 20 L-BFGS-B steps | `549.3528225499904s` combined | `1699550.875` validated | `5.094226837158203` validated | Fast validated route, `0.375` bits lower than the old long B-continuation tail in about `42%` of its wall time. Fixed8, fixed16, and fixed32 validation all returned `1699550.875` bits. |
| A + L-BFGS-B continuation | Continue the `lr=0.6` tail for 10 more L-BFGS-B steps | `746.7829839309561s` combined | `1699505.5` validated | `9.706039428710938` validated | Fixed8, fixed16, and fixed32 validation all returned `1699505.5` bits, but the final accepted step still improved by `8.625` bits. |
| A + longer L-BFGS-B continuation | Continue the `lr=0.6` tail for 20 more L-BFGS-B steps total | `944.7325820129481s` combined | `1699470.25` validated | `3.031355857849121` validated | Segmented 20+10+10-step measurement. Fixed8, fixed16, and fixed32 validation all returned `1699470.25` bits; the final two optimizer steps improved by only `0.375` bits each. |
| A + uninterrupted L-BFGS-B tail | Resume A, `lbfgsb`, `lr=0.6`, 40 L-BFGS-B steps | `911.1706395330257s` combined | `1699469.5` validated | `2.8489274978637695` | Best fixed-length endpoint so far. Fixed8, fixed16, and fixed32 validation all returned `1699469.5` bits; avoiding two resume/final-eval cycles saved about `33.56s` versus the segmented route. |
| A + adaptive L-BFGS-B tail | Resume A, `lbfgsb`, `lr=0.6`, max 60 L-BFGS-B steps, `loss_change_tol=1`, `loss_patience=2`, `projected_grad_tol=10` | `908.8200418340275s` combined | `1699470.125` validated | `4.6751389503479` | Practical no-hand-tuned endpoint: stopped by `loss_change_patience` after two sub-1-bit improvements. Fixed8, fixed16, and fixed32 validation all returned `1699470.125` bits. |
| A + post-stall continuation check, current lowest | Continue the uninterrupted 40-step `lr=0.6` tail with `loss_change_tol=1`, `loss_patience=2`, `projected_grad_tol=10` | `982.2694296170375s` combined | `1699468.25` validated | `1.4578275680541992` | Lowest validated objective observed so far. This is a segmented continuation check; fixed8, fixed16, and fixed32 validation all returned `1699468.25` bits. |
| A + L-BFGS-B tail, rejected high step | Resume A, `lbfgsb`, `lr=0.7`, 20 L-BFGS-B steps | `548.9326881880406s` combined | `1699557.125` | `13.341646194458008` | Upper-bracket reject: worse than `lr=0.6` by `6.25` bits with a much larger residual, so no manual fixed16/fixed32 validation was run. |
| A + longer L-BFGS-B tail | Resume A, `lbfgsb`, `lr=0.1`, 30 L-BFGS-B steps total | `745.2935658869683s` combined | `1699746.125` | `20.155467987060547` | Longer `lr=0.1` tail is slower and worse than the 20-step `lr=0.3` tail. Fixed8, fixed16, and fixed32 validation all returned `1699746.125` bits. |
| B + early L-BFGS-B tail | Resume B, `lbfgsb`, `lr=0.1`, 20 L-BFGS-B steps | `786.3332051589969s` combined | `1700015.5` | `27.92882537841797` | Later switch is slower and slightly worse than the 30-step tail from A. |
| B + fixed-cap continuation | Resume B with `7/4:1.0:12,8:0.5:80`, phase patience `0` | `732.2706705760211s` combined | `1702241.25` | `0.5673627853393555` | Still improving by about `31` bits per step at the cap. |
| B + continuation + L-BFGS-B tail | Resume the 92-step Adagrad point, `lbfgsb`, `lr=0.1`, 30 L-BFGS-B steps total | `1308.1464419650729s` combined | `1699551.25` | `28.2503719329834` | Fixed8, fixed16, and fixed32 validation all returned `1699551.25` bits, but the projected gradient is large, so this is not an optimum certificate. |

The Pi4-start comparison is the cleanest result so far: route B and the direct
fixed8 route C reach the same likelihood band, but B saves the cost of 12 tied
fixed8 gradient evaluations.  The average first-phase step time was
`6.2320073523345245s` for E7/Pi4 versus `7.9824362590006785s` for tied fixed8;
the second fixed8 phase averaged `7.9114916901962715s` for B and
`7.950046648895659s` for C.

The fastest validated endpoint that beats the old long B-continuation tail is
the 22-step route A followed by 20 L-BFGS-B steps at `lr=0.6`: it lands at
`1699550.875` bits in `549.35s`, `0.375` bits below the previous best long
B-continuation tail while saving about `758.79s`.  Continuing that same L-BFGS-B
state for 10 more steps reaches `1699505.5` bits in `746.78s`.  Running the
full 40-step L-BFGS-B tail uninterrupted from route A reaches `1699469.5` bits
in `911.17s`.  This replaces the segmented 20+10+10-step measurement: it is
`0.75` bits lower and about `33.56s` faster because it avoids two
resume/final-evaluation cycles.

For a practical dataset-independent tail stop, the one-shot route with
`loss_change_tol=1`, `loss_patience=2`, and `projected_grad_tol=10` stopped
automatically at `1699470.125` bits in `908.82s`, after two consecutive
sub-1-bit improvements.  A post-stall continuation check from the fixed
40-step endpoint reached the lowest objective observed so far, `1699468.25`
bits, in `982.27s`; the extra `71.10s` bought only `1.25` bits beyond the
uninterrupted 40-step endpoint.

This is still not a formal optimum, but it is the first point in this sequence
where the marginal objective movement dropped sharply.  In the segmented run,
the final two optimizer steps improved by only `0.375` bits each; in the
uninterrupted run, the late improvements were still small and the final
projected-gradient infinity norm was `2.84893`.  The adaptive one-shot endpoint
is the route to prefer when avoiding per-dataset step tuning; the continuation
check is only a lower-objective reference point.

The `lr=0.4` variant has a smaller residual, `2.95082`, but is `13.375` bits
higher and `1.27s` slower than the 20-step `lr=0.6` route in the measured runs.
Pushing to `lr=0.7` worsened the 20-step endpoint to `1699557.125` bits with
projected-gradient infinity norm `13.34`, so the current useful initial
L-BFGS-B step scale is bracketed between `0.5` and `0.7`.

The more conservative 92-step Adagrad point has a higher NLL, `1702241.25`,
but a smaller projected-gradient residual, `0.56736`.  The current known gap
from the short 52-step Pi4-start route B to the best objective seen in this
round is `4827.25` bits.  Switching from B directly into L-BFGS-B reaches
`1700015.5` bits in `786.33s`, which is `464.25` bits above the longer
B-continuation L-BFGS-B tail while saving about `522s`.

Switch-point screening shows that L-BFGS-B should not start immediately after
the first 12 E7/Pi4 steps, but also should not wait for 52 Adagrad steps if the
target is time-to-low-NLL.  The best measured switch so far is after route A's
22 steps.  With `lr=0.1`, 20 L-BFGS-B steps reached `1700309.75` bits in
`547.46s`, and 30 L-BFGS-B steps reached `1699746.125` bits in `745.29s`.
Raising the L-BFGS-B step scale to `lr=0.3` made the 20-step tail reach
`1699565.875` bits in `548.89s`; `lr=0.4` reached `1699564.25` bits in
`550.62s`; `lr=0.5` reached `1699559.875` bits in `548.29s`; and `lr=0.6`
reached `1699550.875` bits in `549.35s`; `lr=0.7` fell back to `1699557.125`
bits.  Continuing the accepted `lr=0.6` L-BFGS-B state for 10 more steps reached
`1699505.5` bits in `746.78s` combined, and another 10 steps reached
`1699470.25` bits in `944.73s` when measured as segmented resumes.  A single
uninterrupted 40-step `lr=0.6` tail reached `1699469.5` bits in `911.17s`.
Adding `loss_change_tol=1`, `loss_patience=2`, and `projected_grad_tol=10`
stopped the same route automatically at `1699470.125` bits in `908.82s`.  The
improvement is coming from end-to-end parameter movement, not from spending
more work on Pi/E iterations.

While testing resumed continuations, a workflow stop-rule issue was found and
fixed: extending a completed checkpoint used to restore `previous_objective`
from the final evaluation row, so the first resumed optimizer row could see a
fake zero loss delta and immediately trip `adagrad_restart_phase_loss_patience`.
Completed-checkpoint resumes now reset the loss-stall baseline when more
optimizer steps are requested; running checkpoints still keep their live
stop-rule state.

Adaptive fixed-point stopping is also rejected for this likelihood-only route.
The benchmark harness can now pass `--adaptive-iters`, but the convergence
checks were more expensive than the fixed split-budget shortcut.  At the
current `E=8, Pi=4` budget, adaptive checks measured
`2.4835817339480855s` with unchanged `2157098.25`-bit loss.  Adaptive max
`E=8, Pi=8` measured `2.9879215210094117s` at `pi_max_diff_tol=1e-5` and
`2.8730849079438485s` at `1e-6`, with tied fixed8 loss
`2157097.25`; adaptive max `E=16, Pi=16` was slower still at
`3.4208932110341266s` and `4.050838949973695s`.

Split-budget near-reference command:

```bash
env PYTHONDONTWRITEBYTECODE=1 GPUREC_MEMORY_POLICY_RESERVE_GIB=0 \
  python profiling/bench_resident_likelihood.py \
  --dataset tests/data/test_trees_1000 \
  --mode specieswise \
  --fixed-iters-e 7 \
  --fixed-iters-pi 4 \
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

The same 2026-05-26 sanity run measured a direct fixed8 cold total of
`3.3119525730144233s`, with `0.949136951006949s` model construction and
`2.3628156220074743s` for the measured pass.

Following the fixed4 pass with a fixed8 pass on the same cold resident model is
more expensive than a direct fixed8 validation, because the current fixed-Pi
path does not reuse the previous fixed4 Pi state.  A fresh 2026-05-26
`--fixed-iters 4,8` sample measured:

| Step | Incremental pass | Cumulative from process start | loss bits | delta vs fixed8 |
|---:|---:|---:|---:|---:|
| build | `0.9365219409810379s` | `0.9365219409810379s` | n/a | n/a |
| 4 | `1.3344161220011301s` | `2.270938062982168s` | `2156427.0` | `670.25` |
| 8 | `2.3105589839979075s` | `4.5814970469800755s` | `2157097.25` | `0.0` |

Progressive fixed4-start sample on the same route, using one cold model and
then evaluating `4,6,8` in sequence.  A 2026-05-26 rerun after the scalar-loss
accumulation rejection measured:

| Step | Incremental pass | Cumulative from process start | loss bits | delta vs fixed8 |
|---:|---:|---:|---:|---:|
| build | `0.9357642909744754s` | `0.9357642909744754s` | n/a | n/a |
| 4 | `1.3270101890084334s` | `2.262774479982909s` | `2156427.0` | `670.25` |
| 6 | `1.7997798179858364s` | `4.062554297968745s` | `2157095.0` | `2.25` |
| 8 | `2.3256094389944337s` | `6.388163736963179s` | `2157097.25` | `0.0` |

A direct cold fixed8 sample under the same cached code path took
`3.365096386987716s`.  So starting at fixed4 is useful when the caller can use
an early approximate likelihood, or for optimizer phases that will move theta
before promotion.  It is not a shortcut to the fixed8 likelihood itself because
the current fixed-Pi path does not carry a reusable Pi state from the fixed4
pass into the fixed8 pass.

The retained Rust chunked-layout builder now schedules chunks through borrowed
CCP slices instead of cloning each family's split arrays into scheduler items,
and its scheduler allocates child-dedup hash sets only for high-degree parents
instead of creating one empty set per clade.  On this dataset, isolated
retained-layout generation after dataset preprocessing measured
`0.4546270810533315s`, `0.448365535994526s`, `0.41906961804488674s`,
`0.41099122702144086s`, and `0.4032032579998486s`; the previous detailed setup
split had the same native chunked layout generation near `0.496s`.  The
end-to-end win is partly hidden by overlapping layout work with CUDA setup, but
direct fixed4/fixed6 cold samples improved to the rows above.

Reusing the root-loss Pi/Pibar scratch tensors changed the memory behavior more
than the timing before the later prefetch-worker resweep.  Fixed4 lazy samples
at that point measured `2.29978136200225s`,
`2.3235359659884125s`, `2.3087819020147435s`, `2.2922973530367017s`,
`2.2839784509851597s`, `2.2908750560018234s`,
`2.2710854980396107s`, `2.2995317989843898s`, and
`2.299258347018622s`; all reserved about `5.16 GiB` instead of the previous
`21.36 GiB` high-memory all-prefetch route.  In a one-sample prefetch-depth
check after this change, `--prefetch-batches none` was still slower
(`2.543537666031625s`), while depth `4` (`2.2950046080513857s`) and `all`
(`2.2922973530367017s`) were essentially tied in memory and timing.  A later
same-route `16`-core repeat produced a then-current `2.2710854980396107s` low.

Before removing the unused DTS compile-shape arguments from the eq1 and ge2
kernel signatures, the retained-layout fixed4 route took about `2.55s` to the
first likelihood.
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
`1.282079865981359s`, `1.2819753359653987s`, and
`1.2804367360076867s` medians before the eq1 DTS specialization cleanup, and
`1.2814912779722363s` after removing the eq1 split-count specialization.  After
also moving ge2 group/tile shape arguments to runtime values, a shorter
three-repetition steady check measured `1.2800687580020167s` median, with
unchanged loss.
Two fresh paired samples kept lazy construction and eager materialization within
noise: lazy totals were about `2.580s` and `2.538s`, while materialize-all totals
were about `2.559s` and `2.592s`.  Materialize-all lowers the first likelihood
pass itself by about `50ms`, but the up-front materialization cost cancels that
on the end-to-end total.  After removing `n_eq1` from the DTS eq1 compile shape,
lazy cold totals repeated at `2.3492117439745925s` and `2.3551931240945123s`;
materialize-all reached `2.3555265389732085s` and kept reserved memory lower
than the lazy all-prefetch sample.  After the ge2 compile-shape cleanup, lazy
fixed4 totals measured `2.3258029819699004s`, `2.326640389917884s`, and
`2.3110102329519577s`.  The first local runs immediately after changing Triton
signatures paid one-time cache rebuilds and are not route timing samples.

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
| 4 | `1.2767350050271489s` | `2156427.0` | `670.25` |
| 6 | `1.816627103020437s` | `2157095.0` | `2.25` |
| 8 | `2.3491738659795374s` | `2157097.25` | `0.0` |
| 128 | `35.236629224033095s` | `2157097.25` | `0.0` |

With `--materialize-batches all`, the latest DTS-parameter-layout-hoist fixed4
check measured seven fixed4 repetitions after one warmup, with a minimum of
`1.2721429850207642s`.  The `6/8` rows above are earlier three-repetition
medians after one warmup; the `128` row is a single validation sample.

The last detailed fixed4 bottleneck profile on the same resident layout
predates the DTS signature cleanup, but it still identifies the dominant kernel
buckets:

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
Hoisting DTS forward parameter layout checks is a CPU-side cleanup around those
split waves: it improved the materialized fixed4 median to
`1.2767350050271489s`, while lazy cold samples after the change measured
`2.316268014954403s`, `2.2867282620281912s`, `2.3230369999655522s`, and
`2.3498632150003687s`.

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

- Tied `4` Pi/E iterations is a good warm phase.  It is about `40%` cheaper
  than tied `8` for likelihood-only and about `27%` cheaper than tied `8` for
  loss+backward.
- Do not start a tied-budget ladder at `2` on this dataset.  A tied `E=2, Pi=2`
  steady-state sample measured `0.7704234900302254s`, but its loss was only
  `2098963.0` bits, about `58134.25` bits below the fixed8/fixed128 value.
  The useful fast low-budget point is split `E=4, Pi=2`, not tied `2`.
- Do not treat tied `4` as final fidelity on this dataset.  It is still `670`
  bits away from the fixed128 reference, in the optimistic direction.
  `E=4, Pi=2` is faster and closer in absolute loss error, and `E=7, Pi=4` is
  the current near-reference point.
- `8`, `32`, and `128` agree at the printed precision, so higher fixed budgets
  are useful mainly as periodic validation points, not every-step work.
- The fixed4 error is spread across the resident batches, not concentrated in
  a small subset.  In a per-batch fixed4/fixed6/fixed8 split, full batches each
  contributed about `29` to `38` bits of the fixed4-to-fixed8 delta, and the
  small tail contributed about `16` bits.  Upgrading only a few batches to
  fixed6 would therefore leave hundreds of bits of fixed4 error; uniform fixed6
  remains the practical near-reference budget.

Rejected follow-ups:

- Earlier lazy-prefetch worker sweeps before scratch reuse were inconclusive:
  two workers improved the fixed4 loss pass in isolation, but the four-process
  cold total stayed around `3.178s`, and three/four workers were slower.  After
  scratch reuse and the later cold-path cleanups, the same-checkout resweep
  changed the answer: two workers reached `2.266941985988524s`, three workers
  reached `2.2608266090392135s`, four workers had a slower outlier at
  `2.3021931539988145s`, and eight workers regressed to
  `2.296754330978729s`.  The retained cap is therefore three workers, not an
  unbounded prefetch pool.
- Finite lazy prefetch depths were a memory tradeoff before root-loss scratch
  reuse.  After the DTS compile-shape cleanup, `--prefetch-batches all`
  produced fixed4 cold samples of `2.307332646974828s` and
  `2.308705120929517s`, but depth `4` was close (`2.317219710967038s`,
  `2.319942276983056s`, and `2.324342556006741s`) while reserving about
  `14.34 GiB` instead of `21.36 GiB`.  With scratch reuse, depth `4` and `all`
  both reserve about `5.16 GiB`, and `all` keeps a slightly better sample.
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
- The resident uniform kernel warmup and all-batch lazy prefetch still trade
  construction overlap for cold latency, but root-loss scratch reuse removes
  the old large reserve from the all-prefetch route.  Current depth `4` and
  `all` samples both reserve about `5.16 GiB`, while disabling prefetch still
  slows the first fixed4 likelihood.
- Deferring the resident warmup wait until after prefetch scheduling did not
  improve the median.  Caching E-derived Pi constants did not move the measured
  path, and `torch.inference_mode()` is not valid here because sparse uniform
  ancestor matmul attempts to update inference tensor version counters.
- After the DTS compile-shape cleanup, nearby and larger clade budgets still do
  not clearly beat `315000`.  Current single fixed4 cold samples measured
  `2.3146543949842453s` at `310000`, `2.3289775589946657s` at `400000`, and
  `2.373008435999509s` at `500000`; `330000` reduced the layout to `20`
  batches and reached `2.3149897510302253s` in one sample, but repeats stayed
  around `2.324s` to `2.397s`.
- DTS species tile caps of `128` and `2048` lost on the steady fixed4 path.
  `2048` also paid an unacceptable first compile.  The `512` cap was the best
  measured option for this `S=1999` generated benchmark.
- Sweeping the multi-child DTS parent-reduction `tile_splits` setting across
  `32`, `64`, `96`, `128`, and `256` did not produce a reliable improvement.
  Steady fixed4 medians ranged from `1.2849044169997796s` to
  `1.2880120140034705s`, within noise of the then-current
  `1.2834076849976555s` route, so the code keeps the existing `64` split tile.
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
- Collapsing the `21` per-batch root-row reductions into one full-family
  root-row buffer was also slower.  The prototype preserved global family order
  and passed the targeted no-grad/origination tests, but the extra root-row copy
  raised fixed4 steady timing to `1.2807044959627092s` median, with a
  `1.2760750549496152s` minimum, so the per-batch root reductions remain.
- Computing the shared root denominator once outside the batch loop was also
  slower in a direct timing split: the existing path measured
  `1.2793854749761522s` while the denominator-once loop measured
  `1.2841742759919725s`, and it changed the final fp32 sum by `0.25` bits from
  accumulation order.
- Disabling lazy prefetch/resident kernel warmup still loses.  It lowers model
  construction to about `1.01s`, but the first likelihood pass grows to about
  `1.90s`; the current prefetch/warmup path keeps first likelihood near
  `1.34s`.
- The latest clade-budget sweep shows the current route is no longer dominated
  by first-use shape cliffs at neighboring budgets, but the lower batch counts
  from larger budgets do not translate into a faster first fixed4 likelihood.
  The pass itself stays near `1.34s`; model construction and allocator noise
  decide the cold total.
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
- A later post-scratch/current-code thread resweep likewise failed to replace
  `16`.  Single samples at `12`, `14`, `16`, `18`, and `20` threads measured
  `2.285087577009108s`, `2.311186804960016s`, `2.2968016039812937s`,
  `2.2749476860044524s`, and `2.349719259014819s`, but the low `18`-thread
  sample did not repeat (`2.3079166330280714s`, `2.2973673989763483s`,
  `2.3156927170348354s`).  The `12`-thread repeat also landed in-band
  (`2.297266439010855s`, `2.300386350019835s`, `2.318546007038094s`).
- Family chunk size was not an active lever under the current `315000` clade
  budget.  Sweeping `250`, `333`, `400`, `500`, `625`, `750`, and `1000` all
  produced `21` batches and similar fixed4 likelihood times.  A paired cold
  comparison of `250` and `500` kept `500` as the documented route: the `250`
  totals were about `2.565s`, `2.523s`, and `2.601s`, while `500` was about
  `2.540s`, `2.553s`, and `2.582s`.
- The HOGENOM-style no-family-cap setting (`--family-chunk-size 0`) also did
  not become a clear route improvement here.  It still produced `21` batches
  with the same max clade and wave envelope; fixed4 steady medians were
  `1.2764153979951516s` and `1.2823010300053284s`, while cold totals were
  about `2.543s` and `2.567s`, overlapping the documented `500`-family route.
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
- There are no additional whole-wave final-Pibar skips beyond the current
  root-only waves.  A layout audit found that the current `21` skipped waves
  cover exactly the same `65` rows as the set of waves whose rows are never
  used as later DTS children; no extra whole-wave skip is available.
- Precomputing the root-only wave flags in the retained Rust layout was not a
  win.  The prototype passed targeted layout and resident forward tests after
  guarding the flags to the no-grad root-row path, but fixed4 steady timing
  measured `1.278411909006536s` median and `1.274784742970951s` minimum, and a
  cold fixed4 sample measured `2.293933361012023s` total to first likelihood.
  That is in the existing timing band but slower than the current best steady
  median, so the added layout field was reverted.
- There is also no row-level final-Pibar skip hiding inside mixed waves on this
  generated layout.  A row audit found `6,416,248` rows used as later DTS
  children and `1,000` root rows across `6,417,248` total rows, with `0`
  non-root rows unused by later DTS.
- Forcing the forward parent-reduced DTS kernels to `num_warps=8` did not help
  this large-`S` generated shape.  The first warmup paid `31.28786089399364s`
  to compile the new Triton variants, and the post-compile fixed4 steady-state
  median was `1.2863251420203596s`, slower than the current route.
- Specializing dense eq1 DTS waves so the parent row was the program id instead
  of loading `eq1_reduce_idx` was also neutral-to-slower.  The targeted tests
  passed, but fixed4 steady timing measured `1.2779726349981502s` median and
  `1.27439327101456s` minimum, not better than the DTS-parameter-layout-hoist
  route, so the extra Triton variant was reverted.
- Caching prepared DTS parameter metadata in `species_helpers` across the `21`
  resident batch Pi calls was also not a stable win.  The targeted
  specieswise/chunked tests passed, and the first fixed4 steady check measured
  `1.276590519992169s` median with a `1.2730084469658323s` minimum, but the
  cold sample stayed in-band at `2.297392721986398s`.  A leaner cache-key repeat
  regressed to `1.2817456009797752s` median, so the cache was reverted and the
  direct per-pass `prepare_dts_forward_param` calls remain.
- Suppressing Pi solver telemetry collection on the batched no-grad full-loss
  stream was likewise too small.  The targeted no-grad/chunked/specieswise tests
  passed, but fixed4 steady timing measured `1.2773516669985838s` median and
  `1.2740960630471818s` minimum, and the cold fixed4 total was
  `2.311466729035601s`.  The normal telemetry path remains in place.
- Making core NVTX ranges opt-in by environment variable did not improve the
  normal likelihood run.  With NVTX disabled by default, fixed4 steady timing
  measured `1.2794013429665938s` median and `1.2745843709562905s` minimum, and
  the cold/materialized build-plus-first-measured sample was
  `2.369563494983595s`.  The profiling helper therefore keeps its existing
  always-available NVTX behavior.
- Precomputing the shared root-origination denominator once from the reused E
  solve is a small accepted loss-only cleanup.  It removes repeated
  `exp2/mean/log2` denominator work across the `21` resident batches.  The
  steady fixed4 materialized check measured `1.2784311089781113s` median and
  `1.2743548050057143s` minimum, and the cold route produced a new low
  `2.257182637054939s`.
- A post-denominator-reuse route resweep still kept the documented
  `315000` clade budget.  Single fixed4 cold samples measured:

  | clade budget | batches | first pass | total to first likelihood | peak reserved |
  |---:|---:|---:|---:|---:|
  | `250000` | `26` | `1.3293938959832303s` | `2.303362254984677s` | `4.189453125 GiB` |
  | `315000` | `21` | `1.3200315139838494s` | `2.2602628639433533s` | `5.16796875 GiB` |
  | `400000` | `17` | `1.3185203209868632s` | `2.289328532991931s` | `6.458984375 GiB` |
  | `500000` | `13` | `1.3148925850400701s` | `2.279398131009657s` | `7.984375 GiB` |
  | `650000` | `10` | `1.3140173489809968s` | `2.342297429975588s` | `10.267578125 GiB` |

  Larger batches slightly reduced the measured likelihood pass, but construction
  time and peak memory rose enough that they did not improve end-to-end time.
  This differs from the HOGENOM tuning pattern: here the generated-family shape
  already gives enough work per batch, so fewer resident batches are not a clear
  win.
- A current-code `max_wave_size` resweep also kept `8192`.  One pass over
  `4096`, `8192`, `12288`, and `16384` measured cold totals of
  `2.298833917011507s`, `2.264543463010341s`, `2.257202097971458s`, and
  `2.307180391973816s`, respectively.  The `12288` sample was a near-tie with
  the best historical `8192` sample and reduced the max wave count from `61` to
  `58`, but paired repeats did not hold the win: `8192` totals were
  `2.280936630035285s`, `2.2816185380215757s`, and
  `2.264976181962993s`, while `12288` totals were
  `2.264912271988578s`, `2.2910129600204527s`, and
  `2.28301867301343s`.
- A narrow current-code CPU-thread recheck at `12`, `16`, and `18` preprocessing
  threads did not displace the documented `16`-thread route.  Paired fixed4 cold
  totals were `12`: `2.296057304018177s`, `2.2838334499974735s`; `16`:
  `2.2834402269800194s`, `2.3150741640129127s`; and `18`:
  `2.314814416982699s`, `2.2755942060030065s`.  The spread is construction
  noise rather than a stable route change.
- A scalar root-row NLL-sum helper was prototyped and reverted.  It added a
  `compute_nll_root_rows_sum()` loss-only helper so no-grad callers could avoid
  materializing the final per-family NLL vector.  Targeted tests passed while the
  prototype was present (`24 passed`), but fixed4 cold samples stayed in-band
  (`2.313574720057659s`, `2.3038053709897213s`, `2.2635737379896455s`) and the
  materialized steady median regressed to `1.2852614839794114s` versus the
  current `1.2784311089781113s` median.  The root reductions were already a
  small part of the pass, so this was not an aligned improvement.  Repeating
  the same idea on the current `E=7, Pi=4` route was still not stable: the first
  materialized check improved to `1.2744981619762257s` median, but the repeat
  regressed to `1.280531055002939s`; cold totals stayed in-band at
  `2.264361812034622s`, `2.2908622620161623s`, `2.273455304966774s`, and
  `2.2519061420462094s`, with no new low.
- Removing redundant-looking `Pi.contiguous()` and `Pibar.contiguous()` calls
  from the DTS launch wrapper was also reverted.  The prototype passed the
  targeted DTS/resident tests (`14 passed`) and a materialized fixed4 sample
  improved slightly to `1.2767550030257553s` median, but cold end-to-end samples
  did not improve (`2.2904487129999325s`, `2.2644186799880117s`,
  `2.269676612049807s`).  Keeping the explicit contiguous boundary is preferable
  until it produces a cold-path win.
- Warming the two regular phase-1 leaf/no-DTS wave-step variants during resident
  CUDA warmup was also rejected.  An audit of actual retained batches found
  `206` phase-1 no-split waves, `922` phase-2 eq1-only split waves, `30`
  phase-2 mixed split waves, and `21` phase-3 ge2-only split waves.  The
  prototype passed targeted resident tests (`13 passed`) but did not improve
  cold end-to-end timing: fixed4 totals were `2.26270189601928s`,
  `2.3115326509578153s`, `2.304071097052656s`, `2.260307011019904s`,
  `2.284385512000881s`, `2.2860829039709643s`, `2.2775780250085518s`,
  `2.2877858880674466s`, and `2.274424984003417s`.
- A fresh post-denominator timing split confirmed that wave-step/Pibar and DTS
  remain the fixed4 bottlenecks.  In a materialized post-warm measured pass,
  total wall time was `1.2839083970175125s`; the largest CUDA-event buckets were
  final DTS wave-steps (`331.646 ms` across `952` calls), eq1-only phase-2 DTS
  (`203.701 ms` across `922` calls), non-final DTS wave-steps (`370.149 ms`
  across `1967` calls), phase-1 leaf/no-DTS wave-steps (`229.317 ms` across
  `618` calls), and mixed phase-2 eq1/ge2 DTS (`115.757 ms` across `30` calls).
  This points away from further root-likelihood work and toward Pi/DTS kernels.
- Lowering the uniform wave-step launch from `8` warps to `4` was rejected.  The
  targeted CUDA/kernel tests passed (`7 passed`), but fixed4 materialized timing
  regressed to `1.2981207949924283s` median and `1.2949794699670747s` minimum.
  The wave-step kernels therefore keep `num_warps=8`.
- Unrolling the fixed4 wave self-loop in Python for the no-trace/no-convergence
  path was also reverted.  The prototype passed targeted tests (`14 passed`) and
  gave a small materialized fixed4 median improvement to `1.2767849509837106s`,
  but cold end-to-end samples did not beat the current route:
  `2.3544185089995153s`, `2.278216293954756s`,
  `2.2834861860610545s`, `2.340617485984694s`, and
  `2.2646979700075462s`.  The added branch complexity is not justified without a
  cold-path win.
- Conditional `tile_splits=128` for mixed eq1/ge2 DTS waves improved
  materialized fixed4 timing but was still rejected for the cold route.  The
  layout has `30` mixed phase-2 waves carrying `935` ge2 groups, about `3.0M`
  ge2 splits, and `47,326` 64-split tiles; the `21` pure phase-3 ge2 waves carry
  only `65` groups, `208,509` ge2 splits, and `3,287` 64-split tiles.  Using
  `128` only for mixed waves passed targeted tests (`14 passed`) and measured
  `1.2752698680269532s` materialized median with `1.2719556670053862s` minimum,
  but cold samples without an added warmup were `2.279916053987108s`,
  `2.293243663967587s`, `2.2992008170112967s`, `2.364783897995949s`, and
  `2.3065213810186833s`.  Adding a tiny mixed-DTS warmup for the new variant
  also passed tests (`14 passed`) but raised construction enough that cold
  samples still did not improve: `2.4507630320149474s`,
  `2.3197203549789265s`, `2.2956745679839514s`,
  `2.3365640210104175s`, and `2.273633966979105s`.  The default
  `tile_splits=64` therefore remains the end-to-end route.
- The current `clade_first_fit` batches are already balanced for fixed4 Pi
  timing.  A post-warm per-batch split measured `20` full batches between
  about `0.061s` and `0.063s`, plus a tail batch at `0.03352210501907393s`;
  the summed batch time was `1.2789311109809205s`.  That rules out another
  simple batch-ordering win for the steady path.
- Replacing root-row advanced indexing with `index_select` did not improve the
  loss-only root gather path.  Two fixed4 steady-state medians were
  `1.2822566300164908s` and `1.2869528399896808s`, so the existing root gather
  remains in place.
- Precomputing a dense species ancestor table for branch-free parent walks was
  also slower here.  The prototype passed the targeted uniform forward/specieswise
  tests, but a seven-repetition fixed4 steady run regressed to
  `1.4639405140187591s` median with the same `2156427.0`-bit loss, versus the
  restored parent-chain path at `1.2804367360076867s` median.
- Applying `torch.inference_mode()` only around the Pi/root-loss portion that
  reuses a pre-solved E was neutral.  The targeted uniform forward/specieswise
  tests passed, but fixed4 steady-state timing was `1.2804541109944694s`
  median, indistinguishable from the current path, so the no-grad wrapper stays
  in plain `torch.no_grad()`.
- Disabling lazy prefetch/resident warmup while still materializing all batches
  before the first likelihood was slower.  The build took
  `1.0070954780094326s`, but the first fixed4 pass rose to
  `1.8282219489919953s`, for about `2.835s` end-to-end.
- Warming the direct local-DTS input variant used by the accepted first
  non-leaf optimization did not improve the cold route.  The prototype passed
  targeted no-grad/specieswise/uniform tests (`13 passed`), but five fixed4
  cold samples were `2.2882246950175613s`, `2.3021020019659773s`,
  `2.2710205909679644s`, `2.3239040829939768s`, and
  `2.2830842310213484s`, with pass times around `1.324s` to `1.329s`.
  That extra warmup is therefore rejected.
- Starting lazy background prefetch before building batch 0 was also rejected.
  The prototype passed `py_compile` and the targeted no-grad/specieswise/uniform
  tests (`13 passed`), but then-current-route `E=8, Pi=4` samples stayed in the
  existing band (`2.273360916005913s`, `2.2933849390246905s`) and did not beat
  the best documented near-reference cold total.  Finite prefetch depths were
  still slower than the all-prefetch route.
- Adaptive fixed-point stopping is rejected for the generated-tree likelihood
  route.  It is useful as a validation mode, but the per-wave convergence checks
  are too expensive here and do not beat the explicit split budgets.
- In-place accumulation of the `21` per-batch scalar losses in the shared-theta
  loss-only stream was also rejected.  The prototype passed the targeted
  no-grad/specieswise/uniform tests (`13 passed`), but a seven-repetition
  fixed4 materialized run measured `1.2801847120281309s` median and
  `1.2755018040188588s` minimum, slower than the current documented
  `1.2784311089781113s` median.  The root/loss scalar path remains too small to
  matter next to the Pi/DTS kernels.
- Disabling forward solver-stat recording in a throwaway process was also
  rejected.  The normal materialized `E=8, Pi=4` route measured `1.2724366540205665s`
  median over three post-warm passes, while monkeypatching the no-grad stat
  recorder to a no-op measured `1.2777643840527162s` median.  The stat conversion
  is not a visible bottleneck, and preserving workflow diagnostics is the better
  tradeoff.
- Replacing the root-row base-2 logsumexp helper with an equivalent
  `torch.logsumexp(root_rows * ln2) / ln2` form was neutral to slightly worse.
  A throwaway monkeypatch preserved the `2157098.25`-bit `E=8, Pi=4` loss, but
  measured `1.2732835609931499s` median over three post-warm materialized passes,
  versus `1.2724366540205665s` for the current helper.  The base-2 helper stays
  in place.
- Reducing the `21` scalar batch losses by stacking them and summing once at the
  end also did not help the current split-budget route.  The targeted
  no-grad/model tests passed (`7 passed`), but the cold `E=8, Pi=4` sample
  measured `2.2636918509961106s`, worse than the documented `2.253481016959995s`
  low, so the existing per-batch scalar accumulation remains in place.
- A scalar-only root-row NLL helper was also rejected.  The prototype returned
  `compute_nll_root_rows(...).sum()` directly for loss-only callers and passed
  targeted origination/no-grad/model tests (`16 passed`), but the cold
  `E=8, Pi=4` route measured `2.282638431992382s` and a post-warm materialized
  check measured `1.274828838009853s` median, worse than the recent
  `1.2724366540205665s` materialized baseline.  The existing vector helper plus
  `.sum()` remains in place.
- Fresh continuation rechecks on 2026-05-26 kept `E=7, Pi=4, Neumann=4` as the
  near-reference route.  Three cold samples measured `2.274839742050972s`,
  `2.2544504269608296s`, and `2.266165839973837s`, all with
  `2157098.25`-bit loss.  The measured likelihood pass stayed in the
  `1.313s` to `1.319s` band, while construction accounted for most of the
  total spread.
- Rechecking the slightly optimistic `E=6, Pi=4, Neumann=4` route did not expose
  a speed win.  Cold samples measured `2.3143601480405778s`,
  `2.2669091350398958s`, and `2.2747382109519094s`, with
  `2157096.0`-bit loss.  Its pass time was effectively identical to E7/Pi4, so
  the only difference was construction noise plus the less conservative loss
  bias.
- Lowering Neumann terms for loss-only computation was also rejected.  Although
  no-gradient likelihood does not use the implicit-gradient Neumann series,
  `E=7, Pi=4, Neumann=1` measured `2.4379382629995234s`,
  `2.401937930029817s`, and `2.31613227003254s`; `Neumann=2` measured
  `2.2946807079715654s`, `2.2932448709616438s`, and
  `2.2546982710482553s`.  The loss stayed `2157098.25` bits, but the route did
  not beat the documented `Neumann=4` command.
- Building the same benchmark through `GeneReconModel.from_alerax_families()`
  was not a cold-path replacement for the documented direct tree-list source.
  Fresh cold samples from `families.txt` measured `2.3390709669911303s`,
  `2.2643726890091784s`, and `2.3282454780419357s`, all with the same
  `2157098.25`-bit loss and `21` resident batches.  Occasional faster
  same-process samples were warm-cache effects rather than a reproducible
  end-to-end route.
- A fresh `E=7, Pi=4` CPU-thread sweep still left `16` as the documented local
  setting.  Single cold totals were `10`: `2.360891352989711s`, `12`:
  `2.261137897032313s`, `14`: `2.279004193027504s`, `16`:
  `2.3394252699799836s`, `18`: `2.2783811920089647s`, and `20`:
  `2.283500710967928s`; the default Rayon pool was slower in two follow-up
  samples at `2.393924950971268s` and `2.390638910001144s`.  The isolated
  `12`-thread low did not beat the existing E7/Pi4 best and matches the earlier
  conclusion that this knob is construction noise around the `16`-thread
  setting rather than a new route.
- Direct lower-level construction through `GeneDataset.from_retained_preprocess`
  plus `GeneReconModel(...)` was also only in-band, measuring
  `2.274539925972931s` with the same `2157098.25`-bit loss.  The public
  `from_trees()` constructor is not the active bottleneck.
- A direct-parent-row specialization for eq1 DTS waves was prototyped and
  reverted.  An audit showed all `952` eq1 DTS waves in the retained
  `test_trees_1000` layout have sequential parent rows, including `922` pure
  eq1 waves, so the prototype marked that metadata and let the Triton eq1 DTS
  kernel use `program_id(0)` instead of loading `eq1_reduce_idx[n]`.  Targeted
  scheduler/kernel checks passed while the prototype was present, but cold
  E7/Pi4 timings did not improve.  Without a matching warmup, samples were
  `2.461927135940641s`, `2.276818887970876s`, and
  `2.2694924459792674s`; changing the existing resident DTS warmup to compile
  the direct-parent variant still measured `2.422379998024553s`,
  `2.3362076219636947s`, `2.2770920139737427s`, and
  `2.282298989011906s`.  The extra specialization does not pay for itself on
  the cold end-to-end route.

Differences from HOGENOM:

- HOGENOM's successful counts-free specieswise route used a `8 -> 16 -> 32`
  optimizer budget ladder with a `128` validation check.  On `test_trees_1000`,
  the first useful fidelity points are lower and decoupled: `E=4, Pi=2` is the
  best fast approximate likelihood point found so far, and `E=7, Pi=4` is the
  best measured near-reference point, within one bit of tied fixed8/fixed128
  while staying in the fixed4 Pi timing band.  Those split-budget shortcuts were
  not the HOGENOM route, where the accepted schedule used tied higher-fidelity
  optimizer phases.
- In actual outer optimization from uniform `0.05`, `test_trees_1000` also
  benefits from starting with E7/Pi4: the Pi4-start Adagrad route matched the
  direct fixed8 route's final likelihood band while saving about `22.88s`.
  HOGENOM did not show the same behavior: fixed4/Pi4 starts there saved time
  but reached a materially worse basin unless the run was allowed a longer
  fixed32 repair phase.
- The best `test_trees_1000` outer route now switches early from the short
  E7/Pi4-start Adagrad route into L-BFGS-B with a larger step scale (`lr=0.6`).
  That differs from HOGENOM's accepted pattern, where the useful basin came
  from a longer tied-budget ladder and high-fidelity polish rather than an early
  aggressive L-BFGS-B tail.
- For specieswise use across datasets, `test_trees_1000` now has a working
  likelihood-stall route: the early L-BFGS-B tail can stop with
  `loss_change_tol=1`, `loss_patience=2`, and a relaxed projected-gradient guard
  instead of a hand-counted number of tail steps.  That is again different from
  HOGENOM's route, where fixed high-fidelity phase lengths were the accepted
  tuning handle.
- The first `test_trees_1000` high-fidelity promotion does not currently need
  HOGENOM's tied fixed16/fixed32 validation ladder for objective consistency:
  fixed8, fixed16, and fixed32 validations matched exactly at the measured
  52-step Adagrad point and at the L-BFGS-B tail point.  HOGENOM required the
  higher-budget ladder because low-budget checkpoints could validate thousands
  of bits away from the true objective.
- HOGENOM worked best with `depth_first_fit`.  On `test_trees_1000`, depth-first
  originally gave similar steady likelihood timing but paid an extra Python
  construction prepass.  Retaining the native Rust layouts removes that Python
  prepass; with the current timings, `clade_first_fit` remains the best measured
  default for the generated tree benchmark.  A fresh `E=8, Pi=4` recheck
  measured `3.5548714509932324s` for `depth_first_fit` and
  `2.360776627960149s` for `sequential`, versus the `clade_first_fit`
  `2.27s` band.
- HOGENOM benchmark scripts are naturally AleRax-family-file driven.  On
  `test_trees_1000`, switching this likelihood benchmark from direct tree paths
  to `from_alerax_families()` did not improve cold time; both sources produce
  the same `21` retained batches and loss, and source parsing is not the active
  bottleneck once CUDA/module warmup is cold.
- HOGENOM's depth-first path should keep the exhaustive scheduler policy because
  wave compaction mattered there.  On `test_trees_1000`, the retained
  `clade_first_fit` chunks had the same measured max wave count with the
  forward scheduler policy, so skipping the extra candidate schedulers saves
  about `0.2s` of cold construction.
- HOGENOM needed full family tensors during the accepted route.  On
  `test_trees_1000`, the resident Rust layout already contains the tensors used
  for likelihood, so compact family summaries remove about `0.3s` from cold
  construction without changing likelihood values.
- The retained `family_idx` omission is likewise a generated likelihood-path
  cleanup rather than a HOGENOM optimizer change.  Shared/global/specieswise
  retained batches never index rates by per-clade family id, while genewise
  retained batches still opt in to that tensor.  On `test_trees_1000`, avoiding
  it trims memory and layout work across `21` resident batches; HOGENOM's
  accepted specieswise route is dominated more by optimizer-gradient work.
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
- The shared Pi constant hoist is similarly specific to the generated
  likelihood-only path: `test_trees_1000` streams `21` resident batches from one
  shared E solve, so removing repeated per-batch shared-parameter preparation is
  visible in the first pass.  HOGENOM's accepted route spent more time in
  gradient-bearing optimizer steps and fewer resident batch transitions.
- HOGENOM resident batching had fewer batches under the accepted policy.  This
  dataset splits into `21` batches, so reusing a single global/specieswise
  resident E solve across no-grad batches removes repeated E work and is worth
  about two percent on likelihood-only timing.
- The new root-loss Pi/Pibar scratch reuse is also more visible on
  `test_trees_1000` because the generated route streams `21` resident batches.
  It mostly fixes allocator reserve rather than raw math time here; HOGENOM's
  accepted route had fewer resident batches and was more gradient/optimizer
  dominated, so this is less route-changing there.
- Three-worker lazy prefetch is also a generated-dataset-specific cold-path
  improvement.  `test_trees_1000` still has enough lazy batch-static work during
  the first fixed4 pass to hide with a small worker pool.  The accepted HOGENOM
  multifidelity script materializes resident batches before optimizing and has
  fewer/larger optimizer-gradient passes, so this change is not expected to
  move its route in the same way.
- The shared-denominator cleanup is likewise more visible in the generated
  likelihood-only benchmark than in the accepted HOGENOM route.  It removes one
  repeated scalar denominator computation per resident batch after the shared E
  solve; HOGENOM's route spends much more wall time in gradient and optimizer
  work.
- HOGENOM had a faster high-memory no-family-cap scheduling option.  On
  `test_trees_1000`, no-family-cap does not reduce the batch or wave envelope
  under `clade_first_fit` at the current clade budget, so the finite
  `--family-chunk-size 500` route remains the documented command.
- Both HOGENOM and `test_trees_1000` have fully covered split waves under the
  measured layouts, so the DTS-fill skip is structurally applicable to both.
  The generated benchmark has many more split rows in its likelihood pass
  (`4.81M` rows across `973` split waves versus HOGENOM's `0.94M` rows across
  `244` split waves in the accepted `depth_first_fit` layout), so the cold
  likelihood win is more visible here.
- The DTS parameter-layout hoist follows the same shape asymmetry.  It removes
  repeated Python validation around every split wave, which is more visible on
  `test_trees_1000` because its accepted route launches about four times as many
  split waves as the local HOGENOM fixture.
- The same split-wave asymmetry makes the DTS compile-shape cleanup more valuable
  on `test_trees_1000`: hundreds of pure single-child split waves no longer
  generate redundant eq1 DTS variants during the first likelihood pass, and the
  smaller number of mixed/multi-child waves no longer specialize ge2 kernels on
  group/tile counts.
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
- The Rust scheduler input and child-dedup cleanups are also more visible here
  than on HOGENOM.  The generated dataset builds `21` retained resident layouts
  and scans about `6.4M` clades, so avoiding scheduler-side split-array clones
  and millions of empty child-dedup sets moves the construction split
  measurably.  HOGENOM's local fixture has fewer resident batches and much
  smaller preprocessing/layout cost, so this is mostly a cleanliness and
  memory-traffic improvement there rather than a route-changing optimizer
  result.
- The scheduler also skips a redundant sort after enumerating leaf clades:
  the enumeration already yields `(family, clade)` order.  The latest fixed4
  cold sample with that cleanup was `2.2928918009856716s`, effectively tied
  with the current best.  The cleanup matters more to `test_trees_1000` than
  HOGENOM because this generated layout has `1.61M` leaf rows across `206` leaf
  waves.
- This dataset also pays initial Pi setup once per resident batch.  Deriving the
  initial state in the wave kernel, using the first non-leaf DTS state directly,
  and letting iteration 1 read that local DTS tensor saves about `0.24s` versus
  the CUDA-math-warmed retained path (`3.160s` to `2.925s`), including about
  `0.07s` beyond the copy-based first non-leaf path.  The same optimization
  should help HOGENOM too, but its accepted end-to-end route has fewer resident
  batches and is dominated more by optimizer/gradient work.
- Adding a separate warmup launch for that local-DTS input variant does not
  help `test_trees_1000`, so it is also unlikely to be worth carrying into the
  HOGENOM route.  The accepted resident warmup already covers the expensive
  general uniform/DTS kernel families well enough; extra one-off warmup launches
  mainly add cold construction noise.
- Larger clade budgets and larger wave caps do not buy the same kind of win
  here that the HOGENOM high-memory route found.  Current `330000` to `500000`
  clade-budget samples reduce the number of resident batches from `21` down to
  as few as `13`, but fixed4 cold totals stay around `2.32s` to `2.37s` or
  worse because the larger wave envelopes and allocation behavior offset the
  lower batch count.  The
  `clade_first_fit`, `315000` clade-budget, `8192` max-wave policy remains the
  main measured route while avoiding those shape cliffs.
