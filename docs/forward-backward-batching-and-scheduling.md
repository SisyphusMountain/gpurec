# Forward/Backward Batching and Wave Scheduling Notes

This note records a practical constraint for likelihood and gradient evaluation:
the useful batch size is not determined by the forward pass alone.  When we need
gradients, the batch must leave enough GPU memory for both the forward values
that backward reuses and the temporary buffers used by the reverse pass.

## Forward and backward are coupled

The backward pass is not an independent likelihood computation.  It is the
reverse pass through the values produced by the forward dynamic program/fixed
point iterations.

In simplified form:

```text
forward:
  for wave in topological order:
      Pi[wave], Pibar[wave] = update(Pi[past rows], parameters)

  loss = root_reduce(Pi[root rows])

backward:
  seed dPi[root rows] from dloss/droot

  for wave in reverse topological order:
      use saved Pi, Pibar, row statistics, and split metadata
      accumulate dPi[past rows] and parameter gradients
```

Backward reuses the converged forward values because the derivative formulas
contain normalized terms such as:

```text
exp(term - Pi[parent])
exp(Pi[ancestor] - row_max)
exp(Pibar[child] - normalization)
```

Recomputing all of these values during backward would cost time, add numerical
drift, and complicate the optimized kernels.  The current optimized paths
therefore keep the relevant forward state resident or reconstruct only selected
small/stable intermediates.

## Memory budget for gradient batches

For gradient evaluation, the chunk size must satisfy roughly:

```text
forward saved state
+ backward adjoints
+ backward scratch buffers
+ DAG / split / scheduling metadata
+ safety margin
< available GPU memory
```

The dominant forward state is usually:

```text
Pi        [C, S]
Pibar     [C, S]
row stats [C] or similar
```

where `C` is the number of clade rows in the current family chunk and `S` is the
number of species.

The dominant backward state is usually:

```text
dPi / rhs / v_k style adjoints   [C, S] or wave-sized buffers
DTS / Pibar VJP scratch          wave-sized or split-sized buffers
parameter-gradient partials
active/pruning masks
```

This matters most in uniform and specieswise modes because the large arrays are
dense over the full species dimension.  Uniform has shared parameters and
specieswise has species-indexed parameters, but both modes still carry large
`[clade, species]` tensors.  Genewise changes parameter layout, but the same
memory accounting applies to `Pi`, `Pibar`, and adjoints.

## Practical batching strategy

A robust training/evaluation loop should stream family chunks:

```text
for chunk in scheduled_family_chunks:
    forward(chunk)
    save Pi/Pibar/root values needed by backward

    backward(chunk)
    accumulate parameter gradients into global gradient buffers

    release chunk-local Pi/Pibar/adjoint/scratch state
```

This avoids keeping all families' full forward and backward state alive at the
same time.  It also gives us a knob for balancing occupancy against memory.

The largest chunk that fits in memory is not necessarily the fastest one.  A
chunk can be too small and underfill the GPU, but it can also be too large and
increase scratch traffic, worsen cache behavior, or force less favorable wave
composition.  Chunk size should therefore be treated as a benchmarked scheduling
parameter, not only as an OOM guard.

## Wave scheduling still needs more work

We should continue looking at wave scheduling as a first-class optimization for
total likelihood time, especially when computing many families together.

The forward pass is wave-structured.  Small waves launch too little work, while
very large or irregular waves can concentrate expensive split reductions into a
few kernels.  The scheduler should try to balance:

```text
1. enough rows per wave to keep the GPU occupied;
2. bounded memory footprint for Pi, Pibar, and scratch;
3. manageable split fanout in root and high-level clades;
4. compatibility with the reverse pass, which will traverse related waves
   backward and reuse the same saved state.
```

A useful future direction is a memory-aware wave scheduler:

```text
build candidate waves across many families
estimate:
    row_count * S state memory
    split_count and max parent fanout
    expected scratch size
pack waves/chunks until the memory budget is near but below the target
prefer batches that avoid tiny launches and avoid extreme split-fanout spikes
```

This should be benchmarked on the large test-tree datasets, not only on the
small three-family cases.  The key metric is end-to-end likelihood or
forward+backward time for the full dataset under a fixed GPU memory budget.
