# gpurec-preprocess Source

This folder contains the Rust implementation of the preprocessing extension.
It turns Newick species/gene inputs into JSON payloads that the Python
scheduling layer converts to tensors.

## Files

- `lib.rs`: PyO3 module entry point and dataset parser. It exposes
  `preprocess_dataset` and `plan_batch_layouts`, reads Newick inputs, builds
  species topology arrays, constructs clade/split data for each family, and
  joins batching plus layout planning into the returned JSON.
- `batch_planning.rs`: family batching logic. It validates batch constraints and
  implements sequential, clade-first-fit, and depth-first-fit packing.
- `scheduler.rs`: wave scheduler. It orders leaf and internal clades into
  bounded evaluation waves with phase metadata.
- `layout.rs`: layout encoder. It converts scheduled waves into compact
  permutation, root, leaf, split, reduction, and family-index arrays consumed by
  the GPU kernels.

The Python wrapper in `gpurec/core/scheduling/batching.py` is responsible for
loading this extension and converting the JSON arrays into PyTorch tensors.
