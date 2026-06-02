# gpurec.core.scheduling

Batching and wave-layout planning for core GPU kernels. This layer converts
family dictionaries and native preprocessing output into tensor metadata used by
the inference and backward kernels.

## Files

- `batching.py`: Loads the native `gpurec_preprocess` extension, preprocesses
  datasets, plans family batches under family-count and clade-count limits, and
  builds wave-layout dictionaries containing permutation, leaf-species, root,
  family-index, and split-reduction tensors.

## Layout Notes

- Native planning entry points return JSON payloads that are converted into
  PyTorch tensors on the requested device.
- The Python `build_wave_layout` fallback groups leaf and internal clades into
  bounded waves, remaps clade indices through a wave-order permutation, and
  prepares split metadata for the DTS kernels.
- Batch packing supports sequential order plus first-fit strategies keyed by
  family depth or clade count.
