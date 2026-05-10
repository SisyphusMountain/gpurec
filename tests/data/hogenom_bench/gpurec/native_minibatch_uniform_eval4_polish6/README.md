# Native uniform mini-batch benchmark

This run uses one resident `UniformChunkedReconModel` and samples existing
family chunks instead of rebuilding per-mini-batch models.

Command:

```bash
python -u profiling/optimize_hogenom_minibatch_uniform.py \
  --dataset tests/data/hogenom_bench \
  --output-dir tests/data/hogenom_bench/gpurec/native_minibatch_uniform_eval4_polish6 \
  --family-chunk-size 32 \
  --schedule 64:8,128:8,256:6,512:4,1055:6 \
  --fixed-pi 20 \
  --reference-pi 160 \
  --neumann-terms 8 \
  --max-wave-size 8192 \
  --lr 0.05 \
  --init-rates 0.05,0.05,0.05 \
  --seed 4 \
  --full-eval-interval 4
```

Result on `hogenom_bench`:

- Initial fixed-20 NLL: `717620.5625`
- Best fixed-20 NLL: `654216.25`
- Validation fixed-20 NLL: `654216.1875`
- Validation fixed-160 NLL: `654216.1875`
- Max per-family `abs(fixed20 - fixed160)`: `0.0`
- Families over `0.1` fixed-pass error: `0`
- Wall time: `42.84s`
- Rates: `D=0.08098245`, `L=0.15133692`, `T=0.13993219`

For comparison, `native_minibatch_uniform/fullbatch_adam_28_comparison.csv`
ran full-batch Adam with the same fixed-pass likelihood settings for 28
updates. It reached NLL `660086.125` in `88.67s`; at comparable wall time to
this run it was around step 13, with NLL `687803.6875`.
