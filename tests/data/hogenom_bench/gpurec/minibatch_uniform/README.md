# Mini-Batch Uniform-Mode Gradient Test

This run tests stochastic optimization of one shared global `[D, L, T]` vector
in uniform-transfer mode.  Gradients are accumulated over fixed 64-family
micro-batches, and the effective batch size increases over the run:

- 64 families for 3 updates
- 128 families for 2 updates
- 256 families for 1 update
- 512 families for 1 update

Settings:

- fixed Pi passes: 20
- reference Pi passes: 160
- backward adjoint: `neumann_terms=8`
- optimizer: Adam on the averaged mini-batch gradient
- learning rate: 0.05
- initial rates: `(0.05, 0.05, 0.05)`

Result:

- initial full fixed20 NLL: 717620.5 bits
- final full fixed20 NLL: 701115.0 bits
- final rates: `(D=0.06231172, L=0.06360151, T=0.06345343)`
- fixed160 validation NLL: 701115.0 bits
- worst per-family `abs(fixed20 - fixed160)`: 0.0 bits

The full-batch Adam comparison in `fullbatch_adam_comparison.csv` used the same
learning rate, fixed-pass settings, and seven optimizer updates. It reached
700914.8125 bits, so in this current implementation the increasing mini-batch
schedule is slightly worse and slower than full-batch Adam. The main cost is
that mini-batches are separate resident `GeneReconModel` instances, so larger
effective batches require several Python-level forward/backward calls.

Reproduce:

```bash
python profiling/optimize_hogenom_minibatch_uniform.py \
  --dataset tests/data/hogenom_bench \
  --output-dir tests/data/hogenom_bench/gpurec/minibatch_uniform \
  --micro-batch-size 64 \
  --schedule 64:3,128:2,256:1,512:1 \
  --fixed-pi 20 \
  --reference-pi 160 \
  --neumann-terms 8 \
  --lr 0.05 \
  --init-rates 0.05,0.05,0.05 \
  --seed 4
```
