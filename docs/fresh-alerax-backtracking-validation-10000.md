# Fresh AleRax Backtracking Validation, 10000 Samples

## Target

Historical target using an untracked generated fixture:

- Species tree: `tests/data/test_trees_100/sp.nwk`
- Gene tree: `tests/data/test_trees_100/g_0003.nwk`
- Family: `family_0003`
- Rates: `D=0.0191209`, `L=0.0199312`, `T=0.0208267`
- Reason for choosing this tree: it has a nontrivial event mixture with
  speciation-loss, duplication, transfer, and transfer-loss scenarios.

## Fresh AleRax Run

Output directory:

```text
output_alerax_backtracking_validation/family_0003_10000/alerax
```

Command:

```bash
AleRax_oliver/build/bin/alerax \
  -s tests/data/test_trees_100/sp.nwk \
  -f output_alerax_backtracking_validation/family_0003_10000/families_family_0003.txt \
  -p output_alerax_backtracking_validation/family_0003_10000/alerax \
  -g 10000 \
  --seed 20260517 \
  --model-parametrization GLOBAL \
  --fix-rates \
  --d 0.0191209 \
  --l 0.0199312 \
  --t 0.0208267 \
  --species-tree-search SKIP \
  --rec-model UndatedDTL
```

AleRax wrote 10000 `family_0003_eventCounts_*.txt` files.

## Mean Event Counts

```text
event  alerax_mean  gpurec_mean  mean_delta
S      117.332800   117.330500   -0.002300
SL     6.737000     6.737900      0.000900
D      4.996200     4.994400     -0.001800
DL     0.000000     0.000000      0.000000
T      3.671000     3.675100      0.004100
TL     0.340600     0.339100     -0.001500
L      0.000000     0.000000      0.000000
Leaf   127.000000   127.000000    0.000000
```

The largest absolute mean difference over 10000 samples is `0.0041` events.

## Statistical Tests

For each event count distribution, I ran:

- A nonparametric bootstrap 95% CI for `mean(gpurec) - mean(AleRax)`.
- Two-sample Kolmogorov-Smirnov test.
- Two-sided Mann-Whitney U test.
- Chi-square contingency test over the discrete event-count histogram.

```text
event  mean_delta  bootstrap_95_ci        ks_p        mannwhitney_p  chi2_p
S      -0.002300   [-0.015803, 0.010600]  1.000000    0.762055       0.586921
SL      0.000900   [-0.014200, 0.016000]  1.000000    0.730255       0.679558
D      -0.001800   [-0.004303, 0.000800]  1.000000    0.164705       0.324848
DL      0.000000   [0.000000, 0.000000]   1.000000    1.000000       1.000000
T       0.004100   [-0.009100, 0.017600]  1.000000    0.604296       0.766119
TL     -0.001500   [-0.014900, 0.011300]  1.000000    0.792405       0.726463
L       0.000000   [0.000000, 0.000000]   1.000000    1.000000       1.000000
Leaf    0.000000   [0.000000, 0.000000]   1.000000    1.000000       1.000000
```

All bootstrap intervals contain zero and no test rejects similarity between the
AleRax and gpurec event-count distributions.
