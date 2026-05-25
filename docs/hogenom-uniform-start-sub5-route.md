# HOGENOM Uniform-Start Sub-5-Minute Route

Date: 2026-05-25.

Status: validated on local HOGENOM with CUDA on an NVIDIA GeForce RTX 4090.

## Requirement

Optimize HOGENOM specieswise D/L/T rates end to end from 0.05 for every
species branch, finish in less than 5 minutes, and accept a final NLL within
10 bits of the current best reference.

## Current Reference

The best repeated fixed128 reference from the fast route is:

```text
/tmp/gpurec_hogenom_counts_adagrad_route_goal_verify
fixed128 NLL = 526789.625 bits
```

The accepted `best + 10` threshold is therefore `526799.625` bits.

## Route

The route is implemented by:

```bash
python scripts/benchmark_hogenom_counts_adagrad_route.py \
  --out-dir /tmp/gpurec_hogenom_uniform_start_counts_route_verify \
  --device cuda
```

The script records a real uniform start before applying the counts-guided
optimizer jump:

1. Build the specieswise HOGENOM model with `theta_init_d/l/t = 0.05`.
2. Evaluate and save `checkpoints/uniform_start.pt`.
3. Apply a counts-guided D/L/T step from AleRax species event counts and save
   `checkpoints/counts_guided_start.pt`.
4. Run fixed16 Adagrad to step 40 with `lr=1.0`.
5. Reset Adagrad state, run fixed16 Adagrad to step 100 with `lr=0.5`.
6. Reset Adagrad state, run fixed32 Adagrad to step 110 with `lr=0.2`.
7. Validate the final theta with fixed128 E/Pi/Neumann iterations.

The counts-guided step uses:

```text
D = 2.0 * (duplications + 0.1) / (copies + 0.1)
L = 2.0 * (losses + 0.1) / (copies + 0.1)
T = 0.5 * (transfers + 0.1) / (copies + 0.1)
```

with a `1e-5` floor and the workflow bounds `min_rate=1e-10`,
`max_rate=100.0`.

## Validation Run

Output directory:

```text
/tmp/gpurec_hogenom_uniform_start_counts_route_verify
```

Summary:

```text
total_wall_s = 278.0164458240033
final fixed32 NLL = 526791.3125
final fixed128 NLL = 526791.3125
target NLL = 526799.625
status = accepted
```

Uniform start evidence:

```text
uniform_theta_init_rates = [0.05, 0.05, 0.05]
uniform_theta_max_abs_delta = 0.0
uniform_start.pt max abs delta from log2(0.05) = 0.0
initial uniform-start fixed16 NLL = 667284.25
counts-guided post-step fixed16 NLL = 554791.125
missing count species = 1
```

Stage timings:

```text
stage0 uniform start + counts-guided step  12.814677072979975 s
stage1 fixed16 Adagrad to step 40          88.90524461300811 s
stage2 reset fixed16 Adagrad to step 100   125.70018449000781 s
stage3 reset fixed32 Adagrad to step 110   50.56113163399277 s
```

The final fixed128 NLL is `1.6875` bits above the current reference best and
`8.3125` bits below the accepted `best + 10` threshold.

## Verification Commands

Compile check:

```bash
python -m py_compile scripts/benchmark_hogenom_counts_adagrad_route.py
```

Uniform checkpoint check:

```bash
python - <<'PY'
import pathlib
import torch

base = pathlib.Path("/tmp/gpurec_hogenom_uniform_start_counts_route_verify")
ckpt = torch.load(
    base / "stage0_uniform_start_counts_guided/checkpoints/uniform_start.pt",
    map_location="cpu",
)
theta = ckpt["theta"]
expected = torch.full_like(theta, torch.log2(torch.tensor(0.05, dtype=theta.dtype)))
print(float((theta - expected).abs().max()))
PY
```

Expected output:

```text
0.0
```
