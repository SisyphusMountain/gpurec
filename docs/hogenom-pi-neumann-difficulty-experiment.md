# HOGENOM Pi/Neumann Difficulty Predictor Experiment

Date: 2026-05-24.

Status: experiment design and reporting template. This document describes the
measurement needed to decide whether solver convergence difficulty measured at
initial DTL rates can be used to rebatch HOGENOM gene families by optimization
difficulty. It is intentionally written as a reproducible protocol plus the
report that should be filled once the measurement has been run.

## Question

Can the number of Pi and Neumann iterations needed at the initial DTL rates
predict the number needed at the optimized genewise DTL rates?

If yes, the scheduler can use cheap initial convergence statistics as a
family-difficulty feature before optimization starts. That would be better than
batching only by static size features such as leaf count, clade count, split
count, or schedule depth.

## Measurement Overview

For each HOGENOM family:

1. Build a genewise resident model with one family per resident batch.
2. Configure adaptive Pi and adaptive Neumann iteration counts with a high cap.
3. Evaluate one full genewise loss plus gradient at initialization.
4. Record per-family forward and backward solver telemetry.
5. Load optimized `theta_final.pt` from a completed HOGENOM genewise run.
6. Repeat the same full genewise loss plus gradient at optimized theta.
7. Correlate initialization telemetry with optimized-theta telemetry.

One-family batches are important. Current solver stats are stored per resident
static state, not per family inside a multi-family batch. Setting
`family_chunk_size=1` makes each solver-stat record correspond to exactly one
family while still using the normal resident-batch model path.

## Inputs

Use the complete HOGENOM dataset:

- species tree: `tests/data/hogenom_bench/sp.nwk`
- families file: `tests/data/hogenom_bench/families.txt`

Use a completed genewise HOGENOM optimization directory whose theta was produced
from the same family file and same family order:

- optimized theta: `<run_dir>/theta_final.pt`
- optional metadata checks: `<run_dir>/run_config.json`, `<run_dir>/summary.json`

For current benchmark runs in this workspace, suitable candidate run
directories have been under `/tmp/gpurec_hogenom_*`. Prefer the fastest accepted
run with a good final-check status and a final NLL close to or better than the
current baseline.

## Solver Configuration

The goal is to measure iteration demand, not to optimize DTL rates during this
experiment.

Recommended settings:

- `mode="genewise"`
- `family_chunk_size=1`
- `clade_budget=500_000`
- `batch_packing="depth_first_fit"` so the model builds leaf, nonleaf, and
  depth schedule summaries. `family_chunk_size=1` still forces one family per
  resident batch, so the packing policy does not merge families.
- `fixed_iters_E=16` or the production full-stage E setting
- `fixed_iters_Pi=128` as a high adaptive Pi cap
- `neumann_terms=128` as a high adaptive Neumann cap
- `adaptive_iters=True`
- `adaptive_neumann_terms=True`
- `pi_max_diff_tol=1e-5`
- `gradient_change_tol=1e-4`
- `gradient_change_rtol=1e-4`
- `final_check_iters` is irrelevant because this is not an optimization run

Keep the convergence tolerances identical between the initialization and optimum
passes. If the predictor is sensitive to tolerances, run a second sweep with
tighter values rather than mixing tolerances inside one report.

## Telemetry To Record

From each one-family batch, record these identifiers and static features:

- `family_index`
- `family_name`
- `clade_count`
- `split_count`
- `leaf_count`
- `nonleaf_count`
- `schedule_depth`

Record these convergence features at initialization and at optimized theta:

- `E_iterations`
- `E_convergence_delta`
- `Pi_max_iterations`
- `Pi_wave_count`
- `Pi_converged_waves`
- `Pi_wave_iterations_max`
- `Pi_wave_iterations_mean`
- `Pi_wave_iterations_sum`
- `Pi_hit_cap`, true if any Pi wave used `Pi_max_iterations`
- `Neumann_terms`
- `Gradient_converged`
- `Gradient_convergence_delta`
- `Gradient_convergence_threshold`
- `E_adjoint_iterations`
- `E_adjoint_rel_res`
- `E_adjoint_success`
- per-family NLL
- gradient infinity norm for the family's D/L/T row

The most important predictor variables are expected to be
`Pi_wave_iterations_max`, `Pi_wave_iterations_mean`, `Pi_wave_iterations_sum`,
and `Neumann_terms`.

## Reference Measurement Script

Save this as a temporary script, for example
`/tmp/measure_hogenom_pi_neumann_difficulty.py`, and run it from the repo root.
It writes one CSV with one row per family.

```python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from gpurec.workflow.config import RunConfig
from gpurec.workflow.model_factory import build_alerax_workflow_model


def _clear_solver_state(model) -> None:
    for static in model.cached_static_states:
        static.warm_E = None
        static.last_solver_stats = None
    model.theta.grad = None


def _pi_summary(stats: dict[str, Any]) -> dict[str, float | bool]:
    values = [int(v) for v in stats.get("Pi_wave_iterations", []) or []]
    cap = int(stats.get("Pi_max_iterations", 0))
    if not values:
        return {
            "pi_wave_iterations_max": 0.0,
            "pi_wave_iterations_mean": 0.0,
            "pi_wave_iterations_sum": 0.0,
            "pi_hit_cap": False,
        }
    return {
        "pi_wave_iterations_max": float(max(values)),
        "pi_wave_iterations_mean": float(sum(values) / len(values)),
        "pi_wave_iterations_sum": float(sum(values)),
        "pi_hit_cap": bool(cap and max(values) >= cap),
    }


def _measure_state(model, label: str) -> dict[int, dict[str, Any]]:
    _clear_solver_state(model)
    loss_vec, grad = model.full_genewise_nll_and_grad(need_grad=True)
    if grad is None:
        raise RuntimeError("expected gradient from full_genewise_nll_and_grad")

    stats_by_batch = model.solver_stat_records()
    if len(stats_by_batch) != len(model.batch_metadata):
        raise RuntimeError(
            f"got {len(stats_by_batch)} solver-stat records for "
            f"{len(model.batch_metadata)} one-family batches"
        )

    out: dict[int, dict[str, Any]] = {}
    for metadata, stats in zip(model.batch_metadata, stats_by_batch):
        if metadata.family_count != 1:
            raise RuntimeError(
                "family_chunk_size=1 should produce one family per batch, "
                f"got {metadata.family_count}"
            )
        family_index = int(metadata.family_indices[0])
        row = {
            f"{label}_nll_bits": float(loss_vec[family_index].detach().cpu()),
            f"{label}_grad_inf": float(
                grad[family_index].detach().abs().amax().cpu()
            ),
            f"{label}_e_iterations": int(stats.get("E_iterations", 0)),
            f"{label}_e_convergence_delta": stats.get("E_convergence_delta"),
            f"{label}_pi_max_iterations": int(stats.get("Pi_max_iterations", 0)),
            f"{label}_pi_wave_count": int(stats.get("Pi_wave_count", 0)),
            f"{label}_pi_converged_waves": int(
                stats.get("Pi_converged_waves", 0)
            ),
            f"{label}_neumann_terms": int(stats.get("Neumann_terms", 0)),
            f"{label}_gradient_converged": bool(
                stats.get("Gradient_converged", False)
            ),
            f"{label}_gradient_convergence_delta": stats.get(
                "Gradient_convergence_delta"
            ),
            f"{label}_gradient_convergence_threshold": stats.get(
                "Gradient_convergence_threshold"
            ),
            f"{label}_e_adjoint_iterations": int(
                stats.get("E_adjoint_iterations", 0)
            ),
            f"{label}_e_adjoint_rel_res": stats.get("E_adjoint_rel_res"),
            f"{label}_e_adjoint_success": bool(
                stats.get("E_adjoint_success", True)
            ),
        }
        for key, value in _pi_summary(stats).items():
            row[f"{label}_{key}"] = value
        out[family_index] = row
    return out


def _finite_or_empty(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species-tree", required=True)
    parser.add_argument("--families-file", required=True)
    parser.add_argument("--theta-final", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--pi-cap", type=int, default=128)
    parser.add_argument("--neumann-cap", type=int, default=128)
    parser.add_argument("--fixed-iters-e", type=int, default=16)
    args = parser.parse_args()

    config = RunConfig(
        species_tree=Path(args.species_tree),
        families_file=Path(args.families_file),
        out_dir=Path("/tmp/hogenom_pi_neumann_difficulty_dummy_out"),
        mode="genewise",
        device=args.device,
        dtype=args.dtype,
        family_chunk_size=1,
        clade_budget=500_000,
        batch_packing="depth_first_fit",
        fixed_iters_e=args.fixed_iters_e,
        fixed_iters_pi=args.pi_cap,
        neumann_terms=args.neumann_cap,
        adaptive_iters=True,
        adaptive_neumann_terms=True,
        pi_max_diff_tol=1e-5,
        gradient_change_tol=1e-4,
        gradient_change_rtol=1e-4,
    )

    model = build_alerax_workflow_model(config, prefetch_batches=0)
    try:
        model.materialize_batches()
        model.configure_solver_iterations(
            fixed_iters_E=args.fixed_iters_e,
            fixed_iters_Pi=args.pi_cap,
            neumann_terms=args.neumann_cap,
            pi_max_diff_tol=1e-5,
            gradient_change_tol=1e-4,
            adaptive_neumann_terms=True,
        )

        schedule_stats = model._family_schedule_stats
        if schedule_stats is None:
            raise RuntimeError("missing family schedule stats")

        init_rows = _measure_state(model, "init")

        theta_final = torch.load(
            args.theta_final,
            map_location=model.theta.device,
            weights_only=False,
        )
        theta_final = theta_final.to(device=model.theta.device, dtype=model.theta.dtype)
        if tuple(theta_final.shape) != tuple(model.theta.shape):
            raise RuntimeError(
                f"theta shape mismatch: {tuple(theta_final.shape)} vs "
                f"{tuple(model.theta.shape)}"
            )
        with torch.no_grad():
            model.theta.copy_(theta_final)
        opt_rows = _measure_state(model, "opt")

        rows: list[dict[str, Any]] = []
        for idx, metadata in enumerate(model.batch_metadata):
            family_index = int(metadata.family_indices[0])
            rows.append(
                {
                    "family_index": family_index,
                    "family_name": metadata.family_names[0],
                    "clade_count": int(schedule_stats.clade_counts[family_index]),
                    "split_count": int(schedule_stats.split_counts[family_index]),
                    "leaf_count": (
                        None
                        if schedule_stats.leaf_counts is None
                        else int(schedule_stats.leaf_counts[family_index])
                    ),
                    "nonleaf_count": (
                        None
                        if schedule_stats.nonleaf_counts is None
                        else int(schedule_stats.nonleaf_counts[family_index])
                    ),
                    "schedule_depth": (
                        None
                        if schedule_stats.schedule_depths is None
                        else int(schedule_stats.schedule_depths[family_index])
                    ),
                    **init_rows[family_index],
                    **opt_rows[family_index],
                }
            )

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _finite_or_empty(v) for k, v in row.items()})

        metadata_path = out_path.with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps(
                {
                    "species_tree": args.species_tree,
                    "families_file": args.families_file,
                    "theta_final": args.theta_final,
                    "rows": len(rows),
                    "pi_cap": args.pi_cap,
                    "neumann_cap": args.neumann_cap,
                    "fixed_iters_e": args.fixed_iters_e,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        model.close()


if __name__ == "__main__":
    main()
```

Example command:

```bash
python /tmp/measure_hogenom_pi_neumann_difficulty.py \
  --species-tree tests/data/hogenom_bench/sp.nwk \
  --families-file tests/data/hogenom_bench/families.txt \
  --theta-final /tmp/gpurec_hogenom_warm4_loss002/theta_final.pt \
  --out /tmp/hogenom_pi_neumann_difficulty.csv
```

## Correlation And Reporting Script

Run this after producing the CSV. It prints Pearson and Spearman correlations
for the important initialization-versus-optimum pairs.

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PAIRS = [
    ("init_pi_wave_iterations_max", "opt_pi_wave_iterations_max"),
    ("init_pi_wave_iterations_mean", "opt_pi_wave_iterations_mean"),
    ("init_pi_wave_iterations_sum", "opt_pi_wave_iterations_sum"),
    ("init_neumann_terms", "opt_neumann_terms"),
    ("init_gradient_convergence_delta", "opt_gradient_convergence_delta"),
    ("init_grad_inf", "opt_grad_inf"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    rows = []
    for init_col, opt_col in PAIRS:
        usable = df[[init_col, opt_col]].dropna()
        rows.append(
            {
                "init_metric": init_col,
                "opt_metric": opt_col,
                "n": len(usable),
                "pearson": usable[init_col].corr(usable[opt_col], method="pearson"),
                "spearman": usable[init_col].corr(usable[opt_col], method="spearman"),
                "init_median": usable[init_col].median(),
                "opt_median": usable[opt_col].median(),
                "init_max": usable[init_col].max(),
                "opt_max": usable[opt_col].max(),
            }
        )

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
```

Example:

```bash
python /tmp/correlate_hogenom_pi_neumann_difficulty.py \
  /tmp/hogenom_pi_neumann_difficulty.csv \
  --out /tmp/hogenom_pi_neumann_difficulty_correlations.csv
```

## Report To Produce After Running

After running the scripts, report these items.

### Run Metadata

- HOGENOM family count:
- optimized-theta source:
- optimized-theta final NLL:
- solver caps:
- Pi tolerance:
- Neumann gradient tolerance:
- device and dtype:
- measurement wall time:

### Correlation Summary

| Metric pair | N | Pearson | Spearman | Interpretation |
| --- | ---: | ---: | ---: | --- |
| init Pi max vs opt Pi max | | | | |
| init Pi mean vs opt Pi mean | | | | |
| init Pi sum vs opt Pi sum | | | | |
| init Neumann terms vs opt Neumann terms | | | | |
| init gradient delta vs opt gradient delta | | | | |

Use Spearman as the primary scheduler signal because batching only needs a
stable ranking or binning of families by difficulty. Pearson is useful for
checking whether the relationship is approximately linear.

### Family Bins

Create quantile bins by initial difficulty:

- easy: bottom 25% of `init_pi_wave_iterations_sum`
- normal: 25% to 75%
- hard: top 25%
- capped: any family with `init_pi_hit_cap=True`

For each bin, report:

- family count
- total clades
- total splits
- median leaf count
- median optimized Pi iterations
- median optimized Neumann terms
- median optimizer rows if available from a later optimization run

### Decision Rule

Use this interpretation:

- If Spearman correlation for Pi metrics is at least about `0.7`, initial Pi
  convergence is a strong scheduler feature. Add it to batch planning.
- If Spearman is `0.4` to `0.7`, use it as a secondary feature after static
  size/depth.
- If Spearman is below `0.4`, initial Pi convergence alone is too noisy. Prefer
  adaptive rebatching based on observed optimizer traces.
- If Neumann correlation is weaker than Pi correlation, use initial Pi
  difficulty for forward-solver batching and keep Neumann scheduling conservative
  or batch-level.

## Scheduler Implication If Correlation Is Strong

If the experiment shows strong rank correlation, add an opt-in scheduler feature:

1. Before optimization, run a cheap diagnostic pass at initialization to compute
   per-family initial Pi/Neumann difficulty.
2. Add a difficulty score, for example:

   ```text
   difficulty = init_pi_wave_iterations_sum
              + alpha * init_neumann_terms
              + beta  * init_pi_hit_cap
   ```

3. Group families by difficulty bins.
4. Inside each bin, keep the existing clade/depth first-fit packing so GPU
   batches remain memory balanced.

This would separate small-but-stiff families from genuinely easy small
families, which current leaf-count bands cannot do.

## Scheduler Implication If Correlation Is Weak

If initialization difficulty does not predict optimized-theta difficulty, do
not use it as a primary static scheduler feature. Instead:

1. Keep current `depth_first_fit` static batching.
2. During optimization, track per-family NLL improvement, stable-loss counters,
   low-acceptance rows, and line-search activity.
3. Rebatch remaining families with `model.replan_resident_batches(...)` so
   observed-hard families are grouped together after one or two probe stages.

This is more complex, but it uses the actual optimization trajectory rather
than a proxy measured at initialization.
