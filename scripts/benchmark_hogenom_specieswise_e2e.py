"""Summarize the checkout-local HOGENOM specieswise end-to-end benchmark route.

The route to the best current fixed128 HOGENOM likelihood was discovered across
multiple long-running local experiments.  This script turns those run
directories and candidate checkpoints into a reproducible timing table: it
extracts per-stage objectives, projected-gradient residuals, optimizer step
counts, and known wall time from workflow histories.

Candidate checkpoints created by manual pulse probes often do not contain
reliable probe wall time.  Their elapsed time is therefore reported as unknown
by default, so the route total is a lower bound unless every stage has a
workflow history or summary.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpurec.workflow.checkpoint import load_checkpoint  # noqa: E402


DEFAULT_ROUTE: tuple[tuple[str, str], ...] = (
    ("lbfgsb32_init_000_099", "/tmp/gpurec_hogenom_specieswise_lbfgsb_high32_from_init"),
    (
        "lbfgsb32_continue_100_219",
        "/tmp/gpurec_hogenom_specieswise_lbfgsb_high32_from_init_continue",
    ),
    (
        "lbfgsb32_continue_220_283",
        "/tmp/gpurec_hogenom_specieswise_lbfgsb_high32_from_init_continue2",
    ),
    (
        "lbfgsb32_continue_284_360",
        "/tmp/gpurec_hogenom_specieswise_lbfgsb_high32_from_init_continue3",
    ),
    (
        "lbfgsb32_continue_361_433",
        "/tmp/gpurec_hogenom_specieswise_lbfgsb_high32_from_init_continue4",
    ),
    (
        "lbfgsb64_polish_434_449",
        "/tmp/gpurec_hogenom_specieswise_high64_polish_from_frominit4",
    ),
    (
        "lbfgsb64_polish_450_469",
        "/tmp/gpurec_hogenom_specieswise_high64_polish_continue",
    ),
    (
        "lbfgsb64_competitive_470",
        "/tmp/gpurec_hogenom_specieswise_lbfgsb_high64_competitive_from_latest",
    ),
    (
        "lbfgsb64_lr0125_471",
        "/tmp/gpurec_hogenom_specieswise_lbfgsb_high64_clean_lr0125_h5_from_step470",
    ),
    (
        "projected_sgd64_repair_472_481",
        "/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr1e4_from_step471",
    ),
    (
        "projected_sgd64_repair_482_531",
        "/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr1e4_continue50",
    ),
    (
        "projected_sgd64_lr5e4_532_561",
        "/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr5e4_from_pg0719",
    ),
    (
        "projected_sgd64_lr5e4_562_591",
        "/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr5e4_reset_from_pg0695",
    ),
    (
        "projected_sgd64_lr5e3_592_602",
        "/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr5e3_from_pg0683",
    ),
    (
        "projected_sgd64_repair_594_643",
        "/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr1e4_repair_from_step594",
    ),
    (
        "projected_sgd64_lr1e3_644_656",
        "/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr1e3_from_pg0673",
    ),
    (
        "projected_sgd64_lr5e4_649_678",
        "/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr5e4_from_step649",
    ),
    (
        "topk_pulse_step679",
        "/tmp/gpurec_hogenom_specieswise_topk_probe_from_step679/candidate.pt",
    ),
    (
        "projected_sgd64_repair_679_728",
        "/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr1e4_repair_from_topk",
    ),
    (
        "topk_balanced_step729",
        "/tmp/gpurec_hogenom_specieswise_topk_probe_from_step729/candidate_balanced.pt",
    ),
    (
        "projected_sgd64_repair_729_778",
        "/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr1e4_repair_from_balanced729",
    ),
    (
        "topk_objective_step779",
        "/tmp/gpurec_hogenom_specieswise_topk_probe_from_step779/candidate_objective.pt",
    ),
    (
        "projected_sgd64_repair_779_828",
        "/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr1e4_repair_from_objective779",
    ),
    (
        "topk_objective_step829",
        "/tmp/gpurec_hogenom_specieswise_topk_probe_from_step829/candidate_objective.pt",
    ),
    (
        "projected_sgd64_repair_829_878",
        "/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr1e4_repair_from_objective829",
    ),
    (
        "topk_tradeoff_step879",
        "/tmp/gpurec_hogenom_specieswise_topk_probe_from_step879/candidate_tradeoff.pt",
    ),
    (
        "kkt_topk_step879",
        "/tmp/gpurec_hogenom_specieswise_kkt_probe_from_topk879/candidate_kkt.pt",
    ),
    (
        "frontier_top2_step879",
        "/tmp/gpurec_hogenom_specieswise_frontier_grad_probe_from_kkt879/candidate_frontier.pt",
    ),
    (
        "frontier_objective_step879",
        "/tmp/gpurec_hogenom_specieswise_frontier2_objective_candidate/candidate_objective.pt",
    ),
    (
        "greedy_objective_step879",
        "/tmp/gpurec_hogenom_specieswise_greedy_frontier_from_objective875/candidate_greedy_objective.pt",
    ),
    (
        "greedy_objective2_step879",
        "/tmp/gpurec_hogenom_specieswise_greedy_frontier2_from_objective875/candidate_greedy_objective.pt",
    ),
    (
        "coord3147_micro_step879",
        "/tmp/gpurec_hogenom_specieswise_coord3147_micro_from_objective875_cycle2/candidate_coord3147.pt",
    ),
    (
        "coord3141_micro_step879",
        "/tmp/gpurec_hogenom_specieswise_coord3141_micro_from_coord3147/candidate_coord3141.pt",
    ),
    (
        "fixed128_validation",
        "/tmp/gpurec_hogenom_specieswise_truecheck_coord3141_micro_e128",
    ),
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    path: Path


def _json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _history_rows(path: Path) -> list[dict[str, Any]]:
    history = path / "history.jsonl"
    if not history.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in history.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _last_metric_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in reversed(rows):
        if "likelihood/data_nll_bits" in row:
            return row
    return None


def _final_check_loss(row: dict[str, Any]) -> float | None:
    if row.get("optimizer/final_check_status") != "ok":
        return None
    return _float_or_none(row.get("optimizer/final_check_loss_bits"))


def _non_final_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("optimizer/phase") != "final_eval"
        and "likelihood/data_nll_bits" in row
    ]


def _apply_metric_row(
    stage_row: dict[str, Any],
    metric_row: dict[str, Any],
    *,
    include_validation_loss: bool,
    loss_source: str,
) -> None:
    configured_loss = _float_or_none(metric_row.get("likelihood/data_nll_bits"))
    validation_loss = _final_check_loss(metric_row) if include_validation_loss else None
    stage_row["configured_loss_bits"] = configured_loss
    stage_row["validation_loss_bits"] = validation_loss
    stage_row["loss_source"] = loss_source
    stage_row["loss_bits"] = (
        validation_loss if validation_loss is not None else configured_loss
    )
    stage_row["projected_grad_inf"] = _float_or_none(metric_row.get("grad/projected_inf"))
    stage_row["grad_inf"] = _float_or_none(metric_row.get("grad/inf"))
    if not include_validation_loss:
        stage_row["validation_projected_grad_inf"] = None
        stage_row["validated_iters"] = _validated_iteration_budget(
            stage_row.get("fixed_iters_e"),
            stage_row.get("fixed_iters_pi"),
        stage_row.get("neumann_terms"),
        )


def _inside_checkpoint_dir(checkpoint: Path, run_dir: Path) -> bool:
    try:
        checkpoint.resolve().relative_to((run_dir / "checkpoints").resolve())
    except (OSError, ValueError):
        return False
    return True


def _next_resume_checkpoint(
    current: StageSpec,
    next_row: dict[str, Any],
) -> Path | None:
    resume_from = next_row.get("resume_from")
    if not resume_from or not current.path.is_dir():
        return None
    checkpoint = Path(str(resume_from))
    if not checkpoint.is_file() or not _inside_checkpoint_dir(checkpoint, current.path):
        return None
    return checkpoint


def _elapsed_to_checkpoint_metric(
    run_dir: Path,
    checkpoint: Path,
) -> tuple[float, dict[str, Any], int | None] | None:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    target_step = _int_or_none(payload.get("step"))
    last_row = payload.get("last_row") if isinstance(payload.get("last_row"), dict) else {}
    target_phase = last_row.get("optimizer/phase")
    target_position = last_row.get("optimizer/eval_position")
    rows = _history_rows(run_dir)
    elapsed = 0.0
    for row in rows:
        if row.get("optimizer/phase") == "final_eval":
            continue
        elapsed += float(row.get("step_s", 0.0) or 0.0)
        if (
            target_step is not None
            and _int_or_none(row.get("step")) == target_step
            and "likelihood/data_nll_bits" in row
        ):
            if target_phase is not None and row.get("optimizer/phase") != target_phase:
                continue
            if (
                target_position is not None
                and row.get("optimizer/eval_position") != target_position
            ):
                continue
            return elapsed, row, target_step

    metric_rows = _non_final_metric_rows(rows)
    if last_row.get("optimizer/phase") == "final_eval" and metric_rows:
        elapsed = sum(
            float(row.get("step_s", 0.0) or 0.0)
            for row in rows
            if row.get("optimizer/phase") != "final_eval"
        )
        return elapsed, metric_rows[-1], target_step
    if "likelihood/data_nll_bits" in last_row:
        return elapsed, last_row, target_step
    return None


def _apply_effective_resume_elapsed(
    rows: list[dict[str, Any]],
    stages: list[StageSpec],
) -> None:
    for row in rows:
        row["original_known_elapsed_s"] = row.get("known_elapsed_s")

    for index, (row, stage) in enumerate(zip(rows, stages)):
        if row.get("artifact_type") != "run_dir" or index + 1 >= len(rows):
            continue
        checkpoint = _next_resume_checkpoint(stage, rows[index + 1])
        if checkpoint is None:
            continue
        result = _elapsed_to_checkpoint_metric(stage.path, checkpoint)
        if result is None:
            continue
        elapsed, metric_row, resume_step = result
        row["known_elapsed_s"] = elapsed
        row["elapsed_source"] = "history.step_s_to_next_resume_checkpoint"
        row["effective_elapsed_source"] = "history.step_s_to_next_resume_checkpoint"
        row["effective_resume_checkpoint"] = str(checkpoint)
        row["effective_resume_step"] = resume_step
        row["last_step"] = metric_row.get("step", row.get("last_step"))
        _apply_metric_row(
            row,
            metric_row,
            include_validation_loss=False,
            loss_source="history.next_resume_checkpoint_metric",
        )


def _pulse_validation_stage(spec: StageSpec) -> dict[str, Any]:
    summary = _json_or_none(spec.path / "summary.json") or {}
    rows = _history_rows(spec.path)
    if not rows:
        jsonl = spec.path / "pulse_benchmark.jsonl"
        if jsonl.is_file():
            rows = [
                json.loads(line)
                for line in jsonl.read_text(encoding="utf-8").splitlines()
                if line
            ]
    validate_row = next(
        (
            row
            for row in rows
            if row.get("stage") == "base_validate" and row.get("label") == "base"
        ),
        {},
    )
    checkpoint = summary.get("checkpoint")
    return {
        "name": spec.name,
        "path": str(spec.path),
        "artifact_type": "pulse_validation_dir",
        "exists": spec.path.exists(),
        "optimizer": "fixed_validation",
        "steps_configured": None,
        "resume_from": checkpoint,
        "fixed_iters_e": summary.get("validate_iters"),
        "fixed_iters_pi": summary.get("validate_iters"),
        "neumann_terms": summary.get("validate_iters"),
        "adaptive_iters": None,
        "lr": None,
        "lbfgs_lr": None,
        "lbfgs_history_size": None,
        "lbfgs_max_ls": None,
        "row_count": len(rows),
        "optimizer_row_count": 0,
        "first_step": None,
        "last_step": None,
        "known_elapsed_s": _float_or_none(validate_row.get("elapsed_s")),
        "original_known_elapsed_s": _float_or_none(validate_row.get("elapsed_s")),
        "elapsed_source": "pulse_benchmark.base_validate.elapsed_s",
        "effective_elapsed_source": None,
        "effective_resume_checkpoint": None,
        "effective_resume_step": None,
        "validated_iters": _int_or_none(summary.get("validate_iters")),
        "configured_loss_bits": _float_or_none(validate_row.get("loss_bits")),
        "validation_loss_bits": _float_or_none(validate_row.get("loss_bits")),
        "loss_source": "pulse_benchmark.base_validate.loss_bits",
        "loss_bits": _float_or_none(validate_row.get("loss_bits")),
        "projected_grad_inf": _float_or_none(validate_row.get("projected_grad_inf")),
        "validation_projected_grad_inf": _float_or_none(
            validate_row.get("projected_grad_inf")
        ),
        "grad_inf": _float_or_none(validate_row.get("grad_inf")),
        "status": "validated" if validate_row else None,
        "reason": "pulse_benchmark_base_validation",
    }


def _dir_stage(spec: StageSpec) -> dict[str, Any]:
    if (spec.path / "pulse_benchmark.jsonl").is_file():
        return _pulse_validation_stage(spec)
    summary = _json_or_none(spec.path / "summary.json") or {}
    config = _json_or_none(spec.path / "run_config.json") or {}
    rows = _history_rows(spec.path)
    metric_row = _last_metric_row(rows) or {}
    optimizer_rows = [
        row for row in rows if row.get("optimizer/phase") not in {None, "final_eval"}
    ]
    known_elapsed = _float_or_none(summary.get("elapsed_s"))
    elapsed_source = "summary.elapsed_s" if known_elapsed is not None else None
    if known_elapsed is None and rows:
        known_elapsed = sum(float(row.get("step_s", 0.0) or 0.0) for row in rows)
        elapsed_source = "history.step_s_sum"
    configured_loss = _float_or_none(summary.get("final_nll_bits"))
    if configured_loss is None:
        configured_loss = _float_or_none(metric_row.get("likelihood/data_nll_bits"))
    validation_loss = _final_check_loss(metric_row)
    loss = validation_loss if validation_loss is not None else configured_loss
    loss_source = (
        "history.optimizer/final_check_loss_bits"
        if validation_loss is not None
        else "summary.final_nll_bits"
    )
    projected = _float_or_none(summary.get("final_projected_grad_inf"))
    if projected is None:
        projected = _float_or_none(metric_row.get("grad/projected_inf"))
    config_validated_iters = _validated_iteration_budget(
        config.get("fixed_iters_e"),
        config.get("fixed_iters_pi"),
        config.get("neumann_terms"),
    )
    final_check_iters = _int_or_none(metric_row.get("optimizer/final_check_iters"))
    summary_validate_iters = _int_or_none(summary.get("validate_iters"))
    validated_iter_candidates = [
        value
        for value in (
            config_validated_iters,
            final_check_iters,
            summary_validate_iters,
        )
        if value is not None
    ]
    validated_iters = max(validated_iter_candidates, default=None)
    return {
        "name": spec.name,
        "path": str(spec.path),
        "artifact_type": "run_dir",
        "exists": spec.path.exists(),
        "optimizer": config.get("optimizer"),
        "steps_configured": config.get("steps"),
        "resume_from": config.get("resume_from"),
        "fixed_iters_e": config.get("fixed_iters_e"),
        "fixed_iters_pi": config.get("fixed_iters_pi"),
        "neumann_terms": config.get("neumann_terms"),
        "adaptive_iters": config.get("adaptive_iters"),
        "lr": config.get("lr"),
        "lbfgs_lr": config.get("lbfgs_lr"),
        "lbfgs_history_size": config.get("lbfgs_history_size"),
        "lbfgs_max_ls": config.get("lbfgs_max_ls"),
        "row_count": len(rows),
        "optimizer_row_count": len(optimizer_rows),
        "first_step": optimizer_rows[0].get("step") if optimizer_rows else None,
        "last_step": optimizer_rows[-1].get("step") if optimizer_rows else None,
        "known_elapsed_s": known_elapsed,
        "original_known_elapsed_s": known_elapsed,
        "elapsed_source": elapsed_source,
        "effective_elapsed_source": None,
        "effective_resume_checkpoint": None,
        "effective_resume_step": None,
        "validated_iters": validated_iters,
        "configured_loss_bits": configured_loss,
        "validation_loss_bits": validation_loss,
        "loss_source": loss_source,
        "loss_bits": loss,
        "projected_grad_inf": projected,
        "validation_projected_grad_inf": None,
        "grad_inf": _float_or_none(summary.get("final_grad_inf"))
        or _float_or_none(metric_row.get("grad/inf")),
        "status": summary.get("status"),
        "reason": summary.get("reason"),
    }


def _checkpoint_stage(spec: StageSpec, *, trust_checkpoint_elapsed: bool) -> dict[str, Any]:
    payload = load_checkpoint(spec.path, map_location="cpu")
    status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
    row = payload.get("last_row") if isinstance(payload.get("last_row"), dict) else {}
    known_elapsed = None
    elapsed_source = None
    if trust_checkpoint_elapsed:
        known_elapsed = _float_or_none(status.get("elapsed_s"))
        if known_elapsed is not None:
            elapsed_source = "checkpoint.status.elapsed_s"
    config = payload.get("config") or {}
    validated_iters = _validated_iteration_budget(
        config.get("fixed_iters_e"),
        config.get("fixed_iters_pi"),
        config.get("neumann_terms"),
    )
    return {
        "name": spec.name,
        "path": str(spec.path),
        "artifact_type": "checkpoint",
        "exists": spec.path.exists(),
        "optimizer": payload.get("optimizer_phase") or row.get("optimizer/phase"),
        "steps_configured": None,
        "resume_from": status.get("probe_base_checkpoint"),
        "fixed_iters_e": config.get("fixed_iters_e"),
        "fixed_iters_pi": config.get("fixed_iters_pi"),
        "neumann_terms": config.get("neumann_terms"),
        "adaptive_iters": config.get("adaptive_iters"),
        "lr": config.get("lr"),
        "lbfgs_lr": config.get("lbfgs_lr"),
        "lbfgs_history_size": config.get("lbfgs_history_size"),
        "lbfgs_max_ls": config.get("lbfgs_max_ls"),
        "row_count": 0,
        "optimizer_row_count": 0,
        "first_step": payload.get("step"),
        "last_step": payload.get("step"),
        "known_elapsed_s": known_elapsed,
        "original_known_elapsed_s": known_elapsed,
        "elapsed_source": elapsed_source,
        "effective_elapsed_source": None,
        "effective_resume_checkpoint": None,
        "effective_resume_step": None,
        "validated_iters": validated_iters,
        "configured_loss_bits": _float_or_none(row.get("likelihood/data_nll_bits"))
        or _float_or_none(status.get("best_nll_bits")),
        "validation_loss_bits": None,
        "loss_source": "checkpoint.last_row_or_status",
        "loss_bits": _float_or_none(row.get("likelihood/data_nll_bits"))
        or _float_or_none(status.get("best_nll_bits")),
        "projected_grad_inf": _float_or_none(row.get("grad/projected_inf"))
        or _float_or_none(status.get("probe_projected_grad_inf_fixed64")),
        "validation_projected_grad_inf": None,
        "grad_inf": _float_or_none(row.get("grad/inf")),
        "status": status.get("status"),
        "reason": status.get("reason"),
    }


def _missing_stage(spec: StageSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "path": str(spec.path),
        "artifact_type": "missing",
        "exists": False,
        "optimizer": None,
        "steps_configured": None,
        "resume_from": None,
        "fixed_iters_e": None,
        "fixed_iters_pi": None,
        "neumann_terms": None,
        "adaptive_iters": None,
        "lr": None,
        "lbfgs_lr": None,
        "lbfgs_history_size": None,
        "lbfgs_max_ls": None,
        "row_count": 0,
        "optimizer_row_count": 0,
        "first_step": None,
        "last_step": None,
        "known_elapsed_s": None,
        "original_known_elapsed_s": None,
        "elapsed_source": None,
        "effective_elapsed_source": None,
        "effective_resume_checkpoint": None,
        "effective_resume_step": None,
        "validated_iters": None,
        "configured_loss_bits": None,
        "validation_loss_bits": None,
        "loss_source": None,
        "loss_bits": None,
        "projected_grad_inf": None,
        "validation_projected_grad_inf": None,
        "grad_inf": None,
        "status": None,
        "reason": "missing",
    }


def _stage_row(spec: StageSpec, *, trust_checkpoint_elapsed: bool) -> dict[str, Any]:
    if spec.path.is_dir():
        return _dir_stage(spec)
    if spec.path.is_file():
        return _checkpoint_stage(spec, trust_checkpoint_elapsed=trust_checkpoint_elapsed)
    return _missing_stage(spec)


def _parse_stage(value: str) -> StageSpec:
    if "=" in value:
        name, path = value.split("=", 1)
        if not name:
            raise ValueError("stage name cannot be empty")
        return StageSpec(name=name, path=Path(path))
    path = Path(value)
    return StageSpec(name=path.name, path=path)


def _default_stages() -> list[StageSpec]:
    return [StageSpec(name=name, path=Path(path)) for name, path in DEFAULT_ROUTE]


def _replace_step879_tail_with_replay(
    stages: list[StageSpec],
    replay_dir: Path,
) -> list[StageSpec]:
    for index, stage in enumerate(stages):
        if stage.name == "topk_tradeoff_step879":
            return stages[:index] + [
                StageSpec(name="tail_replay_step879", path=replay_dir)
            ]
    raise ValueError(
        "cannot replace step-879 tail: stage topk_tradeoff_step879 was not found"
    )


def _replace_stage(stages: list[StageSpec], replacement: StageSpec) -> list[StageSpec]:
    replaced = False
    next_stages: list[StageSpec] = []
    for stage in stages:
        if stage.name == replacement.name:
            next_stages.append(replacement)
            replaced = True
        else:
            next_stages.append(stage)
    if not replaced:
        raise ValueError(f"cannot replace missing stage: {replacement.name}")
    return next_stages


def _truncate_after_stage(stages: list[StageSpec], name: str) -> list[StageSpec]:
    for index, stage in enumerate(stages):
        if stage.name == name:
            return stages[: index + 1]
    raise ValueError(f"cannot truncate after missing stage: {name}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _validated_iteration_budget(*values: Any) -> int | None:
    numeric: list[int] = []
    for value in values:
        if value is None:
            return None
        try:
            numeric.append(int(value))
        except (TypeError, ValueError):
            return None
    return min(numeric)


def _is_validated(row: dict[str, Any], *, target_validation_iters: int) -> bool:
    value = row.get("validated_iters")
    if value is None:
        return False
    return int(value) >= target_validation_iters


def _target_index(
    rows: list[dict[str, Any]],
    target_nll: float,
    *,
    target_validation_iters: int,
) -> int | None:
    for index, row in enumerate(rows):
        if not _is_validated(row, target_validation_iters=target_validation_iters):
            continue
        loss = row.get("loss_bits")
        if loss is not None and float(loss) <= target_nll:
            return index
    return None


def _summary(
    rows: list[dict[str, Any]],
    *,
    target_nll: float,
    target_validation_iters: int,
) -> dict[str, Any]:
    known_elapsed = [
        float(row["known_elapsed_s"])
        for row in rows
        if row.get("known_elapsed_s") is not None
    ]
    missing_elapsed = [
        row["name"]
        for row in rows
        if row.get("exists") and row.get("known_elapsed_s") is None
    ]
    best_row = min(
        (
            row
            for row in rows
            if row.get("configured_loss_bits") is not None
            or row.get("loss_bits") is not None
        ),
        key=lambda row: float(
            row["configured_loss_bits"]
            if row.get("configured_loss_bits") is not None
            else row["loss_bits"]
        ),
        default=None,
    )
    validated_rows = [
        row
        for row in rows
        if row.get("loss_bits") is not None
        and _is_validated(row, target_validation_iters=target_validation_iters)
    ]
    best_validated_row = min(
        validated_rows,
        key=lambda row: float(row["loss_bits"]),
        default=None,
    )
    final_row = rows[-1] if rows else None
    target_idx = _target_index(
        rows,
        target_nll,
        target_validation_iters=target_validation_iters,
    )
    if target_idx is None:
        known_elapsed_to_target = None
    else:
        known_elapsed_to_target = sum(
            float(row["known_elapsed_s"])
            for row in rows[: target_idx + 1]
            if row.get("known_elapsed_s") is not None
        )
    return {
        "stage_count": len(rows),
        "existing_stage_count": sum(1 for row in rows if row.get("exists")),
        "target_nll_bits": target_nll,
        "target_validation_iters": target_validation_iters,
        "target_stage": None if target_idx is None else rows[target_idx]["name"],
        "known_elapsed_s": sum(known_elapsed),
        "known_elapsed_h": sum(known_elapsed) / 3600.0,
        "known_elapsed_to_target_s": known_elapsed_to_target,
        "known_elapsed_to_target_h": (
            None if known_elapsed_to_target is None else known_elapsed_to_target / 3600.0
        ),
        "unknown_elapsed_stage_count": len(missing_elapsed),
        "unknown_elapsed_stages": missing_elapsed,
        "raw_best_stage": None if best_row is None else best_row["name"],
        "raw_best_nll_bits": None
        if best_row is None
        else (
            best_row["configured_loss_bits"]
            if best_row.get("configured_loss_bits") is not None
            else best_row["loss_bits"]
        ),
        "raw_best_projected_grad_inf": None
        if best_row is None
        else best_row.get("projected_grad_inf"),
        "validated_best_stage": None
        if best_validated_row is None
        else best_validated_row["name"],
        "validated_best_nll_bits": None
        if best_validated_row is None
        else best_validated_row["loss_bits"],
        "validated_best_projected_grad_inf": None
        if best_validated_row is None
        else best_validated_row.get("projected_grad_inf"),
        "final_stage": None if final_row is None else final_row["name"],
        "final_nll_bits": None if final_row is None else final_row.get("loss_bits"),
        "final_projected_grad_inf": None
        if final_row is None
        else final_row.get("projected_grad_inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        action="append",
        default=None,
        help="Stage as name=/path or /path. Defaults to the known best route.",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--target-nll-bits", type=float, default=526821.875)
    parser.add_argument("--target-validation-iters", type=int, default=128)
    parser.add_argument(
        "--trust-checkpoint-elapsed",
        action="store_true",
        help=(
            "Use checkpoint status.elapsed_s for manual candidate checkpoints. "
            "Off by default because several pulse checkpoints inherited parent run time."
        ),
    )
    parser.add_argument(
        "--tail-replay-dir",
        type=Path,
        help=(
            "Replace the manual post-step879 candidate stages with a measured "
            "exact-delta replay directory. This measures accepted delta "
            "application/evaluation time, not the original manual search cost."
        ),
    )
    parser.add_argument(
        "--replace-stage",
        action="append",
        default=None,
        help=(
            "Replace a route stage as existing_stage_name=/path. Useful for "
            "substituting measured exact-delta replay directories for manual "
            "candidate checkpoints."
        ),
    )
    parser.add_argument(
        "--truncate-after-stage",
        help="Drop every route stage after the named stage.",
    )
    parser.add_argument(
        "--append-stage",
        action="append",
        default=None,
        help="Append a route stage as name=/path or /path after replacements/truncation.",
    )
    parser.add_argument(
        "--effective-resume-elapsed",
        action="store_true",
        help=(
            "For run-directory stages, charge elapsed time only through the "
            "checkpoint consumed by the next stage. This omits intermediate "
            "final validation rows and abandoned post-best steps that do not "
            "produce the resumed checkpoint."
        ),
    )
    args = parser.parse_args()

    stages = (
        [_parse_stage(value) for value in args.stage]
        if args.stage is not None
        else _default_stages()
    )
    if args.replace_stage is not None:
        for value in args.replace_stage:
            stages = _replace_stage(stages, _parse_stage(value))
    if args.tail_replay_dir is not None:
        stages = _replace_step879_tail_with_replay(stages, args.tail_replay_dir)
    if args.truncate_after_stage is not None:
        stages = _truncate_after_stage(stages, args.truncate_after_stage)
    if args.append_stage is not None:
        stages.extend(_parse_stage(value) for value in args.append_stage)
    rows = [
        _stage_row(stage, trust_checkpoint_elapsed=args.trust_checkpoint_elapsed)
        for stage in stages
    ]
    if args.effective_resume_elapsed:
        _apply_effective_resume_elapsed(rows, stages)
    summary = _summary(
        rows,
        target_nll=float(args.target_nll_bits),
        target_validation_iters=int(args.target_validation_iters),
    )

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(args.out_dir / "hogenom_e2e_route.csv", rows)
        (args.out_dir / "hogenom_e2e_route.json").write_text(
            json.dumps(rows, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, allow_nan=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
