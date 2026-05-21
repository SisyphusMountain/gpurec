"""Shared helpers for checkout-local fixed-dataset HOGENOM launchers.

This module exists only while the legacy specieswise/global uniform experiment
scripts remain.  Migrate reusable optimizer schedules, regularizers, and output
schemas into ``gpurec.workflow`` before promoting or deleting those launchers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gpurec import GeneReconModel


LN2 = math.log(2.0)
RATE_QUANTILES = torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0])
RATE_COLUMNS = (("D", 0), ("T", 2), ("L", 1))


@dataclass(frozen=True)
class DatasetConfig:
    species_tree: Path
    families_file: Path
    preprocess_cache: Path
    out_dir: Path
    device: str
    dtype: torch.dtype
    max_families: int | None
    family_chunk_size: int
    fixed_iters_E: int
    fixed_iters_Pi: int
    neumann_terms: int
    use_pruning: bool
    initial_rates: tuple[float, float, float]
    min_rate: float
    max_rate: float


@dataclass(frozen=True)
class RegularizationConfig:
    kind: str
    weight: float
    theta_center: float
    theta_std: float
    beta_ps_alpha: float
    beta_ps_beta: float


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def uniform_origination_probs(
    n_species: int,
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs_cpu = torch.full((n_species,), 1.0 / n_species, dtype=dtype)
    return probs_cpu.to(device=device, dtype=dtype), probs_cpu


def build_model(
    config: DatasetConfig,
) -> GeneReconModel:
    return GeneReconModel.from_alerax_families(
        str(config.species_tree),
        config.families_file,
        mode="specieswise",
        start=0,
        max_families=config.max_families,
        device=config.device,
        dtype=config.dtype,
        theta_init_rates=config.initial_rates,
        preprocess_cache_dir=config.preprocess_cache,
        fixed_iters_E=config.fixed_iters_E,
        fixed_iters_Pi=config.fixed_iters_Pi,
        neumann_terms=config.neumann_terms,
        use_pruning=config.use_pruning,
        family_chunk_size=config.family_chunk_size,
        lazy_preprocess=True,
        prefetch_batches="all",
    )


def _theta_natural_logits(theta: torch.Tensor) -> torch.Tensor:
    zeros = theta.new_zeros((*theta.shape[:-1], 1))
    return torch.cat((zeros, theta), dim=-1) * LN2


def pS_values(theta: torch.Tensor) -> torch.Tensor:
    return torch.softmax(_theta_natural_logits(theta), dim=-1)[..., 0]


def regularization_penalty(
    theta: torch.Tensor,
    config: RegularizationConfig,
) -> torch.Tensor:
    if config.kind == "none" or config.weight == 0.0:
        return theta.new_zeros(())

    if config.kind == "square-theta":
        shifted = theta - theta.new_tensor(config.theta_center)
        penalty = shifted.square().sum()
    elif config.kind == "gaussian-theta":
        shifted = (theta - theta.new_tensor(config.theta_center)) / config.theta_std
        penalty = 0.5 * shifted.square().sum()
    elif config.kind == "beta-ps":
        logits = _theta_natural_logits(theta)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_pS = log_probs[..., 0]
        log_one_minus_pS = torch.logsumexp(logits[..., 1:], dim=-1) - torch.logsumexp(
            logits,
            dim=-1,
        )
        alpha = config.beta_ps_alpha
        beta = config.beta_ps_beta
        penalty = -((alpha - 1.0) * log_pS + (beta - 1.0) * log_one_minus_pS).sum()
        penalty = penalty / LN2
    else:
        raise ValueError(f"unknown regularization: {config.kind}")

    return penalty * config.weight


def full_objective(
    model: GeneReconModel,
    regularization: RegularizationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data_loss = model.full_loss()
    penalty = regularization_penalty(model.theta, regularization)
    return data_loss, penalty, data_loss + penalty


def active_batch_objective(
    model: GeneReconModel,
    regularization: RegularizationConfig,
    *,
    penalty_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data_loss = model()
    penalty = regularization_penalty(model.theta, regularization) * penalty_scale
    return data_loss, penalty, data_loss + penalty


def zero_grad(optimizer: torch.optim.Optimizer | None, model: GeneReconModel) -> None:
    if optimizer is None:
        model.theta.grad = None
    else:
        optimizer.zero_grad(set_to_none=True)


def evaluate_full(
    model: GeneReconModel,
    regularization: RegularizationConfig,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    zero_grad(optimizer, model)
    data_loss, penalty, objective = full_objective(model, regularization)
    objective.backward()
    synchronize()
    if model.theta.grad is None:
        raise RuntimeError("missing theta gradient")
    grad = model.theta.grad.detach()
    return {
        "data_nll_bits": float(data_loss.detach().cpu()),
        "regularization_penalty_bits": float(penalty.detach().cpu()),
        "objective_bits": float(objective.detach().cpu()),
        "log_likelihood_bits": float(-data_loss.detach().cpu()),
        "grad_inf": float(grad.abs().amax().cpu()),
        "grad_norm": float(torch.linalg.vector_norm(grad).cpu()),
    }


def make_first_order_optimizer(
    name: str,
    model: GeneReconModel,
    lr: float,
) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD([model.theta], lr=lr)
    if name == "adagrad":
        return torch.optim.Adagrad([model.theta], lr=lr)
    if name == "adam":
        return torch.optim.Adam([model.theta], lr=lr)
    if name == "adamw":
        return torch.optim.AdamW([model.theta], lr=lr)
    raise ValueError(f"not a first-order optimizer: {name}")


def activate_batch(model: GeneReconModel, batch_idx: int) -> None:
    model.select_batch(batch_idx)


def rate_distribution_rows(
    theta: torch.Tensor,
    *,
    iteration: int,
    phase: str,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    rates = torch.exp(theta.detach() * LN2).cpu()
    for rate_name, column in RATE_COLUMNS:
        values = rates[:, column]
        quantiles = torch.quantile(values, RATE_QUANTILES.to(values))
        rows.append(
            {
                "phase": phase,
                "iteration": iteration,
                "quantity": rate_name,
                "min": float(quantiles[0]),
                "p05": float(quantiles[1]),
                "median": float(quantiles[2]),
                "p95": float(quantiles[3]),
                "max": float(quantiles[4]),
                "mean": float(values.mean()),
                "std": float(values.std(unbiased=False)),
            }
        )

    pS = pS_values(theta.detach()).cpu()
    quantiles = torch.quantile(pS, RATE_QUANTILES.to(pS))
    rows.append(
        {
            "phase": phase,
            "iteration": iteration,
            "quantity": "pS",
            "min": float(quantiles[0]),
            "p05": float(quantiles[1]),
            "median": float(quantiles[2]),
            "p95": float(quantiles[3]),
            "max": float(quantiles[4]),
            "mean": float(pS.mean()),
            "std": float(pS.std(unbiased=False)),
        }
    )
    return rows


def format_distribution(rows: list[dict[str, float | int | str]]) -> str:
    return " ".join(
        f"{row['quantity']}[min={row['min']:.3g} p05={row['p05']:.3g} "
        f"med={row['median']:.3g} p95={row['p95']:.3g} max={row['max']:.3g}]"
        for row in rows
    )


def record_distribution(
    model: GeneReconModel,
    *,
    iteration: int,
    phase: str,
    distribution_history: list[dict[str, float | int | str]],
) -> str:
    rows = rate_distribution_rows(model.theta, iteration=iteration, phase=phase)
    distribution_history.extend(rows)
    return format_distribution(rows)


def log_step(row: dict[str, Any], distribution: str) -> None:
    print(
        f"phase={row['phase']} iter={row['iteration']:04d} "
        f"data_nll_bits={row['data_nll_bits']:.6f} "
        f"regularization_penalty_bits={row['regularization_penalty_bits']:.6f} "
        f"objective_bits={row['objective_bits']:.6f} "
        f"grad_inf={row['grad_inf']:.6g} grad_norm={row['grad_norm']:.6g} "
        f"theta_step_inf={row['theta_step_inf']:.3g} "
        f"iteration_s={row['iteration_s']:.3f}",
        flush=True,
    )
    print("  distributions " + distribution, flush=True)


def run_first_order_steps(
    *,
    model: GeneReconModel,
    regularization: RegularizationConfig,
    optimizer_name: str,
    steps: int,
    lr: float,
    phase: str,
    dataset_config: DatasetConfig,
    history: list[dict[str, Any]],
    distribution_history: list[dict[str, float | int | str]],
    start_iteration: int,
) -> int:
    optimizer = make_first_order_optimizer(optimizer_name, model, lr)
    iteration = start_iteration
    for local_step in range(steps):
        started = time.perf_counter()
        theta_before = model.theta.detach().clone()
        metrics = evaluate_full(model, regularization, optimizer)
        optimizer.step()
        synchronize()
        model.clamp_theta_(
            min_rate=dataset_config.min_rate,
            max_rate=dataset_config.max_rate,
        )
        theta_step = float((model.theta.detach() - theta_before).abs().amax().cpu())
        model.clear()
        row = {
            "phase": phase,
            "iteration": iteration,
            "phase_iteration": local_step,
            "lr": lr,
            "theta_step_inf": theta_step,
            "iteration_s": time.perf_counter() - started,
            **metrics,
        }
        history.append(row)
        distribution = record_distribution(
            model,
            iteration=iteration,
            phase=phase,
            distribution_history=distribution_history,
        )
        log_step(row, distribution)
        iteration += 1
    return iteration


def run_minibatch_epochs(
    *,
    model: GeneReconModel,
    regularization: RegularizationConfig,
    optimizer_name: str,
    epochs: int,
    lr: float,
    dataset_config: DatasetConfig,
    history: list[dict[str, Any]],
    distribution_history: list[dict[str, float | int | str]],
    start_iteration: int,
) -> int:
    optimizer = make_first_order_optimizer(optimizer_name, model, lr)
    iteration = start_iteration
    num_batches = len(model.batch_metadata)
    penalty_scale = 1.0 / num_batches
    print(
        f"minibatch_start optimizer={optimizer_name} epochs={epochs} "
        f"lr={lr} batches={num_batches} penalty_scale={penalty_scale:.6g}",
        flush=True,
    )
    for epoch in range(epochs):
        for batch_idx in range(num_batches):
            activate_batch(model, batch_idx)
            meta = model.current_batch_metadata
            started = time.perf_counter()
            theta_before = model.theta.detach().clone()

            optimizer.zero_grad(set_to_none=True)
            data_loss, penalty, objective = active_batch_objective(
                model,
                regularization,
                penalty_scale=penalty_scale,
            )
            objective.backward()
            synchronize()
            if model.theta.grad is None:
                raise RuntimeError("missing theta gradient")
            grad = model.theta.grad.detach()
            optimizer.step()
            synchronize()
            model.clamp_theta_(
                min_rate=dataset_config.min_rate,
                max_rate=dataset_config.max_rate,
            )
            theta_step = float((model.theta.detach() - theta_before).abs().amax().cpu())
            model.clear()

            row = {
                "phase": f"minibatch-{optimizer_name}",
                "iteration": iteration,
                "phase_iteration": epoch,
                "epoch": epoch,
                "batch_index": batch_idx,
                "batch_family_count": int(meta.family_count),
                "batch_clade_count": int(meta.clade_count),
                "lr": lr,
                "data_nll_bits": float(data_loss.detach().cpu()),
                "regularization_penalty_bits": float(penalty.detach().cpu()),
                "objective_bits": float(objective.detach().cpu()),
                "log_likelihood_bits": float(-data_loss.detach().cpu()),
                "grad_inf": float(grad.abs().amax().cpu()),
                "grad_norm": float(torch.linalg.vector_norm(grad).cpu()),
                "theta_step_inf": theta_step,
                "iteration_s": time.perf_counter() - started,
            }
            history.append(row)
            distribution = record_distribution(
                model,
                iteration=iteration,
                phase=row["phase"],
                distribution_history=distribution_history,
            )
            log_step(row, distribution)
            iteration += 1
    activate_batch(model, 0)
    return iteration


def run_lbfgs_steps(
    *,
    model: GeneReconModel,
    regularization: RegularizationConfig,
    steps: int,
    lr: float,
    dataset_config: DatasetConfig,
    history: list[dict[str, Any]],
    distribution_history: list[dict[str, float | int | str]],
    start_iteration: int,
) -> int:
    optimizer = torch.optim.LBFGS(
        [model.theta],
        lr=lr,
        max_iter=1,
        max_eval=20,
        history_size=10,
        line_search_fn="strong_wolfe",
    )
    iteration = start_iteration
    for local_step in range(steps):
        started = time.perf_counter()
        theta_before = model.theta.detach().clone()
        closure_calls = 0

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            closure_calls += 1
            optimizer.zero_grad(set_to_none=True)
            _, _, objective = full_objective(model, regularization)
            objective.backward()
            synchronize()
            return objective

        optimizer.step(closure)
        synchronize()
        model.clamp_theta_(
            min_rate=dataset_config.min_rate,
            max_rate=dataset_config.max_rate,
        )
        theta_step = float((model.theta.detach() - theta_before).abs().amax().cpu())
        metrics = evaluate_full(model, regularization, optimizer)
        model.clear()

        row = {
            "phase": "lbfgs",
            "iteration": iteration,
            "phase_iteration": local_step,
            "lr": lr,
            "closure_evals": closure_calls,
            "theta_step_inf": theta_step,
            "iteration_s": time.perf_counter() - started,
            **metrics,
        }
        history.append(row)
        distribution = record_distribution(
            model,
            iteration=iteration,
            phase="lbfgs",
            distribution_history=distribution_history,
        )
        log_step(row, distribution)
        iteration += 1
    return iteration


def run_training(
    *,
    model: GeneReconModel,
    optimizer_name: str,
    steps: tuple[int, ...],
    lrs: tuple[float, ...],
    regularization: RegularizationConfig,
    dataset_config: DatasetConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, float | int | str]]]:
    history: list[dict[str, Any]] = []
    distribution_history: list[dict[str, float | int | str]] = []
    iteration = 0

    if optimizer_name in ("sgd", "adagrad", "adam", "adamw"):
        iteration = run_first_order_steps(
            model=model,
            regularization=regularization,
            optimizer_name=optimizer_name,
            steps=steps[0],
            lr=lrs[0],
            phase=optimizer_name,
            dataset_config=dataset_config,
            history=history,
            distribution_history=distribution_history,
            start_iteration=iteration,
        )
    elif optimizer_name.startswith("minibatch-") and not optimizer_name.endswith("-lbfgs"):
        base = optimizer_name.removeprefix("minibatch-")
        iteration = run_minibatch_epochs(
            model=model,
            regularization=regularization,
            optimizer_name=base,
            epochs=steps[0],
            lr=lrs[0],
            dataset_config=dataset_config,
            history=history,
            distribution_history=distribution_history,
            start_iteration=iteration,
        )
    elif optimizer_name == "lbfgs":
        iteration = run_lbfgs_steps(
            model=model,
            regularization=regularization,
            steps=steps[0],
            lr=lrs[0],
            dataset_config=dataset_config,
            history=history,
            distribution_history=distribution_history,
            start_iteration=iteration,
        )
    elif optimizer_name == "adagrad-lbfgs":
        iteration = run_first_order_steps(
            model=model,
            regularization=regularization,
            optimizer_name="adagrad",
            steps=steps[0],
            lr=lrs[0],
            phase="adagrad",
            dataset_config=dataset_config,
            history=history,
            distribution_history=distribution_history,
            start_iteration=iteration,
        )
        iteration = run_lbfgs_steps(
            model=model,
            regularization=regularization,
            steps=steps[1],
            lr=lrs[1],
            dataset_config=dataset_config,
            history=history,
            distribution_history=distribution_history,
            start_iteration=iteration,
        )
    elif optimizer_name == "minibatch-adagrad-lbfgs":
        iteration = run_minibatch_epochs(
            model=model,
            regularization=regularization,
            optimizer_name="adagrad",
            epochs=steps[0],
            lr=lrs[0],
            dataset_config=dataset_config,
            history=history,
            distribution_history=distribution_history,
            start_iteration=iteration,
        )
        iteration = run_lbfgs_steps(
            model=model,
            regularization=regularization,
            steps=steps[1],
            lr=lrs[1],
            dataset_config=dataset_config,
            history=history,
            distribution_history=distribution_history,
            start_iteration=iteration,
        )
    else:
        raise ValueError(f"unknown optimizer: {optimizer_name}")

    return history, distribution_history


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def write_outputs(
    *,
    model: GeneReconModel,
    species_names: list[str],
    origination_probs_cpu: torch.Tensor,
    history: list[dict[str, Any]],
    distribution_history: list[dict[str, float | int | str]],
    dataset_config: DatasetConfig,
    regularization: RegularizationConfig,
    optimizer_name: str,
    steps: tuple[int, ...],
    lrs: tuple[float, ...],
) -> None:
    dataset_config.out_dir.mkdir(parents=True, exist_ok=True)

    history_path = dataset_config.out_dir / "specieswise_optimization_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    distribution_path = dataset_config.out_dir / "specieswise_parameter_history.csv"
    pd.DataFrame(distribution_history).to_csv(distribution_path, index=False)

    with torch.no_grad():
        theta = model.theta.detach().cpu().numpy()
        rates = torch.exp(model.theta.detach() * LN2).cpu().numpy()
        pS = pS_values(model.theta.detach()).cpu().numpy()

    rates_path = dataset_config.out_dir / "optimized_specieswise_rates.csv"
    pd.DataFrame(
        {
            "species_index": np.arange(model.n_species),
            "species": species_names,
            "duplication_theta_log2": theta[:, 0],
            "loss_theta_log2": theta[:, 1],
            "transfer_theta_log2": theta[:, 2],
            "duplication_rate": rates[:, 0],
            "loss_rate": rates[:, 1],
            "transfer_rate": rates[:, 2],
            "pS": pS,
            "origination_prob": origination_probs_cpu.numpy(),
        }
    ).to_csv(rates_path, index=False)

    origination_path = dataset_config.out_dir / "uniform_origination_distribution.csv"
    pd.DataFrame(
        {
            "species_index": np.arange(len(species_names)),
            "species": species_names,
            "origination_prob": origination_probs_cpu.numpy(),
        }
    ).to_csv(origination_path, index=False)

    config_path = dataset_config.out_dir / "run_config.json"
    config = {
        "dataset": _jsonable(asdict(dataset_config)),
        "regularization": _jsonable(asdict(regularization)),
        "optimizer": optimizer_name,
        "steps": list(steps),
        "lr": list(lrs),
    }
    config_path.write_text(json.dumps(config, indent=2))

    print("wrote", history_path)
    print("wrote", distribution_path)
    print("wrote", rates_path)
    print("wrote", origination_path)
    print("wrote", config_path)
