"""Legacy checkout-local HOGENOM experiment launcher.

Prefer the installed ``gpurec optimize`` workflow for supported production
runs.  This script is retained for reproducing historical HOGENOM experiments
with experiment-specific optimizer, W&B, and plotting logic.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import json
import math
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpurec import GeneReconModel  # noqa: E402


HOGENOM_DIR = REPO / "tests" / "data" / "HOGENOM" / "hogenom"
FAMILIES_FILE = HOGENOM_DIR / "hogenom_families.local.txt"
ALERAX_OUTPUT = HOGENOM_DIR / "output_alerax_corrected"
INFERRED_SPECIES_TREE = ALERAX_OUTPUT / "species_trees" / "inferred_species_tree.newick"
SPECIES_TREE = (
    INFERRED_SPECIES_TREE
    if INFERRED_SPECIES_TREE.exists()
    else HOGENOM_DIR / "hogenom_S.tree"
)
OUT_DIR = HOGENOM_DIR / "output_gpurec_wandb_adam"

LN2 = math.log(2.0)
RATE_FIELDS = (("D", 0), ("T", 2), ("L", 1))
QUANTILE_PROBS = torch.tensor([0.0, 0.5, 0.95, 1.0])
STAGED_SOLVER_PHASES = (
    (50, 4, "fixed4"),
    (50, 8, "fixed8"),
)
DEFAULT_SOLVER_BUDGET_ITERS = 4


@dataclass
class TreeNode:
    name: str = ""
    length: float | None = None
    children: list["TreeNode"] | None = None

    def is_leaf(self) -> bool:
        return not self.children


@dataclass(frozen=True)
class RunConfig:
    species_tree: Path
    families_file: Path
    out_dir: Path
    device: str
    max_families: int | None
    family_chunk_size: int | str | None
    clade_budget: int | None
    max_wave_size: int
    max_iters_e: int
    max_iters_pi: int
    max_neumann_terms: int
    convergence_check_interval: int
    e_logsumexp_tol: float
    pi_max_diff_tol: float
    gradient_change_tol: float
    gradient_change_rtol: float
    solver_iteration_schedule: str
    solver_budget_initial_iters: int
    solver_budget_increment: int
    solver_budget_patience: int
    solver_budget_step_interval: int
    parameter_mode: str
    optimizer: str
    adam_warmup_steps: int
    steps: int
    lr: float
    lr_decay_every: int
    lr_decay_factor: float
    lbfgs_lr: float
    lbfgs_history_size: int
    lbfgs_max_iter: int
    lbfgs_max_eval: int | None
    lbfgs_tolerance_grad: float
    lbfgs_tolerance_change: float
    lbfgs_line_search: str
    min_rate: float
    max_rate: float
    grad_inf_tol: float
    loss_change_tol: float
    loss_patience: int
    best_likelihood_patience: int
    best_likelihood_min_delta: float
    beta_ps_alpha: float
    beta_ps_beta: float
    beta_prior_weight: float
    branchscale_prior_weight: float
    diagnostic_mode: str
    log_every: int
    plot_every: int
    checkpoint_every: int
    resume_from: Path | None
    wandb_project: str
    wandb_entity: str | None
    wandb_run_name: str | None
    wandb_mode: str


@dataclass
class BranchScaledParameters:
    shared_theta: torch.nn.Parameter
    branch_log_l: torch.nn.Parameter


@dataclass(frozen=True)
class SolverIterationSettings:
    phase: str
    pi_iters: int
    neumann_terms: int
    pi_max_diff_tol: float
    gradient_change_tol: float


@dataclass
class SolverBudgetState:
    pi_iters: int
    neumann_terms: int
    no_improvement_steps: int = 0
    steps_since_increase: int = 0


class WandbSink:
    def __init__(self, args: argparse.Namespace, config: RunConfig):
        self.enabled = args.wandb_mode != "disabled"
        self._wandb = None
        self._run = None
        if not self.enabled:
            return
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "wandb is not installed. Install the checkout extras with "
                "`pip install -e \".[hogenom,dev]\"`, install wandb directly, "
                "or run with `--wandb-mode disabled` for a local smoke test."
            ) from exc
        self._wandb = wandb
        self._run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=_jsonable(asdict(config)),
        )

    def image(self, path: Path):
        if not self.enabled or self._wandb is None:
            return None
        return self._wandb.Image(str(path))

    def log(self, payload: dict[str, Any], *, step: int) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.log(payload, step=step)

    def save_file(self, path: Path) -> None:
        if not self.enabled or self._wandb is None or not path.exists():
            return
        self._wandb.save(str(path), base_path=str(path.parent))

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_run_config_snapshot(config: RunConfig) -> Path:
    path = config.out_dir / "run_config.json"
    path.write_text(
        json.dumps(_jsonable(asdict(config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def internal_parameter_mode(config: RunConfig) -> str:
    if config.parameter_mode == "branchscaled":
        return "specieswise"
    if config.parameter_mode in {"uniform", "global"}:
        return "global"
    return config.parameter_mode


def final_rates_path(config: RunConfig) -> Path:
    if config.parameter_mode == "branchscaled":
        return config.out_dir / "branchscaled_node_rates_final.tsv"
    if internal_parameter_mode(config) == "global":
        return config.out_dir / "uniform_node_rates_final.tsv"
    return config.out_dir / "specieswise_node_rates_final.tsv"


def is_branchscaled(config: RunConfig) -> bool:
    return config.parameter_mode == "branchscaled"


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_newick(path: Path) -> TreeNode:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty Newick file: {path}")
    idx = 0

    def skip_ws() -> None:
        nonlocal idx
        while idx < len(text) and text[idx].isspace():
            idx += 1

    def read_label() -> str:
        nonlocal idx
        skip_ws()
        if idx < len(text) and text[idx] in "'\"":
            quote = text[idx]
            idx += 1
            start = idx
            while idx < len(text) and text[idx] != quote:
                idx += 1
            label = text[start:idx]
            if idx < len(text):
                idx += 1
            return label
        start = idx
        while idx < len(text) and text[idx] not in ":,();":
            idx += 1
        return text[start:idx].strip()

    def read_length() -> float | None:
        nonlocal idx
        skip_ws()
        if idx >= len(text) or text[idx] != ":":
            return None
        idx += 1
        start = idx
        while idx < len(text) and text[idx] not in ",();":
            idx += 1
        raw = text[start:idx].strip()
        return float(raw) if raw else None

    def parse_subtree() -> TreeNode:
        nonlocal idx
        skip_ws()
        if idx < len(text) and text[idx] == "(":
            idx += 1
            children: list[TreeNode] = []
            while True:
                children.append(parse_subtree())
                skip_ws()
                if idx < len(text) and text[idx] == ",":
                    idx += 1
                    continue
                if idx < len(text) and text[idx] == ")":
                    idx += 1
                    break
                raise ValueError(f"invalid Newick near offset {idx} in {path}")
            return TreeNode(name=read_label(), length=read_length(), children=children)
        return TreeNode(name=read_label(), length=read_length(), children=[])

    root = parse_subtree()
    skip_ws()
    if idx < len(text) and text[idx] == ";":
        idx += 1
    skip_ws()
    if idx != len(text):
        raise ValueError(f"trailing Newick content near offset {idx} in {path}")
    return root


def tree_layout(root: TreeNode) -> tuple[list[TreeNode], dict[int, tuple[float, float]], list[tuple[float, float, float, float]]]:
    nodes: list[TreeNode] = []
    positions: dict[int, tuple[float, float]] = {}
    edges: list[tuple[float, float, float, float]] = []
    leaf_y = 0

    def assign(node: TreeNode, x_parent: float) -> tuple[float, float]:
        nonlocal leaf_y
        branch = 0.0 if node.length is None and not nodes else (node.length or 1.0)
        x = x_parent + branch
        nodes.append(node)
        if node.children:
            child_positions = [assign(child, x) for child in node.children]
            y = sum(pos[1] for pos in child_positions) / len(child_positions)
            for child_x, child_y in child_positions:
                edges.append((x, y, child_x, child_y))
        else:
            y = float(leaf_y)
            leaf_y += 1
        positions[id(node)] = (x, y)
        return x, y

    assign(root, 0.0)
    return nodes, positions, edges


def species_labels(model: GeneReconModel) -> list[str]:
    names = model.static.species_helpers.get("names")
    if names is None:
        raise RuntimeError("species helper names are unavailable")
    return [x.decode() if isinstance(x, bytes) else str(x) for x in names]


def theta_logits(theta: torch.Tensor) -> torch.Tensor:
    theta2 = theta.reshape(-1, 3)
    zeros = theta2.new_zeros((theta2.shape[0], 1))
    return torch.cat((zeros, theta2), dim=1) * LN2


def pS_values(theta: torch.Tensor) -> torch.Tensor:
    return torch.softmax(theta_logits(theta), dim=1)[:, 0]


def beta_ps_prior_bits(theta: torch.Tensor, *, alpha: float, beta: float, weight: float) -> torch.Tensor:
    if weight == 0.0:
        return theta.new_zeros(())
    logits = theta_logits(theta)
    log_probs_bits = torch.log_softmax(logits, dim=1) / LN2
    penalty_terms = theta.new_zeros((theta.reshape(-1, 3).shape[0],))
    if alpha != 1.0:
        penalty_terms = penalty_terms + (alpha - 1.0) * log_probs_bits[:, 0]
    if beta != 1.0:
        log_not_pS = (
            torch.logsumexp(logits[:, 1:], dim=1) / LN2
            - torch.logsumexp(logits, dim=1) / LN2
        )
        penalty_terms = penalty_terms + (beta - 1.0) * log_not_pS
    penalty = -penalty_terms.sum()
    return penalty * weight


def make_branchscaled_parameters(model: GeneReconModel) -> BranchScaledParameters:
    base = model.theta.detach().reshape(-1, 3)[0].clone()
    branch_log_l = torch.zeros(
        (model.n_species,),
        dtype=model.theta.dtype,
        device=model.theta.device,
    )
    model.theta.requires_grad_(False)
    return BranchScaledParameters(
        shared_theta=torch.nn.Parameter(base),
        branch_log_l=torch.nn.Parameter(branch_log_l),
    )


def effective_theta(
    model: GeneReconModel,
    branch_params: BranchScaledParameters | None,
) -> torch.Tensor:
    if branch_params is None:
        return model.theta
    return branch_params.shared_theta.unsqueeze(0) + (
        branch_params.branch_log_l / LN2
    ).unsqueeze(1)


def branchscale_prior_bits(
    branch_params: BranchScaledParameters | None,
    *,
    weight: float,
) -> torch.Tensor:
    if branch_params is None:
        return torch.zeros((), dtype=torch.float32)
    if weight == 0.0:
        return branch_params.branch_log_l.new_zeros(())
    return weight * torch.abs(torch.exp(branch_params.branch_log_l) - 1.0).sum()


def trainable_parameters(
    model: GeneReconModel,
    branch_params: BranchScaledParameters | None,
) -> list[torch.nn.Parameter]:
    if branch_params is None:
        return [model.theta]
    return [branch_params.shared_theta, branch_params.branch_log_l]


def parameters_gradient_stats(params: list[torch.nn.Parameter]) -> dict[str, float]:
    grads = []
    for param in params:
        if param.grad is None:
            raise RuntimeError("missing trainable parameter gradient")
        grads.append(param.grad.detach().reshape(-1))
    return gradient_stats(torch.cat(grads))


def parameters_have_finite_grad(params: list[torch.nn.Parameter]) -> bool:
    return all(param.grad is not None and torch.isfinite(param.grad).all().item() for param in params)


def clamp_parameters_(
    config: RunConfig,
    model: GeneReconModel,
    branch_params: BranchScaledParameters | None,
) -> None:
    if branch_params is None:
        model.clamp_theta_(min_rate=config.min_rate, max_rate=config.max_rate)
        return
    if config.min_rate <= 0:
        raise ValueError("min_rate must be strictly positive")
    if config.max_rate is not None and config.max_rate < config.min_rate:
        raise ValueError("max_rate must be greater than or equal to min_rate")
    min_theta = math.log2(config.min_rate)
    max_theta = None if config.max_rate is None else math.log2(config.max_rate)
    with torch.no_grad():
        branch_params.shared_theta.clamp_(min=min_theta, max=max_theta)
        lower = ((min_theta - branch_params.shared_theta).amax() * LN2).item()
        upper = None
        if max_theta is not None:
            upper = ((max_theta - branch_params.shared_theta).amin() * LN2).item()
        branch_params.branch_log_l.clamp_(min=lower, max=upper)


def branchscale_stats(branch_params: BranchScaledParameters | None) -> dict[str, float]:
    if branch_params is None:
        return {}
    metrics: dict[str, float] = {}
    for stat, value in tensor_stats(torch.exp(branch_params.branch_log_l)).items():
        metrics[f"branchscale/l_{stat}"] = value
    for stat, value in tensor_stats(branch_params.branch_log_l).items():
        metrics[f"branchscale/log_l_{stat}"] = value
    shared_rates = torch.exp2(branch_params.shared_theta.detach())
    for name, column in RATE_FIELDS:
        metrics[f"shared_rates/{name}"] = float(shared_rates[column].cpu())
    return metrics


def initial_solver_budget_state(config: RunConfig) -> SolverBudgetState:
    initial = int(config.solver_budget_initial_iters)
    return SolverBudgetState(
        pi_iters=min(initial, int(config.max_iters_pi)),
        neumann_terms=min(initial, int(config.max_neumann_terms)),
    )


def restore_solver_budget_state(
    raw_state: dict[str, Any] | None,
    config: RunConfig,
) -> SolverBudgetState:
    if not raw_state:
        return initial_solver_budget_state(config)
    return SolverBudgetState(
        pi_iters=min(int(raw_state.get("pi_iters", config.solver_budget_initial_iters)), config.max_iters_pi),
        neumann_terms=min(
            int(raw_state.get("neumann_terms", config.solver_budget_initial_iters)),
            config.max_neumann_terms,
        ),
        no_improvement_steps=max(0, int(raw_state.get("no_improvement_steps", 0))),
        steps_since_increase=max(0, int(raw_state.get("steps_since_increase", 0))),
    )


def solver_iteration_settings(
    config: RunConfig,
    step: int,
    budget_state: SolverBudgetState | None = None,
) -> SolverIterationSettings:
    if config.solver_iteration_schedule == "staged":
        offset = 0
        for length, iterations, phase in STAGED_SOLVER_PHASES:
            if step < offset + length:
                return SolverIterationSettings(
                    phase=phase,
                    pi_iters=iterations,
                    neumann_terms=iterations,
                    pi_max_diff_tol=-1.0,
                    gradient_change_tol=-1.0,
                )
            offset += length
    if config.solver_iteration_schedule == "budget":
        if budget_state is None:
            raise RuntimeError("budget solver schedule requires SolverBudgetState")
        return SolverIterationSettings(
            phase=f"budget{budget_state.pi_iters}",
            pi_iters=budget_state.pi_iters,
            neumann_terms=budget_state.neumann_terms,
            pi_max_diff_tol=config.pi_max_diff_tol,
            gradient_change_tol=config.gradient_change_tol,
        )
    return SolverIterationSettings(
        phase="adaptive",
        pi_iters=config.max_iters_pi,
        neumann_terms=config.max_neumann_terms,
        pi_max_diff_tol=config.pi_max_diff_tol,
        gradient_change_tol=config.gradient_change_tol,
    )


def update_solver_budget_state(
    config: RunConfig,
    budget_state: SolverBudgetState,
    objective_delta: float | None,
) -> str:
    if config.solver_iteration_schedule != "budget":
        return "none"

    budget_state.steps_since_increase += 1
    if objective_delta is None:
        budget_state.no_improvement_steps = 0
    elif objective_delta > config.loss_change_tol:
        budget_state.no_improvement_steps = 0
    else:
        budget_state.no_improvement_steps += 1

    stall_trigger = (
        objective_delta is not None
        and budget_state.no_improvement_steps >= config.solver_budget_patience
    )
    interval_trigger = budget_state.steps_since_increase >= config.solver_budget_step_interval
    if not stall_trigger and not interval_trigger:
        return "none"

    old_pi = budget_state.pi_iters
    old_neumann = budget_state.neumann_terms
    budget_state.pi_iters = min(
        config.max_iters_pi,
        budget_state.pi_iters + config.solver_budget_increment,
    )
    budget_state.neumann_terms = min(
        config.max_neumann_terms,
        budget_state.neumann_terms + config.solver_budget_increment,
    )
    increased = budget_state.pi_iters != old_pi or budget_state.neumann_terms != old_neumann
    if not increased:
        return "none"

    budget_state.no_improvement_steps = 0
    budget_state.steps_since_increase = 0
    if stall_trigger and interval_trigger:
        return "stall_and_interval"
    if stall_trigger:
        return "stall"
    return "interval"


def materialize_batch_statics(model: GeneReconModel) -> None:
    model.materialize_batches()


def _model_statics(model: GeneReconModel) -> list[Any]:
    return model.cached_static_states


def apply_solver_iteration_settings(
    model: GeneReconModel,
    settings: SolverIterationSettings,
) -> None:
    model.configure_solver_iterations(
        fixed_iters_Pi=settings.pi_iters,
        neumann_terms=settings.neumann_terms,
        pi_max_diff_tol=settings.pi_max_diff_tol,
        gradient_change_tol=settings.gradient_change_tol,
    )


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    vals = values.detach().float().reshape(-1).cpu()
    quantiles = torch.quantile(vals, QUANTILE_PROBS.to(vals))
    return {
        "min": float(quantiles[0]),
        "mean": float(vals.mean()),
        "median": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "max": float(quantiles[3]),
    }


def rate_stats(theta: torch.Tensor) -> dict[str, float]:
    theta2 = theta.detach().reshape(-1, 3)
    rates = torch.exp2(theta2)
    metrics: dict[str, float] = {}
    for name, column in RATE_FIELDS:
        for stat, value in tensor_stats(rates[:, column]).items():
            metrics[f"rates/{name}_{stat}"] = value
    for stat, value in tensor_stats(pS_values(theta2)).items():
        metrics[f"rates/pS_{stat}"] = value
    return metrics


def gradient_stats(grad: torch.Tensor) -> dict[str, float]:
    g = grad.detach().float().reshape(-1)
    abs_g = g.abs()
    return {
        "grad/norm": float(torch.linalg.vector_norm(g).cpu()),
        "grad/inf": float(abs_g.amax().cpu()),
        "grad/min": float(g.amin().cpu()),
        "grad/max": float(g.amax().cpu()),
        "grad/abs_min": float(abs_g.amin().cpu()),
        "grad/abs_mean": float(abs_g.mean().cpu()),
        "grad/abs_median": float(torch.quantile(abs_g.cpu(), 0.5)),
    }


def scheduled_lr(config: RunConfig, step: int) -> float:
    if config.lr_decay_every == 0:
        return config.lr
    decay_count = step // config.lr_decay_every
    return config.lr * (config.lr_decay_factor ** decay_count)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def optimizer_phase(config: RunConfig, step: int) -> str:
    if config.optimizer == "adam-lbfgs":
        return "adam" if step < config.adam_warmup_steps else "lbfgs"
    return config.optimizer


def phase_lr(config: RunConfig, phase: str, step: int) -> float:
    if phase == "adam":
        return scheduled_lr(config, step)
    return config.lbfgs_lr


def build_optimizers(
    config: RunConfig,
    params: list[torch.nn.Parameter],
) -> dict[str, torch.optim.Optimizer]:
    optimizers: dict[str, torch.optim.Optimizer] = {}
    if config.optimizer in {"adam", "adam-lbfgs"}:
        optimizers["adam"] = torch.optim.Adam(params, lr=config.lr)
    if config.optimizer in {"lbfgs", "adam-lbfgs"}:
        line_search_fn = (
            None
            if config.lbfgs_line_search == "none"
            else config.lbfgs_line_search
        )
        optimizers["lbfgs"] = torch.optim.LBFGS(
            params,
            lr=config.lbfgs_lr,
            max_iter=config.lbfgs_max_iter,
            max_eval=config.lbfgs_max_eval,
            history_size=config.lbfgs_history_size,
            tolerance_grad=config.lbfgs_tolerance_grad,
            tolerance_change=config.lbfgs_tolerance_change,
            line_search_fn=line_search_fn,
        )
    return optimizers


def _statics_with_solver_stats(model: GeneReconModel) -> list[Any]:
    return [
        static
        for static in _model_statics(model)
        if static.last_solver_stats is not None
    ]


def pi_iteration_count(model: GeneReconModel) -> tuple[int, int]:
    statics = _statics_with_solver_stats(model)
    total = 0
    cap_total = 0
    for static in statics:
        stats = static.last_solver_stats
        wave_iterations = stats.get("Pi_wave_iterations") or []
        if wave_iterations:
            total += sum(int(value) for value in wave_iterations)
            cap_total += int(static.fixed_iters_Pi) * len(wave_iterations)
        else:
            wave_count = int(stats.get("Pi_wave_count", 1))
            total += int(stats.get("Pi_max_iterations", static.fixed_iters_Pi)) * wave_count
            cap_total += int(static.fixed_iters_Pi) * wave_count
    return total, cap_total


def neumann_iteration_count(model: GeneReconModel) -> int:
    terms = [
        int(static.last_solver_stats.get("Neumann_terms", static.neumann_terms))
        for static in _statics_with_solver_stats(model)
    ]
    return max(terms, default=0)


def build_model(config: RunConfig) -> GeneReconModel:
    device = torch.device(config.device)
    dtype = torch.float32
    species_count = count_species_nodes(config.species_tree)
    origination_probs = torch.full(
        (species_count,),
        1.0 / species_count,
        device=device,
        dtype=dtype,
    )
    return GeneReconModel.from_alerax_families(
        str(config.species_tree),
        config.families_file,
        mode=internal_parameter_mode(config),
        start=0,
        max_families=config.max_families,
        device=device,
        dtype=dtype,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_E=None,
        max_iters_E=config.max_iters_e,
        fixed_iters_Pi=config.max_iters_pi,
        neumann_terms=config.max_neumann_terms,
        family_chunk_size=config.family_chunk_size,
        clade_budget=config.clade_budget,
        batch_packing="depth_first_fit",
        max_wave_size=config.max_wave_size,
        lazy_preprocess=True,
        prefetch_batches="all",
        adaptive_iters=True,
        convergence_check_interval=config.convergence_check_interval,
        e_logsumexp_tol=config.e_logsumexp_tol,
        pi_max_diff_tol=config.pi_max_diff_tol,
        gradient_change_tol=config.gradient_change_tol,
        gradient_change_rtol=config.gradient_change_rtol,
        origination_probs=origination_probs,
    )


def count_species_nodes(species_tree: Path) -> int:
    root = parse_newick(species_tree)
    count = 0
    stack = [root]
    while stack:
        node = stack.pop()
        count += 1
        if node.children:
            stack.extend(node.children)
    return count


def plot_tree_rates(
    *,
    root: TreeNode,
    layout_cache,
    rate_by_label: dict[str, dict[str, float]],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    nodes, positions, edges = layout_cache
    fields = [("D", True), ("T", True), ("L", True), ("pS", False)]
    fig, axes = plt.subplots(1, len(fields), figsize=(18, 10), sharey=True)
    if len(fields) == 1:
        axes = [axes]

    for ax, (field, log_scale) in zip(axes, fields):
        for x0, y0, x1, y1 in edges:
            ax.plot([x0, x0], [y0, y1], color="0.82", lw=0.35, zorder=1)
            ax.plot([x0, x1], [y1, y1], color="0.82", lw=0.35, zorder=1)

        xs = []
        ys = []
        values = []
        for node in nodes:
            if node.name in rate_by_label:
                xs.append(positions[id(node)][0])
                ys.append(positions[id(node)][1])
                values.append(rate_by_label[node.name][field])
        if values:
            if log_scale:
                positive = [v for v in values if v > 0.0]
                norm = (
                    LogNorm(vmin=max(min(positive), 1e-12), vmax=max(positive))
                    if positive
                    else None
                )
            else:
                norm = Normalize(vmin=0.0, vmax=1.0)
            scatter = ax.scatter(xs, ys, c=values, s=7, cmap="viridis", norm=norm, zorder=2)
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(field)
        ax.set_xlabel("tree distance")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["left", "right", "top", "bottom"]].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def current_rate_by_label(
    model: GeneReconModel,
    labels: list[str],
    branch_params: BranchScaledParameters | None = None,
) -> dict[str, dict[str, float]]:
    theta = effective_theta(model, branch_params).detach().reshape(-1, 3).cpu()
    rates = torch.exp2(theta)
    pS = pS_values(theta).cpu()
    theta_rows = int(theta.shape[0])
    if theta_rows not in {1, len(labels)}:
        raise RuntimeError(f"species label count {len(labels)} is incompatible with theta rows {theta_rows}")
    return {
        label: {
            "D": float(rates[0 if theta_rows == 1 else row, 0]),
            "T": float(rates[0 if theta_rows == 1 else row, 2]),
            "L": float(rates[0 if theta_rows == 1 else row, 1]),
            "pS": float(pS[0 if theta_rows == 1 else row]),
        }
        for row, label in enumerate(labels)
    }


def write_rate_table(
    path: Path,
    model: GeneReconModel,
    labels: list[str],
    branch_params: BranchScaledParameters | None = None,
) -> None:
    rates = current_rate_by_label(model, labels, branch_params)
    theta = effective_theta(model, branch_params).detach().reshape(-1, 3).cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        columns = ["row", "label", "D", "T", "L", "pS", "theta_D", "theta_T", "theta_L"]
        if branch_params is not None:
            columns.extend(
                [
                    "l_e",
                    "log_l_e",
                    "shared_D",
                    "shared_T",
                    "shared_L",
                    "shared_theta_D",
                    "shared_theta_T",
                    "shared_theta_L",
                ]
            )
            branch_l = torch.exp(branch_params.branch_log_l.detach()).cpu()
            branch_log_l = branch_params.branch_log_l.detach().cpu()
            shared_theta = branch_params.shared_theta.detach().cpu()
            shared_rates = torch.exp2(shared_theta)
        writer.writerow(columns)
        theta_rows = int(theta.shape[0])
        for row, label in enumerate(labels):
            theta_row = 0 if theta_rows == 1 else row
            vals = rates[label]
            output_row = [
                row,
                label,
                vals["D"],
                vals["T"],
                vals["L"],
                vals["pS"],
                float(theta[theta_row, 0]),
                float(theta[theta_row, 2]),
                float(theta[theta_row, 1]),
            ]
            if branch_params is not None:
                output_row.extend(
                    [
                        float(branch_l[row]),
                        float(branch_log_l[row]),
                        float(shared_rates[0]),
                        float(shared_rates[2]),
                        float(shared_rates[1]),
                        float(shared_theta[0]),
                        float(shared_theta[2]),
                        float(shared_theta[1]),
                    ]
                )
            writer.writerow(output_row)


class NonFiniteObjectiveOrGradient(RuntimeError):
    pass


def _move_optimizer_value_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_optimizer_value_to_device(val, device) for key, val in value.items()}
    if isinstance(value, list):
        return [_move_optimizer_value_to_device(val, device) for val in value]
    if isinstance(value, tuple):
        return tuple(_move_optimizer_value_to_device(val, device) for val in value)
    return value


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            state[key] = _move_optimizer_value_to_device(value, device)


def _load_optimizer_state(
    name: str,
    optimizer: torch.optim.Optimizer,
    optimizer_state: dict[str, Any],
    device: torch.device,
) -> None:
    try:
        optimizer.load_state_dict(optimizer_state)
    except ValueError as exc:
        print(f"warning: skipped incompatible {name} optimizer state: {exc}", file=sys.stderr)
        return
    _optimizer_state_to_device(optimizer, device)


def save_checkpoint(
    path: Path,
    *,
    model: GeneReconModel,
    branch_params: BranchScaledParameters | None,
    optimizers: dict[str, torch.optim.Optimizer],
    optimizer_phase: str,
    step: int,
    previous_objective: float | None,
    stable_loss_steps: int,
    solver_budget_state: SolverBudgetState | None,
    config: RunConfig,
    row: dict[str, Any] | None = None,
    best_data_nll: float | None = None,
    best_likelihood_step: int | None = None,
    best_likelihood_no_improvement_steps: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    theta = effective_theta(model, branch_params).detach().cpu()
    payload = {
        "step": int(step),
        "next_step": int(step) + 1,
        "theta": theta,
        "optimizer_phase": optimizer_phase,
        "optimizer_state": optimizers[optimizer_phase].state_dict(),
        "optimizer_states": {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        },
        "previous_objective": previous_objective,
        "stable_loss_steps": int(stable_loss_steps),
        "solver_budget_state": (
            asdict(solver_budget_state) if solver_budget_state is not None else None
        ),
        "config": _jsonable(asdict(config)),
        "last_row": row,
    }
    if best_data_nll is not None:
        payload["best_likelihood"] = {
            "data_nll_bits": float(best_data_nll),
            "step": None if best_likelihood_step is None else int(best_likelihood_step),
            "no_improvement_steps": (
                None
                if best_likelihood_no_improvement_steps is None
                else int(best_likelihood_no_improvement_steps)
            ),
        }
    if branch_params is not None:
        payload["branchscaled"] = {
            "shared_theta": branch_params.shared_theta.detach().cpu(),
            "branch_log_l": branch_params.branch_log_l.detach().cpu(),
        }
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: GeneReconModel,
    branch_params: BranchScaledParameters | None,
    optimizers: dict[str, torch.optim.Optimizer],
    config: RunConfig,
    device: torch.device,
) -> tuple[int, float | None, int, dict[str, Any] | None]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except (OSError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"could not safely load checkpoint {path}; regenerate the artifact "
            "or migrate it from a trusted source before retrying"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"checkpoint {path} must contain a dictionary payload")
    next_step = int(checkpoint.get("next_step", int(checkpoint.get("step", -1)) + 1))
    if branch_params is not None:
        branch_state = checkpoint.get("branchscaled")
        if not isinstance(branch_state, dict):
            raise RuntimeError(
                "checkpoint does not contain branchscaled parameters; "
                "cannot resume it with --mode branchscaled"
            )
        shared_theta = branch_state["shared_theta"].to(
            device=device,
            dtype=branch_params.shared_theta.dtype,
        )
        branch_log_l = branch_state["branch_log_l"].to(
            device=device,
            dtype=branch_params.branch_log_l.dtype,
        )
        if tuple(shared_theta.shape) != tuple(branch_params.shared_theta.shape):
            raise RuntimeError(
                f"checkpoint shared_theta shape {tuple(shared_theta.shape)} does not "
                f"match model shape {tuple(branch_params.shared_theta.shape)}"
            )
        if tuple(branch_log_l.shape) != tuple(branch_params.branch_log_l.shape):
            raise RuntimeError(
                f"checkpoint branch_log_l shape {tuple(branch_log_l.shape)} does not "
                f"match model shape {tuple(branch_params.branch_log_l.shape)}"
            )
        with torch.no_grad():
            branch_params.shared_theta.copy_(shared_theta)
            branch_params.branch_log_l.copy_(branch_log_l)
            model.theta.copy_(effective_theta(model, branch_params))
            for param in trainable_parameters(model, branch_params):
                param.grad = None
    else:
        theta = checkpoint["theta"].to(device=device, dtype=model.theta.dtype)
        if tuple(theta.shape) != tuple(model.theta.shape):
            raise RuntimeError(
                f"checkpoint theta shape {tuple(theta.shape)} does not match "
                f"{config.parameter_mode} model theta shape {tuple(model.theta.shape)}"
            )
        with torch.no_grad():
            model.theta.copy_(theta)
            model.theta.grad = None
    optimizer_states = checkpoint.get("optimizer_states")
    if isinstance(optimizer_states, dict):
        for name, optimizer_state in optimizer_states.items():
            if name in optimizers:
                _load_optimizer_state(name, optimizers[name], optimizer_state, device)
    elif "optimizer_state" in checkpoint:
        phase = checkpoint.get("optimizer_phase") or optimizer_phase(config, next_step)
        if phase in optimizers:
            _load_optimizer_state(phase, optimizers[phase], checkpoint["optimizer_state"], device)
    model.clear()
    return (
        next_step,
        checkpoint.get("previous_objective"),
        int(checkpoint.get("stable_loss_steps", 0)),
        checkpoint.get("solver_budget_state"),
    )


def best_likelihood_state_from_history(
    history_path: Path,
    *,
    start_step: int,
) -> tuple[float | None, int | None, int]:
    if not history_path.exists():
        return None, None, 0
    best_data_nll: float | None = None
    best_step: int | None = None
    last_seen_step: int | None = None
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row.get("step", -1))
        if step >= start_step:
            continue
        value = row.get("likelihood/data_nll_bits")
        if value is None:
            continue
        data_nll = float(value)
        last_seen_step = step
        if best_data_nll is None or data_nll < best_data_nll:
            best_data_nll = data_nll
            best_step = step
    if best_step is None or last_seen_step is None:
        return best_data_nll, best_step, 0
    return best_data_nll, best_step, max(0, last_seen_step - best_step)


def trim_history_for_resume(history_path: Path, *, start_step: int) -> None:
    if not history_path.exists():
        return
    kept: list[str] = []
    removed = 0
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("step", -1)) < start_step:
            kept.append(line)
        else:
            removed += 1
    if removed:
        history_path.write_text(
            "".join(f"{line}\n" for line in kept),
            encoding="utf-8",
        )
        print(
            f"trimmed {removed} history rows at/after resume step {start_step}",
            flush=True,
        )


def evaluate_and_backward(
    model: GeneReconModel,
    config: RunConfig,
    branch_params: BranchScaledParameters | None,
    params: list[torch.nn.Parameter],
) -> tuple[torch.Tensor, dict[str, float]]:
    for param in params:
        param.grad = None
    theta_eval = effective_theta(model, branch_params)
    data_nll = (
        model.full_loss_for_theta(theta_eval)
        if branch_params is not None
        else model.full_loss()
    )
    prior = beta_ps_prior_bits(
        theta_eval,
        alpha=config.beta_ps_alpha,
        beta=config.beta_ps_beta,
        weight=config.beta_prior_weight,
    )
    branch_prior = (
        branchscale_prior_bits(
            branch_params,
            weight=config.branchscale_prior_weight,
        )
        if branch_params is not None
        else data_nll.new_zeros(())
    )
    objective = data_nll + prior + branch_prior
    objective.backward()
    synchronize()
    for param in params:
        if param.grad is None:
            raise RuntimeError("missing trainable parameter gradient")
    pi_iters, pi_iter_cap = pi_iteration_count(model)
    neumann_terms = neumann_iteration_count(model)
    metrics = {
        "likelihood/data_nll_bits": float(data_nll.detach().cpu()),
        "likelihood/log_likelihood_bits": float((-data_nll).detach().cpu()),
        "objective/bits": float(objective.detach().cpu()),
        "regularization/beta_ps_bits": float(prior.detach().cpu()),
        "regularization/branchscale_bits": float(branch_prior.detach().cpu()),
        "solver/pi_iterations": float(pi_iters),
        "solver/pi_iteration_cap": float(pi_iter_cap),
        "solver/neumann_terms": float(neumann_terms),
    }
    metrics.update(parameters_gradient_stats(params))
    return objective, metrics


def evaluate_objective_components_no_backward(
    model: GeneReconModel,
    config: RunConfig,
    branch_params: BranchScaledParameters | None,
) -> dict[str, float]:
    with torch.no_grad():
        theta_eval = effective_theta(model, branch_params)
        data_nll = (
            model.full_loss_for_theta(theta_eval)
            if branch_params is not None
            else model.full_loss()
        )
        prior = beta_ps_prior_bits(
            theta_eval,
            alpha=config.beta_ps_alpha,
            beta=config.beta_ps_beta,
            weight=config.beta_prior_weight,
        )
        branch_prior = (
            branchscale_prior_bits(
                branch_params,
                weight=config.branchscale_prior_weight,
            )
            if branch_params is not None
            else data_nll.new_zeros(())
        )
        objective = data_nll + prior + branch_prior
    return {
        "likelihood/data_nll_bits": float(data_nll.detach().cpu()),
        "likelihood/log_likelihood_bits": float((-data_nll).detach().cpu()),
        "objective/bits": float(objective.detach().cpu()),
        "regularization/beta_ps_bits": float(prior.detach().cpu()),
        "regularization/branchscale_bits": float(branch_prior.detach().cpu()),
    }


def flatten_parameter_values(params: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([param.detach().reshape(-1) for param in params])


def flatten_parameter_grads(params: list[torch.nn.Parameter]) -> torch.Tensor:
    chunks = []
    for param in params:
        if param.grad is None:
            chunks.append(torch.zeros_like(param.detach()).reshape(-1))
        else:
            chunks.append(param.grad.detach().reshape(-1).clone())
    return torch.cat(chunks)


def _safe_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def step_diagnostic_metrics(
    *,
    pre_metrics: dict[str, float],
    post_metrics: dict[str, float],
    pre_grad: torch.Tensor,
    trainable_step: torch.Tensor,
    effective_theta_step: torch.Tensor,
) -> dict[str, float]:
    trainable_step = trainable_step.detach()
    effective_theta_step = effective_theta_step.detach()
    pre_grad = pre_grad.detach().to(
        device=trainable_step.device,
        dtype=trainable_step.dtype,
    )
    trainable_step_norm = torch.linalg.vector_norm(trainable_step)
    trainable_step_inf = trainable_step.abs().amax() if trainable_step.numel() else trainable_step.new_zeros(())
    effective_step_norm = torch.linalg.vector_norm(effective_theta_step)
    effective_step_inf = (
        effective_theta_step.abs().amax()
        if effective_theta_step.numel()
        else effective_theta_step.new_zeros(())
    )
    grad_dot_step = torch.dot(pre_grad, trainable_step)
    pre_grad_norm = torch.linalg.vector_norm(pre_grad)
    denom = pre_grad_norm * trainable_step_norm
    cos_grad_step = (
        grad_dot_step / denom
        if bool((denom > 0).item())
        else trainable_step.new_tensor(float("nan"))
    )
    pre_prior = (
        pre_metrics["regularization/beta_ps_bits"]
        + pre_metrics["regularization/branchscale_bits"]
    )
    post_prior = (
        post_metrics["regularization/beta_ps_bits"]
        + post_metrics["regularization/branchscale_bits"]
    )
    return {
        "diagnostics/pre_data_nll_bits": pre_metrics["likelihood/data_nll_bits"],
        "diagnostics/post_data_nll_bits": post_metrics["likelihood/data_nll_bits"],
        "diagnostics/data_nll_delta_bits": (
            pre_metrics["likelihood/data_nll_bits"]
            - post_metrics["likelihood/data_nll_bits"]
        ),
        "diagnostics/pre_objective_bits": pre_metrics["objective/bits"],
        "diagnostics/post_objective_bits": post_metrics["objective/bits"],
        "diagnostics/objective_delta_bits": (
            pre_metrics["objective/bits"] - post_metrics["objective/bits"]
        ),
        "diagnostics/pre_prior_bits": pre_prior,
        "diagnostics/post_prior_bits": post_prior,
        "diagnostics/prior_delta_bits": pre_prior - post_prior,
        "diagnostics/beta_ps_prior_delta_bits": (
            pre_metrics["regularization/beta_ps_bits"]
            - post_metrics["regularization/beta_ps_bits"]
        ),
        "diagnostics/branchscale_prior_delta_bits": (
            pre_metrics["regularization/branchscale_bits"]
            - post_metrics["regularization/branchscale_bits"]
        ),
        "diagnostics/trainable_step_norm": _safe_float(trainable_step_norm),
        "diagnostics/trainable_step_inf": _safe_float(trainable_step_inf),
        "diagnostics/effective_theta_step_norm": _safe_float(effective_step_norm),
        "diagnostics/effective_theta_step_inf": _safe_float(effective_step_inf),
        "diagnostics/pre_grad_norm": _safe_float(pre_grad_norm),
        "diagnostics/grad_dot_step": _safe_float(grad_dot_step),
        "diagnostics/predicted_objective_decrease_linearized_bits": _safe_float(-grad_dot_step),
        "diagnostics/cos_grad_step": _safe_float(cos_grad_step),
    }


def run(config: RunConfig, args: argparse.Namespace) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_from is None and args.timestamped_out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "latest_run.txt").write_text(
            str(config.out_dir) + "\n",
            encoding="utf-8",
        )
        latest_link = args.out_dir / "latest"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                if latest_link.is_symlink() or latest_link.is_file():
                    latest_link.unlink()
            if not latest_link.exists():
                latest_link.symlink_to(config.out_dir.resolve(), target_is_directory=True)
        except OSError as exc:
            print(f"warning: could not update latest run link: {exc}", file=sys.stderr)
    history_path = config.out_dir / "history.jsonl"
    if config.resume_from is None and history_path.exists():
        history_path.unlink()

    run_config_path = write_run_config_snapshot(config)
    wandb = WandbSink(args, config)
    wandb.save_file(run_config_path)
    hydra_config_path = config.out_dir / "hydra_config.yaml"
    if hydra_config_path.exists():
        wandb.save_file(hydra_config_path)
    root = parse_newick(config.species_tree)
    layout_cache = tree_layout(root)
    model = build_model(config)
    if config.solver_iteration_schedule == "staged":
        materialize_batch_statics(model)
    labels = species_labels(model)
    branch_params = make_branchscaled_parameters(model) if is_branchscaled(config) else None
    params = trainable_parameters(model, branch_params)
    optimizers = build_optimizers(config, params)
    solver_budget_state = initial_solver_budget_state(config)
    start_step = 0
    previous_objective: float | None = None
    stable_loss_steps = 0
    if config.resume_from is not None:
        start_step, previous_objective, stable_loss_steps, raw_solver_budget_state = load_checkpoint(
            config.resume_from,
            model=model,
            branch_params=branch_params,
            optimizers=optimizers,
            config=config,
            device=torch.device(config.device),
        )
        solver_budget_state = restore_solver_budget_state(raw_solver_budget_state, config)
        print(f"resumed checkpoint={config.resume_from} next_step={start_step}", flush=True)
        trim_history_for_resume(history_path, start_step=start_step)
    best_data_nll: float | None = None
    best_likelihood_step: int | None = None
    best_likelihood_no_improvement_steps = 0
    if config.best_likelihood_patience > 0:
        (
            best_data_nll,
            best_likelihood_step,
            best_likelihood_no_improvement_steps,
        ) = best_likelihood_state_from_history(history_path, start_step=start_step)

    build_info = {
        "dataset/families": sum(meta.family_count for meta in model.batch_metadata),
        "dataset/species": model.n_species,
        "dataset/batches": len(model.batch_metadata),
        "dataset/waves": sum(meta.wave_count for meta in model.batch_metadata),
        "model/parameter_mode": config.parameter_mode,
        "model/internal_parameter_mode": internal_parameter_mode(config),
        "optimizer/lbfgs_line_search": config.lbfgs_line_search,
        "outputs/run_dir": str(config.out_dir),
        "diagnostics/mode": config.diagnostic_mode,
        "solver/iteration_schedule": config.solver_iteration_schedule,
        "solver/budget_initial_iters": config.solver_budget_initial_iters,
        "solver/budget_increment": config.solver_budget_increment,
        "solver/budget_patience": config.solver_budget_patience,
        "solver/budget_step_interval": config.solver_budget_step_interval,
        "convergence/best_likelihood_patience": config.best_likelihood_patience,
        "convergence/best_likelihood_min_delta": config.best_likelihood_min_delta,
    }
    wandb.log(build_info, step=0)
    print(
        "built "
        f"families={int(build_info['dataset/families'])} "
        f"species={int(build_info['dataset/species'])} "
        f"batches={int(build_info['dataset/batches'])} "
        f"waves={int(build_info['dataset/waves'])} "
        f"mode={config.parameter_mode} "
        f"out_dir={config.out_dir}",
        flush=True,
    )

    stop_reason = "max_steps"
    last_step = start_step - 1
    started = time.perf_counter()
    try:
        for step in range(start_step, config.steps):
            step_t0 = time.perf_counter()
            solver_settings = solver_iteration_settings(config, step, solver_budget_state)
            apply_solver_iteration_settings(model, solver_settings)
            phase = optimizer_phase(config, step)
            optimizer = optimizers[phase]
            current_lr = phase_lr(config, phase, step)
            set_optimizer_lr(optimizer, current_lr)
            theta_before = effective_theta(model, branch_params).detach().clone()
            trainable_before = (
                flatten_parameter_values(params)
                if config.diagnostic_mode == "step"
                else None
            )
            pre_step_metrics: dict[str, float] | None = None
            pre_step_grad: torch.Tensor | None = None
            closure_evals = 0
            closure_s = 0.0
            objective: torch.Tensor | None = None
            metrics: dict[str, float] | None = None
            best_likelihood_improved = False
            best_likelihood_checked = False

            def closure() -> torch.Tensor:
                nonlocal closure_evals, closure_s, objective, metrics, pre_step_metrics, pre_step_grad
                optimizer.zero_grad(set_to_none=True)
                closure_t0 = time.perf_counter()
                objective_i, metrics_i = evaluate_and_backward(
                    model,
                    config,
                    branch_params,
                    params,
                )
                closure_s += time.perf_counter() - closure_t0
                closure_evals += 1
                if (
                    not torch.isfinite(objective_i).item()
                    or not parameters_have_finite_grad(params)
                ):
                    raise NonFiniteObjectiveOrGradient
                if config.diagnostic_mode == "step" and pre_step_metrics is None:
                    pre_step_metrics = dict(metrics_i)
                    pre_step_grad = flatten_parameter_grads(params)
                objective = objective_i
                metrics = metrics_i
                return objective_i

            def update_best_likelihood_checkpoint() -> None:
                nonlocal best_data_nll
                nonlocal best_likelihood_step
                nonlocal best_likelihood_no_improvement_steps
                nonlocal best_likelihood_improved
                nonlocal best_likelihood_checked
                if config.best_likelihood_patience <= 0:
                    return
                if metrics is None:
                    raise RuntimeError("best-likelihood check ran before metrics were available")
                data_nll = metrics["likelihood/data_nll_bits"]
                threshold = (
                    float("inf")
                    if best_data_nll is None
                    else best_data_nll - config.best_likelihood_min_delta
                )
                if data_nll < threshold:
                    best_data_nll = data_nll
                    best_likelihood_step = step
                    best_likelihood_no_improvement_steps = 0
                    best_likelihood_improved = True
                    checkpoint_dir = config.out_dir / "checkpoints"
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model=model,
                        branch_params=branch_params,
                        optimizers=optimizers,
                        optimizer_phase=phase,
                        step=step,
                        previous_objective=metrics["objective/bits"],
                        stable_loss_steps=stable_loss_steps,
                        solver_budget_state=solver_budget_state,
                        config=config,
                        row=None,
                        best_data_nll=best_data_nll,
                        best_likelihood_step=best_likelihood_step,
                        best_likelihood_no_improvement_steps=best_likelihood_no_improvement_steps,
                    )
                else:
                    best_likelihood_no_improvement_steps += 1
                best_likelihood_checked = True

            try:
                if phase == "lbfgs":
                    optimizer.step(closure)
                else:
                    objective = closure()
                    if objective is None or metrics is None:
                        raise RuntimeError("optimizer did not evaluate the objective")
                    update_best_likelihood_checkpoint()
                    optimizer.step()
            except NonFiniteObjectiveOrGradient:
                stop_reason = "nonfinite_objective_or_gradient"
                break
            if objective is None or metrics is None:
                raise RuntimeError("optimizer did not evaluate the objective")
            if not best_likelihood_checked:
                update_best_likelihood_checkpoint()
            synchronize()
            objective_bits = metrics["objective/bits"]
            delta = None if previous_objective is None else previous_objective - objective_bits
            if delta is not None and abs(delta) <= config.loss_change_tol:
                stable_loss_steps += 1
            else:
                stable_loss_steps = 0
            previous_objective = objective_bits
            solver_budget_increase_reason = update_solver_budget_state(
                config,
                solver_budget_state,
                delta,
            )

            clamp_parameters_(config, model, branch_params)
            theta_after = effective_theta(model, branch_params).detach()
            theta_step = float((theta_after - theta_before).abs().amax().cpu())
            diagnostic_metrics: dict[str, float] = {}
            if config.diagnostic_mode == "step":
                if pre_step_metrics is None or pre_step_grad is None or trainable_before is None:
                    raise RuntimeError("diagnostic mode did not capture the pre-step gradient")
                post_step_metrics = evaluate_objective_components_no_backward(
                    model,
                    config,
                    branch_params,
                )
                trainable_after = flatten_parameter_values(params)
                diagnostic_metrics = step_diagnostic_metrics(
                    pre_metrics=pre_step_metrics,
                    post_metrics=post_step_metrics,
                    pre_grad=pre_step_grad,
                    trainable_step=trainable_after - trainable_before,
                    effective_theta_step=(theta_after - theta_before).reshape(-1),
                )
            model.clear()
            metrics.update(rate_stats(theta_after))
            metrics.update(branchscale_stats(branch_params))
            metrics.update(diagnostic_metrics)

            row = {
                "step": step,
                "model/parameter_mode": config.parameter_mode,
                "solver/schedule_phase": solver_settings.phase,
                "solver/pi_iteration_limit": solver_settings.pi_iters,
                "solver/neumann_limit": solver_settings.neumann_terms,
                "solver/pi_convergence_tolerance": solver_settings.pi_max_diff_tol,
                "solver/gradient_convergence_tolerance": solver_settings.gradient_change_tol,
                "solver/budget_no_improvement_steps": solver_budget_state.no_improvement_steps,
                "solver/budget_steps_since_increase": solver_budget_state.steps_since_increase,
                "solver/budget_increased": float(solver_budget_increase_reason != "none"),
                "solver/budget_increase_reason": solver_budget_increase_reason,
                "solver/budget_next_pi_iteration_limit": solver_budget_state.pi_iters,
                "solver/budget_next_neumann_limit": solver_budget_state.neumann_terms,
                "optimizer/phase": phase,
                "optimizer/lbfgs_line_search": config.lbfgs_line_search,
                "diagnostics/mode": config.diagnostic_mode,
                "lr": current_lr,
                "delta_objective_bits": delta,
                "theta_step_inf": theta_step,
                "closure_evals": closure_evals,
                "closure_mean_s": closure_s / max(closure_evals, 1),
                "step_s": time.perf_counter() - step_t0,
                "best/likelihood_data_nll_bits": best_data_nll,
                "best/likelihood_step": best_likelihood_step,
                "best/likelihood_no_improvement_steps": best_likelihood_no_improvement_steps,
                "best/likelihood_improved": float(best_likelihood_improved),
                **metrics,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            last_step = step

            if config.checkpoint_every and step % config.checkpoint_every == 0:
                checkpoint_dir = config.out_dir / "checkpoints"
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    model=model,
                    branch_params=branch_params,
                    optimizers=optimizers,
                    optimizer_phase=phase,
                    step=step,
                    previous_objective=previous_objective,
                    stable_loss_steps=stable_loss_steps,
                    solver_budget_state=solver_budget_state,
                    config=config,
                    row=row,
                    best_data_nll=best_data_nll,
                    best_likelihood_step=best_likelihood_step,
                    best_likelihood_no_improvement_steps=best_likelihood_no_improvement_steps,
                )
                if step % config.plot_every == 0:
                    save_checkpoint(
                        checkpoint_dir / f"step_{step:06d}.pt",
                        model=model,
                        branch_params=branch_params,
                        optimizers=optimizers,
                        optimizer_phase=phase,
                        step=step,
                        previous_objective=previous_objective,
                        stable_loss_steps=stable_loss_steps,
                        solver_budget_state=solver_budget_state,
                        config=config,
                        row=row,
                        best_data_nll=best_data_nll,
                        best_likelihood_step=best_likelihood_step,
                        best_likelihood_no_improvement_steps=best_likelihood_no_improvement_steps,
                    )

            log_payload = dict(row)
            if step % config.plot_every == 0:
                plot_path = config.out_dir / "tree_plots" / f"rates_step_{step:06d}.png"
                plot_tree_rates(
                    root=root,
                    layout_cache=layout_cache,
                    rate_by_label=current_rate_by_label(model, labels, branch_params),
                    out_path=plot_path,
                    title=f"HOGENOM CCP rates step {step} ({config.parameter_mode})",
                )
                image = wandb.image(plot_path)
                if image is not None:
                    log_payload["tree/rates"] = image
                log_payload["tree/plot_path"] = str(plot_path)
            if step % config.log_every == 0 or "tree/rates" in log_payload:
                wandb.log(log_payload, step=step)
            if step % config.log_every == 0:
                branch_text = ""
                if branch_params is not None:
                    branch_text = (
                        f"branch_prior={metrics['regularization/branchscale_bits']:.6g} "
                        f"l_med={metrics['branchscale/l_median']:.6g} "
                        f"l_p95={metrics['branchscale/l_p95']:.6g} "
                        f"l_max={metrics['branchscale/l_max']:.6g} "
                    )
                diagnostic_text = ""
                if config.diagnostic_mode == "step":
                    diagnostic_text = (
                        f"diag_nll_delta={metrics['diagnostics/data_nll_delta_bits']:.6g} "
                        f"diag_objective_delta={metrics['diagnostics/objective_delta_bits']:.6g} "
                        f"diag_prior_delta={metrics['diagnostics/prior_delta_bits']:.6g} "
                        f"step_norm={metrics['diagnostics/trainable_step_norm']:.6g} "
                        f"grad_dot_step={metrics['diagnostics/grad_dot_step']:.6g} "
                        f"cos_grad_step={metrics['diagnostics/cos_grad_step']:.6g} "
                    )
                solver_budget_text = ""
                if config.solver_iteration_schedule == "budget":
                    solver_budget_text = (
                        f"budget_next={solver_budget_state.pi_iters}/{solver_budget_state.neumann_terms} "
                        f"budget_no_improve={solver_budget_state.no_improvement_steps} "
                        f"budget_steps_since_increase={solver_budget_state.steps_since_increase} "
                        f"budget_increase_reason={solver_budget_increase_reason} "
                    )
                print(
                    f"step={step:04d} "
                    f"mode={config.parameter_mode} "
                    f"phase={phase} "
                    f"line_search={config.lbfgs_line_search} "
                    f"nll_bits={metrics['likelihood/data_nll_bits']:.6f} "
                    f"loglik_bits={metrics['likelihood/log_likelihood_bits']:.6f} "
                    f"objective_bits={objective_bits:.6f} "
                    f"delta={float('nan') if delta is None else delta:.6g} "
                    f"lr={current_lr:.6g} "
                    f"grad_norm={metrics['grad/norm']:.6g} "
                    f"grad_inf={metrics['grad/inf']:.6g} "
                    f"Pi_iter={metrics['solver/pi_iterations']:.0f}/{metrics['solver/pi_iteration_cap']:.0f} "
                    f"Neumann={metrics['solver/neumann_terms']:.0f}/{solver_settings.neumann_terms} "
                    f"solver_phase={solver_settings.phase} "
                    f"{solver_budget_text}"
                    f"{branch_text}"
                    f"{diagnostic_text}"
                        f"closure_evals={closure_evals} "
                        f"best_nll={float('nan') if best_data_nll is None else best_data_nll:.6f} "
                        f"best_step={-1 if best_likelihood_step is None else best_likelihood_step} "
                        f"best_wait={best_likelihood_no_improvement_steps} "
                        f"step_s={row['step_s']:.3f}",
                        flush=True,
                )

            if metrics["grad/inf"] <= config.grad_inf_tol:
                stop_reason = "gradient_tolerance"
                break
            if config.loss_patience > 0 and stable_loss_steps >= config.loss_patience:
                stop_reason = "loss_change_patience"
                break
            if (
                config.best_likelihood_patience > 0
                and best_likelihood_no_improvement_steps >= config.best_likelihood_patience
            ):
                stop_reason = "best_likelihood_patience"
                break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        final_rates = final_rates_path(config)
        write_rate_table(final_rates, model, labels, branch_params)
        final_plot = config.out_dir / "tree_plots" / "rates_final.png"
        plot_tree_rates(
            root=root,
            layout_cache=layout_cache,
            rate_by_label=current_rate_by_label(model, labels, branch_params),
            out_path=final_plot,
            title=f"HOGENOM CCP rates final ({config.parameter_mode})",
        )
        final_payload: dict[str, Any] = {
            "run/stop_reason": stop_reason,
            "run/elapsed_s": time.perf_counter() - started,
            "outputs/run_dir": str(config.out_dir),
            "outputs/final_rates_tsv": str(final_rates),
            "outputs/latest_checkpoint": str(config.out_dir / "checkpoints" / "latest.pt"),
            "outputs/best_likelihood_checkpoint": str(config.out_dir / "checkpoints" / "best.pt"),
            "best/likelihood_data_nll_bits": best_data_nll,
            "best/likelihood_step": best_likelihood_step,
            "best/likelihood_no_improvement_steps": best_likelihood_no_improvement_steps,
        }
        branch_params_path = config.out_dir / "branchscaled_parameters_final.pt"
        if branch_params is not None:
            final_payload["outputs/branchscaled_parameters_pt"] = str(branch_params_path)
        image = wandb.image(final_plot)
        if image is not None:
            final_payload["tree/final_rates"] = image
        wandb.log(final_payload, step=max(0, last_step))
        torch.save(
            effective_theta(model, branch_params).detach().cpu(),
            config.out_dir / "theta_final.pt",
        )
        if branch_params is not None:
            torch.save(
                {
                    "shared_theta": branch_params.shared_theta.detach().cpu(),
                    "branch_log_l": branch_params.branch_log_l.detach().cpu(),
                    "branch_l": torch.exp(branch_params.branch_log_l.detach()).cpu(),
                },
                branch_params_path,
            )
        save_checkpoint(
            config.out_dir / "checkpoints" / "latest.pt",
            model=model,
            branch_params=branch_params,
            optimizers=optimizers,
            optimizer_phase=optimizer_phase(config, max(0, last_step)),
            step=last_step,
            previous_objective=previous_objective,
            stable_loss_steps=stable_loss_steps,
            solver_budget_state=solver_budget_state,
            config=config,
            row=None,
            best_data_nll=best_data_nll,
            best_likelihood_step=best_likelihood_step,
            best_likelihood_no_improvement_steps=best_likelihood_no_improvement_steps,
        )
        model.close()
        wandb.finish()
        print(f"stopped reason={stop_reason}", flush=True)
        print("history", history_path, flush=True)
        print("final_rates", final_rates, flush=True)


def parse_family_chunk_size(text: str) -> int | str | None:
    if text.lower() in {"none", "0"}:
        return 0
    if text.lower() == "auto":
        return "auto"
    return int(text)


def _checkpoint_run_dir(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    return checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent


def _timestamped_run_dir(args: argparse.Namespace) -> Path:
    stamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{stamp}_{args.parameter_mode}_{args.optimizer}"
    candidate = args.out_dir / base_name
    suffix = 1
    while candidate.exists():
        candidate = args.out_dir / f"{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


def resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.resume_from is not None:
        return _checkpoint_run_dir(args.resume_from)
    if args.timestamped_out_dir:
        return _timestamped_run_dir(args)
    return args.out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize HOGENOM CCP D/T/L rates with wandb logging."
    )
    parser.add_argument("--species-tree", type=Path, default=SPECIES_TREE)
    parser.add_argument("--families-file", type=Path, default=FAMILIES_FILE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--timestamped-out-dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For new runs, write outputs under a timestamped child of --out-dir. "
            "Use --no-timestamped-out-dir to restore the old fixed-directory behavior. "
            "When --resume-from is set, the checkpoint's run directory is reused."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--mode",
        dest="parameter_mode",
        choices=("specieswise", "uniform", "global", "branchscaled"),
        default="specieswise",
        help=(
            "Parameter sharing mode. 'specieswise' optimizes one D/T/L vector per "
            "species-tree node; 'uniform'/'global' optimize one shared D/T/L vector; "
            "'branchscaled' optimizes shared D/T/L rates times one branch multiplier per node."
        ),
    )
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument("--family-chunk-size", type=parse_family_chunk_size, default="0")
    parser.add_argument("--clade-budget", type=int, default=600_000)
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--max-iters-e", type=int, default=2000)
    parser.add_argument("--max-iters-pi", type=int, default=64)
    parser.add_argument("--max-neumann-terms", type=int, default=64)
    parser.add_argument("--convergence-check-interval", type=int, default=4)
    parser.add_argument("--e-logsumexp-tol", type=float, default=1e-5)
    parser.add_argument("--pi-max-diff-tol", type=float, default=1e-5)
    parser.add_argument("--gradient-change-tol", type=float, default=1e-4)
    parser.add_argument("--gradient-change-rtol", type=float, default=1e-4)
    parser.add_argument(
        "--solver-iteration-schedule",
        choices=("budget", "staged", "adaptive"),
        default="budget",
        help=(
            "Pi/Neumann schedule. 'budget' starts with a small adaptive iteration "
            "cap and increases it after repeated non-improving objective steps; "
            "'staged' runs exactly 4 iterations for steps 0-49, exactly 8 for "
            "steps 50-99, then adaptive convergence; 'adaptive' uses "
            "--max-iters-pi/--max-neumann-terms from step 0."
        ),
    )
    parser.add_argument("--solver-budget-initial-iters", type=int, default=DEFAULT_SOLVER_BUDGET_ITERS)
    parser.add_argument("--solver-budget-increment", type=int, default=4)
    parser.add_argument("--solver-budget-patience", type=int, default=2)
    parser.add_argument("--solver-budget-step-interval", type=int, default=50)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "lbfgs", "adam-lbfgs"),
        default="adam",
        help="Use Adam, PyTorch LBFGS, or Adam warmup followed by LBFGS.",
    )
    parser.add_argument(
        "--adam-warmup-steps",
        type=int,
        default=100,
        help="Number of Adam steps before switching to LBFGS when --optimizer adam-lbfgs.",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--lr-decay-every",
        type=int,
        default=100,
        help="Multiply the learning rate by --lr-decay-factor every N steps. Use 0 to disable.",
    )
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--lbfgs-lr", type=float, default=0.1)
    parser.add_argument("--lbfgs-history-size", type=int, default=20)
    parser.add_argument(
        "--lbfgs-max-iter",
        type=int,
        default=1,
        help="Maximum number of LBFGS inner iterations per optimizer step.",
    )
    parser.add_argument(
        "--lbfgs-max-eval",
        type=int,
        default=None,
        help="Maximum number of LBFGS closure evaluations per optimizer step.",
    )
    parser.add_argument(
        "--lbfgs-tolerance-grad",
        type=float,
        default=1e-7,
        help="PyTorch LBFGS first-order optimality tolerance for one LBFGS step.",
    )
    parser.add_argument(
        "--lbfgs-tolerance-change",
        type=float,
        default=1e-9,
        help="PyTorch LBFGS parameter/function-change tolerance for one LBFGS step.",
    )
    parser.add_argument(
        "--lbfgs-line-search",
        choices=("none", "strong_wolfe"),
        default="none",
        help="Line search used by torch.optim.LBFGS.",
    )
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=1e9)
    parser.add_argument("--grad-inf-tol", type=float, default=1e-3)
    parser.add_argument("--loss-change-tol", type=float, default=1e-3)
    parser.add_argument("--loss-patience", type=int, default=20)
    parser.add_argument(
        "--best-likelihood-patience",
        type=int,
        default=0,
        help=(
            "When positive, save checkpoints/best.pt whenever data NLL improves "
            "and stop after this many evaluated steps without a new best data NLL."
        ),
    )
    parser.add_argument(
        "--best-likelihood-min-delta",
        type=float,
        default=0.0,
        help="Minimum data-NLL decrease, in bits, required to reset best-likelihood patience.",
    )
    parser.add_argument("--beta-ps-alpha", type=float, default=4.0)
    parser.add_argument("--beta-ps-beta", type=float, default=1.0)
    parser.add_argument("--beta-prior-weight", type=float, default=1.0)
    parser.add_argument("--branchscale-prior-weight", type=float, default=1.0)
    parser.add_argument(
        "--diagnostic-mode",
        choices=("off", "step"),
        default="off",
        help=(
            "Enable extra optimizer diagnostics. 'step' adds one post-step "
            "objective evaluation and logs before/after deltas plus "
            "gradient-step alignment."
        ),
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Alias for --diagnostic-mode step.",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--plot-every", type=int, default=10)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help=(
            "Write an atomic checkpoints/latest.pt every N optimizer steps. "
            "Numbered checkpoints are also archived on plot steps. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume theta and optimizer state from a checkpoint .pt file.",
    )
    parser.add_argument("--wandb-project", default="gpurec-hogenom-ccp")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.max_iters_pi < 1 or args.max_iters_pi % 2 != 0:
        raise ValueError("--max-iters-pi must be a positive even integer")
    if args.max_neumann_terms < 1:
        raise ValueError("--max-neumann-terms must be positive")
    if args.solver_budget_initial_iters < 1 or args.solver_budget_initial_iters % 2 != 0:
        raise ValueError("--solver-budget-initial-iters must be a positive even integer")
    if args.solver_budget_increment < 1 or args.solver_budget_increment % 2 != 0:
        raise ValueError("--solver-budget-increment must be a positive even integer")
    if args.solver_budget_patience < 1:
        raise ValueError("--solver-budget-patience must be positive")
    if args.solver_budget_step_interval < 1:
        raise ValueError("--solver-budget-step-interval must be positive")
    if args.convergence_check_interval < 1:
        raise ValueError("--convergence-check-interval must be positive")
    if args.adam_warmup_steps < 0:
        raise ValueError("--adam-warmup-steps must be non-negative")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.loss_patience < 0:
        raise ValueError("--loss-patience must be non-negative")
    if args.lr_decay_every < 0:
        raise ValueError("--lr-decay-every must be non-negative")
    if not (0.0 < args.lr_decay_factor <= 1.0):
        raise ValueError("--lr-decay-factor must be in (0, 1]")
    if args.lbfgs_lr <= 0.0:
        raise ValueError("--lbfgs-lr must be positive")
    if args.lbfgs_history_size < 1:
        raise ValueError("--lbfgs-history-size must be positive")
    if args.lbfgs_max_iter < 1:
        raise ValueError("--lbfgs-max-iter must be positive")
    if args.lbfgs_max_eval is not None and args.lbfgs_max_eval < 1:
        raise ValueError("--lbfgs-max-eval must be positive")
    if args.lbfgs_tolerance_grad < 0.0:
        raise ValueError("--lbfgs-tolerance-grad must be non-negative")
    if args.lbfgs_tolerance_change < 0.0:
        raise ValueError("--lbfgs-tolerance-change must be non-negative")
    if args.best_likelihood_patience < 0:
        raise ValueError("--best-likelihood-patience must be non-negative")
    if args.best_likelihood_min_delta < 0.0:
        raise ValueError("--best-likelihood-min-delta must be non-negative")
    if args.branchscale_prior_weight < 0.0:
        raise ValueError("--branchscale-prior-weight must be non-negative")
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")
    if args.plot_every < 1:
        raise ValueError("--plot-every must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")
    if args.resume_from is not None and not args.resume_from.exists():
        raise FileNotFoundError(args.resume_from)
    if args.diagnostics:
        args.diagnostic_mode = "step"
    return args


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        species_tree=args.species_tree,
        families_file=args.families_file,
        out_dir=resolve_out_dir(args),
        device=args.device,
        max_families=args.max_families,
        family_chunk_size=args.family_chunk_size,
        clade_budget=args.clade_budget,
        max_wave_size=args.max_wave_size,
        max_iters_e=args.max_iters_e,
        max_iters_pi=args.max_iters_pi,
        max_neumann_terms=args.max_neumann_terms,
        convergence_check_interval=args.convergence_check_interval,
        e_logsumexp_tol=args.e_logsumexp_tol,
        pi_max_diff_tol=args.pi_max_diff_tol,
        gradient_change_tol=args.gradient_change_tol,
        gradient_change_rtol=args.gradient_change_rtol,
        solver_iteration_schedule=args.solver_iteration_schedule,
        solver_budget_initial_iters=args.solver_budget_initial_iters,
        solver_budget_increment=args.solver_budget_increment,
        solver_budget_patience=args.solver_budget_patience,
        solver_budget_step_interval=args.solver_budget_step_interval,
        parameter_mode=args.parameter_mode,
        optimizer=args.optimizer,
        adam_warmup_steps=args.adam_warmup_steps,
        steps=args.steps,
        lr=args.lr,
        lr_decay_every=args.lr_decay_every,
        lr_decay_factor=args.lr_decay_factor,
        lbfgs_lr=args.lbfgs_lr,
        lbfgs_history_size=args.lbfgs_history_size,
        lbfgs_max_iter=args.lbfgs_max_iter,
        lbfgs_max_eval=args.lbfgs_max_eval,
        lbfgs_tolerance_grad=args.lbfgs_tolerance_grad,
        lbfgs_tolerance_change=args.lbfgs_tolerance_change,
        lbfgs_line_search=args.lbfgs_line_search,
        min_rate=args.min_rate,
        max_rate=args.max_rate,
        grad_inf_tol=args.grad_inf_tol,
        loss_change_tol=args.loss_change_tol,
        loss_patience=args.loss_patience,
        best_likelihood_patience=args.best_likelihood_patience,
        best_likelihood_min_delta=args.best_likelihood_min_delta,
        beta_ps_alpha=args.beta_ps_alpha,
        beta_ps_beta=args.beta_ps_beta,
        beta_prior_weight=args.beta_prior_weight,
        branchscale_prior_weight=args.branchscale_prior_weight,
        diagnostic_mode=args.diagnostic_mode,
        log_every=args.log_every,
        plot_every=args.plot_every,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    if not config.species_tree.exists():
        raise FileNotFoundError(config.species_tree)
    if not config.families_file.exists():
        raise FileNotFoundError(config.families_file)
    run(config, args)


if __name__ == "__main__":
    main()
