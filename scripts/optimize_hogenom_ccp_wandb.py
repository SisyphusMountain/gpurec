from __future__ import annotations

import argparse
import csv
import json
import math
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
PREPROCESS_CACHE = HOGENOM_DIR / "output_gpurec_ccp_reconciliation" / "preprocess_cache"
OUT_DIR = HOGENOM_DIR / "output_gpurec_wandb_adam"

LN2 = math.log(2.0)
RATE_FIELDS = (("D", 0), ("T", 2), ("L", 1))
QUANTILE_PROBS = torch.tensor([0.0, 0.5, 0.95, 1.0])


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
    preprocess_cache: Path
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
    steps: int
    lr: float
    lr_decay_every: int
    lr_decay_factor: float
    min_rate: float
    max_rate: float
    grad_inf_tol: float
    loss_change_tol: float
    loss_patience: int
    beta_ps_alpha: float
    beta_ps_beta: float
    beta_prior_weight: float
    log_every: int
    plot_every: int
    checkpoint_every: int
    resume_from: Path | None
    wandb_project: str
    wandb_entity: str | None
    wandb_run_name: str | None
    wandb_mode: str


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
                "wandb is not installed. Install it with `uv pip install wandb`, "
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


def pi_iteration_count(model: GeneReconModel) -> tuple[int, int]:
    if getattr(model, "_batched_resident", False):
        statics = [
            static
            for static in getattr(model, "_batch_statics", [])
            if static is not None and static.last_solver_stats is not None
        ]
    else:
        static = model.static
        statics = [static] if static.last_solver_stats is not None else []

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
        mode="specieswise",
        start=0,
        max_families=config.max_families,
        device=device,
        dtype=dtype,
        theta_init_rates=(0.05, 0.05, 0.05),
        preprocess_cache_dir=config.preprocess_cache,
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


def current_rate_by_label(model: GeneReconModel, labels: list[str]) -> dict[str, dict[str, float]]:
    theta = model.theta.detach().reshape(-1, 3).cpu()
    rates = torch.exp2(theta)
    pS = pS_values(model.theta.detach()).cpu()
    if len(labels) != int(theta.shape[0]):
        raise RuntimeError(f"species label count {len(labels)} != theta rows {theta.shape[0]}")
    return {
        label: {
            "D": float(rates[row, 0]),
            "T": float(rates[row, 2]),
            "L": float(rates[row, 1]),
            "pS": float(pS[row]),
        }
        for row, label in enumerate(labels)
    }


def write_rate_table(path: Path, model: GeneReconModel, labels: list[str]) -> None:
    rates = current_rate_by_label(model, labels)
    theta = model.theta.detach().reshape(-1, 3).cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["row", "label", "D", "T", "L", "pS", "theta_D", "theta_T", "theta_L"])
        for row, label in enumerate(labels):
            vals = rates[label]
            writer.writerow(
                [
                    row,
                    label,
                    vals["D"],
                    vals["T"],
                    vals["L"],
                    vals["pS"],
                    float(theta[row, 0]),
                    float(theta[row, 2]),
                    float(theta[row, 1]),
                ]
            )


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def save_checkpoint(
    path: Path,
    *,
    model: GeneReconModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    previous_objective: float | None,
    stable_loss_steps: int,
    config: RunConfig,
    row: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "next_step": int(step) + 1,
        "theta": model.theta.detach().cpu(),
        "optimizer_state": optimizer.state_dict(),
        "previous_objective": previous_objective,
        "stable_loss_steps": int(stable_loss_steps),
        "config": _jsonable(asdict(config)),
        "last_row": row,
    }
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: GeneReconModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float | None, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    theta = checkpoint["theta"].to(device=device, dtype=model.theta.dtype)
    with torch.no_grad():
        model.theta.copy_(theta)
        model.theta.grad = None
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        _optimizer_state_to_device(optimizer, device)
    model.clear()
    return (
        int(checkpoint.get("next_step", int(checkpoint.get("step", -1)) + 1)),
        checkpoint.get("previous_objective"),
        int(checkpoint.get("stable_loss_steps", 0)),
    )


def evaluate_and_backward(
    model: GeneReconModel,
    config: RunConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    model.theta.grad = None
    data_nll = model.full_loss()
    prior = beta_ps_prior_bits(
        model.theta,
        alpha=config.beta_ps_alpha,
        beta=config.beta_ps_beta,
        weight=config.beta_prior_weight,
    )
    objective = data_nll + prior
    objective.backward()
    synchronize()
    if model.theta.grad is None:
        raise RuntimeError("missing theta gradient")
    pi_iters, pi_iter_cap = pi_iteration_count(model)
    metrics = {
        "likelihood/data_nll_bits": float(data_nll.detach().cpu()),
        "likelihood/log_likelihood_bits": float((-data_nll).detach().cpu()),
        "objective/bits": float(objective.detach().cpu()),
        "regularization/beta_ps_bits": float(prior.detach().cpu()),
        "solver/pi_iterations": float(pi_iters),
        "solver/pi_iteration_cap": float(pi_iter_cap),
    }
    metrics.update(gradient_stats(model.theta.grad))
    return objective, metrics


def run(config: RunConfig, args: argparse.Namespace) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    history_path = config.out_dir / "history.jsonl"
    if config.resume_from is None and history_path.exists():
        history_path.unlink()

    wandb = WandbSink(args, config)
    root = parse_newick(config.species_tree)
    layout_cache = tree_layout(root)
    model = build_model(config)
    labels = species_labels(model)
    optimizer = torch.optim.Adam([model.theta], lr=config.lr)
    start_step = 0
    previous_objective: float | None = None
    stable_loss_steps = 0
    if config.resume_from is not None:
        start_step, previous_objective, stable_loss_steps = load_checkpoint(
            config.resume_from,
            model=model,
            optimizer=optimizer,
            device=torch.device(config.device),
        )
        print(f"resumed checkpoint={config.resume_from} next_step={start_step}", flush=True)

    build_info = {
        "dataset/families": sum(meta.family_count for meta in model.batch_metadata),
        "dataset/species": model.n_species,
        "dataset/batches": len(model.batch_metadata),
        "dataset/waves": sum(meta.wave_count for meta in model.batch_metadata),
    }
    wandb.log(build_info, step=0)
    print(
        "built "
        f"families={int(build_info['dataset/families'])} "
        f"species={int(build_info['dataset/species'])} "
        f"batches={int(build_info['dataset/batches'])} "
        f"waves={int(build_info['dataset/waves'])}",
        flush=True,
    )

    stop_reason = "max_steps"
    last_step = max(0, start_step - 1)
    started = time.perf_counter()
    try:
        for step in range(start_step, config.steps):
            last_step = step
            step_t0 = time.perf_counter()
            current_lr = scheduled_lr(config, step)
            set_optimizer_lr(optimizer, current_lr)
            optimizer.zero_grad(set_to_none=True)
            theta_before = model.theta.detach().clone()
            objective, metrics = evaluate_and_backward(model, config)
            objective_bits = metrics["objective/bits"]
            delta = None if previous_objective is None else previous_objective - objective_bits
            if delta is not None and abs(delta) <= config.loss_change_tol:
                stable_loss_steps += 1
            else:
                stable_loss_steps = 0
            previous_objective = objective_bits

            if not torch.isfinite(objective).item() or not torch.isfinite(model.theta.grad).all().item():
                stop_reason = "nonfinite_objective_or_gradient"
                break

            optimizer.step()
            synchronize()
            model.clamp_theta_(min_rate=config.min_rate, max_rate=config.max_rate)
            theta_step = float((model.theta.detach() - theta_before).abs().amax().cpu())
            model.clear()
            metrics.update(rate_stats(model.theta))

            row = {
                "step": step,
                "lr": current_lr,
                "delta_objective_bits": delta,
                "theta_step_inf": theta_step,
                "step_s": time.perf_counter() - step_t0,
                **metrics,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

            if config.checkpoint_every and step % config.checkpoint_every == 0:
                checkpoint_dir = config.out_dir / "checkpoints"
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    previous_objective=previous_objective,
                    stable_loss_steps=stable_loss_steps,
                    config=config,
                    row=row,
                )
                if step % config.plot_every == 0:
                    save_checkpoint(
                        checkpoint_dir / f"step_{step:06d}.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        previous_objective=previous_objective,
                        stable_loss_steps=stable_loss_steps,
                        config=config,
                        row=row,
                    )

            log_payload = dict(row)
            if step % config.plot_every == 0:
                plot_path = config.out_dir / "tree_plots" / f"rates_step_{step:06d}.png"
                plot_tree_rates(
                    root=root,
                    layout_cache=layout_cache,
                    rate_by_label=current_rate_by_label(model, labels),
                    out_path=plot_path,
                    title=f"HOGENOM CCP rates step {step}",
                )
                image = wandb.image(plot_path)
                if image is not None:
                    log_payload["tree/rates"] = image
                log_payload["tree/plot_path"] = str(plot_path)
            if step % config.log_every == 0 or "tree/rates" in log_payload:
                wandb.log(log_payload, step=step)
            if step % config.log_every == 0:
                print(
                    f"step={step:04d} "
                    f"nll_bits={metrics['likelihood/data_nll_bits']:.6f} "
                    f"loglik_bits={metrics['likelihood/log_likelihood_bits']:.6f} "
                    f"objective_bits={objective_bits:.6f} "
                    f"delta={float('nan') if delta is None else delta:.6g} "
                    f"lr={current_lr:.6g} "
                    f"grad_norm={metrics['grad/norm']:.6g} "
                    f"grad_inf={metrics['grad/inf']:.6g} "
                    f"Pi_iter={metrics['solver/pi_iterations']:.0f}/{metrics['solver/pi_iteration_cap']:.0f} "
                    f"step_s={row['step_s']:.3f}",
                    flush=True,
                )

            if metrics["grad/inf"] <= config.grad_inf_tol:
                stop_reason = "gradient_tolerance"
                break
            if stable_loss_steps >= config.loss_patience:
                stop_reason = "loss_change_patience"
                break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        final_rates = config.out_dir / "specieswise_node_rates_final.tsv"
        write_rate_table(final_rates, model, labels)
        final_plot = config.out_dir / "tree_plots" / "rates_final.png"
        plot_tree_rates(
            root=root,
            layout_cache=layout_cache,
            rate_by_label=current_rate_by_label(model, labels),
            out_path=final_plot,
            title="HOGENOM CCP rates final",
        )
        final_payload: dict[str, Any] = {
            "run/stop_reason": stop_reason,
            "run/elapsed_s": time.perf_counter() - started,
            "outputs/final_rates_tsv": str(final_rates),
            "outputs/latest_checkpoint": str(config.out_dir / "checkpoints" / "latest.pt"),
        }
        image = wandb.image(final_plot)
        if image is not None:
            final_payload["tree/final_rates"] = image
        wandb.log(final_payload, step=last_step)
        torch.save(model.theta.detach().cpu(), config.out_dir / "theta_final.pt")
        save_checkpoint(
            config.out_dir / "checkpoints" / "latest.pt",
            model=model,
            optimizer=optimizer,
            step=last_step,
            previous_objective=previous_objective,
            stable_loss_steps=stable_loss_steps,
            config=config,
            row=None,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize HOGENOM CCP specieswise rates with wandb logging."
    )
    parser.add_argument("--species-tree", type=Path, default=SPECIES_TREE)
    parser.add_argument("--families-file", type=Path, default=FAMILIES_FILE)
    parser.add_argument("--preprocess-cache", type=Path, default=PREPROCESS_CACHE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument("--family-chunk-size", type=parse_family_chunk_size, default="0")
    parser.add_argument("--clade-budget", type=int, default=305_000)
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--max-iters-e", type=int, default=2000)
    parser.add_argument("--max-iters-pi", type=int, default=64)
    parser.add_argument("--max-neumann-terms", type=int, default=64)
    parser.add_argument("--convergence-check-interval", type=int, default=4)
    parser.add_argument("--e-logsumexp-tol", type=float, default=1e-5)
    parser.add_argument("--pi-max-diff-tol", type=float, default=1e-5)
    parser.add_argument("--gradient-change-tol", type=float, default=1e-4)
    parser.add_argument("--gradient-change-rtol", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--lr-decay-every",
        type=int,
        default=100,
        help="Multiply the learning rate by --lr-decay-factor every N steps. Use 0 to disable.",
    )
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=100.0)
    parser.add_argument("--grad-inf-tol", type=float, default=1e-3)
    parser.add_argument("--loss-change-tol", type=float, default=1e-3)
    parser.add_argument("--loss-patience", type=int, default=20)
    parser.add_argument("--beta-ps-alpha", type=float, default=4.0)
    parser.add_argument("--beta-ps-beta", type=float, default=1.0)
    parser.add_argument("--beta-prior-weight", type=float, default=1.0)
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
    if args.convergence_check_interval < 1:
        raise ValueError("--convergence-check-interval must be positive")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.lr_decay_every < 0:
        raise ValueError("--lr-decay-every must be non-negative")
    if not (0.0 < args.lr_decay_factor <= 1.0):
        raise ValueError("--lr-decay-factor must be in (0, 1]")
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")
    if args.plot_every < 1:
        raise ValueError("--plot-every must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")
    if args.resume_from is not None and not args.resume_from.exists():
        raise FileNotFoundError(args.resume_from)
    return args


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        species_tree=args.species_tree,
        families_file=args.families_file,
        preprocess_cache=args.preprocess_cache,
        out_dir=args.out_dir,
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
        steps=args.steps,
        lr=args.lr,
        lr_decay_every=args.lr_decay_every,
        lr_decay_factor=args.lr_decay_factor,
        min_rate=args.min_rate,
        max_rate=args.max_rate,
        grad_inf_tol=args.grad_inf_tol,
        loss_change_tol=args.loss_change_tol,
        loss_patience=args.loss_patience,
        beta_ps_alpha=args.beta_ps_alpha,
        beta_ps_beta=args.beta_ps_beta,
        beta_prior_weight=args.beta_prior_weight,
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
