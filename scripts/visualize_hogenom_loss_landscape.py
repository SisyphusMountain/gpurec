#!/usr/bin/env python3
"""Visualize HOGENOM family D/L/T loss landscapes with gpurec.

This is a checkout-local analysis script.  It follows the loss-landscape
visualization idea from Li et al. 2018, adapted to gpurec's low-dimensional
rate surface: each selected HOGENOM family is evaluated on D-L, D-T, and L-T
log2 fold-change planes around an optimized or supplied D/L/T anchor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpurec import GeneReconModel  # noqa: E402
from gpurec.core.model import GeneDataset, parse_alerax_family_file  # noqa: E402


HOGENOM_DIR = REPO / "tests" / "data" / "HOGENOM" / "hogenom"
SPECIES_TREE = HOGENOM_DIR / "hogenom_S.tree"
FAMILIES_FILE = HOGENOM_DIR / "hogenom_families.local.txt"
OUT_DIR = HOGENOM_DIR / "output_gpurec_loss_landscape"

RATE_FIELDS = (("D", 0), ("L", 1), ("T", 2))
RATE_TO_COL = {name: col for name, col in RATE_FIELDS}
PLANES = (("D", "L"), ("D", "T"), ("L", "T"))


@dataclass(frozen=True)
class FamilySelection:
    index: int
    name: str
    leaf_count: int


@dataclass(frozen=True)
class LandscapePoint:
    family_index: int
    family_name: str
    leaf_count: int
    plane: str
    x_rate: str
    y_rate: str
    x_offset_log2: float
    y_offset_log2: float
    nll_bits: float
    delta_nll_bits: float
    delta_nll_bits_per_leaf: float
    theta_D: float
    theta_L: float
    theta_T: float
    rate_D: float
    rate_L: float
    rate_T: float


@dataclass(frozen=True)
class OptimizationTracePoint:
    family_index: int
    family_name: str
    step: int
    nll_bits: float
    theta_D: float
    theta_L: float
    theta_T: float


@dataclass(frozen=True)
class FamilyLandscape:
    selection: FamilySelection
    center_theta: torch.Tensor
    center_nll_bits: float
    points: list[LandscapePoint]
    hessian: torch.Tensor
    hessian_eigenvalues: torch.Tensor
    trace: list[OptimizationTracePoint]


class BatchedFamilyProbeEvaluator:
    """Evaluate many D/L/T theta rows for one duplicated HOGENOM family."""

    def __init__(
        self,
        *,
        species_tree: Path,
        family_name: str,
        tree_paths: Sequence[str],
        leaf_species_map: dict[str, str],
        batch_size: int,
        device: str,
        dtype: torch.dtype,
        fixed_iters_E: int,
        fixed_iters_Pi: int,
        neumann_terms: int,
        max_wave_size: int,
        preprocess_cpu_cores: int | None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        family_names = [
            f"{family_name}__probe_{idx:05d}"
            for idx in range(self.batch_size)
        ]
        dataset = GeneDataset(
            species_tree_path=str(species_tree),
            gene_tree_paths=[list(tree_paths) for _ in range(self.batch_size)],
            genewise=True,
            specieswise=False,
            dtype=dtype,
            device=torch.device(device),
            preprocess_cpu_cores=preprocess_cpu_cores,
            family_names=family_names,
            leaf_species_maps=[dict(leaf_species_map) for _ in range(self.batch_size)],
        )
        theta_init = torch.zeros(
            (self.batch_size, 3),
            dtype=dtype,
            device=torch.device(device),
        )
        self.model = GeneReconModel(
            dataset=dataset,
            mode="genewise",
            theta_init=theta_init,
            fixed_iters_E=fixed_iters_E,
            fixed_iters_Pi=fixed_iters_Pi,
            neumann_terms=neumann_terms,
            max_wave_size=max_wave_size,
        )

    def evaluate(self, theta_rows: torch.Tensor) -> torch.Tensor:
        rows = theta_rows.detach().cpu().reshape(-1, 3).to(torch.float64)
        if rows.numel() == 0:
            return torch.empty((0,), dtype=torch.float64)
        values = evaluate_thetas_in_blocks(
            rows,
            batch_size=self.batch_size,
            evaluate_block=self._evaluate_block,
        )
        return values.to(dtype=torch.float64)

    def _evaluate_block(self, block: torch.Tensor) -> torch.Tensor:
        n_rows = int(block.shape[0])
        if n_rows <= 0 or n_rows > self.batch_size:
            raise ValueError("block size must be between 1 and batch_size")
        device_block = block.to(device=self.model.theta.device, dtype=self.model.theta.dtype)
        with torch.no_grad():
            self.model.theta[:n_rows].copy_(device_block)
            if n_rows < self.batch_size:
                self.model.theta[n_rows:].copy_(device_block[-1].expand(self.batch_size - n_rows, -1))
            values = self.model.full_nll_per_family().detach().cpu()[:n_rows]
        return values

    def close(self) -> None:
        self.model.close()


def parse_rate_triple(value: str | Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(",", " ").split()]
    else:
        parts = [str(part) for part in value]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected exactly three D,L,T rates")
    try:
        rates = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("D,L,T rates must be numeric") from exc
    if any(not math.isfinite(rate) or rate <= 0.0 for rate in rates):
        raise argparse.ArgumentTypeError("D,L,T rates must be finite and positive")
    return rates  # type: ignore[return-value]


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype {name!r}")


def validate_grid_points(value: int) -> int:
    points = int(value)
    if points < 3 or points % 2 == 0:
        raise argparse.ArgumentTypeError("grid-points must be an odd integer >= 3")
    return points


def positive_int_arg(name: str) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError(f"{name} must be positive")
        return parsed

    return parse


def read_family_metadata(
    families_file: Path,
) -> tuple[list[str], list[list[str]], list[dict[str, str]]]:
    names, tree_paths, leaf_maps = parse_alerax_family_file(families_file)
    return (
        [str(name) for name in names],
        [[str(path) for path in paths] for paths in tree_paths],
        [dict(mapping or {}) for mapping in leaf_maps],
    )


def estimate_leaf_count(tree_paths: Sequence[str], leaf_map: dict[str, str]) -> int:
    if leaf_map:
        return len(leaf_map)
    if not tree_paths:
        return 1
    text = Path(tree_paths[0]).read_text(encoding="utf-8").strip()
    record = text.split(";", 1)[0].strip()
    if not record:
        return 1
    return max(1, record.count(",") + 1)


def representative_family_indices(
    leaf_counts: Sequence[int],
    *,
    target_count: int,
) -> list[int]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if not leaf_counts:
        return []
    if len(leaf_counts) <= target_count:
        return list(range(len(leaf_counts)))

    order = sorted(range(len(leaf_counts)), key=lambda idx: (leaf_counts[idx], idx))
    if target_count == 1:
        return [order[len(order) // 2]]
    positions = [
        round(i * (len(order) - 1) / (target_count - 1))
        for i in range(target_count)
    ]
    return [order[pos] for pos in positions]


def resolve_family_selection(
    *,
    names: Sequence[str],
    tree_paths: Sequence[Sequence[str]],
    leaf_maps: Sequence[dict[str, str]],
    family_indices: Sequence[int] | None,
    family_names: Sequence[str] | None,
    representative_count: int,
) -> list[FamilySelection]:
    if family_indices and family_names:
        raise ValueError("use either family indices or family names, not both")
    leaf_counts = [
        estimate_leaf_count(paths, mapping)
        for paths, mapping in zip(tree_paths, leaf_maps)
    ]
    if family_indices:
        selected = [int(index) for index in family_indices]
    elif family_names:
        by_name = {name: idx for idx, name in enumerate(names)}
        missing = [name for name in family_names if name not in by_name]
        if missing:
            raise ValueError(f"unknown family name(s): {', '.join(missing)}")
        selected = [by_name[name] for name in family_names]
    else:
        selected = representative_family_indices(
            leaf_counts,
            target_count=representative_count,
        )

    out: list[FamilySelection] = []
    seen: set[int] = set()
    for index in selected:
        if index < 0 or index >= len(names):
            raise IndexError(f"family index {index} outside 0..{len(names) - 1}")
        if index in seen:
            continue
        seen.add(index)
        out.append(
            FamilySelection(
                index=index,
                name=names[index],
                leaf_count=max(1, int(leaf_counts[index])),
            )
        )
    return out


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_anchor_theta(path: Path) -> torch.Tensor:
    payload = _torch_load(path)
    if torch.is_tensor(payload):
        theta = payload
    elif isinstance(payload, dict) and torch.is_tensor(payload.get("theta")):
        theta = payload["theta"]
    else:
        raise ValueError(
            f"{path} must contain a tensor or a checkpoint dictionary with 'theta'"
        )
    theta = theta.detach().cpu()
    if theta.numel() == 3:
        return theta.reshape(1, 3).to(dtype=torch.float64)
    if theta.numel() == 0 or theta.numel() % 3 != 0:
        raise ValueError("anchor theta must contain D/L/T triples")
    return theta.reshape(-1, 3).to(dtype=torch.float64)


def anchor_for_family(
    anchor_theta: torch.Tensor,
    *,
    family_index: int,
    family_count: int,
) -> torch.Tensor:
    rows = int(anchor_theta.reshape(-1, 3).shape[0])
    if rows == 1:
        return anchor_theta.reshape(1, 3)[0].clone()
    if rows == family_count:
        return anchor_theta.reshape(-1, 3)[family_index].clone()
    raise ValueError(
        f"anchor theta has {rows} row(s), expected 1 or {family_count} family rows"
    )


def evaluate_thetas_in_blocks(
    theta_rows: torch.Tensor,
    *,
    batch_size: int,
    evaluate_block: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = theta_rows.detach().cpu().reshape(-1, 3).to(torch.float64)
    outputs: list[torch.Tensor] = []
    for start in range(0, int(rows.shape[0]), int(batch_size)):
        block = rows[start:start + int(batch_size)]
        values = evaluate_block(block).detach().cpu().reshape(-1)
        if int(values.numel()) != int(block.shape[0]):
            raise RuntimeError(
                f"evaluate_block returned {int(values.numel())} value(s) "
                f"for {int(block.shape[0])} theta row(s)"
            )
        outputs.append(values.to(torch.float64))
    if not outputs:
        return torch.empty((0,), dtype=torch.float64)
    return torch.cat(outputs)


def theta_rates(theta: torch.Tensor) -> tuple[float, float, float]:
    rates = torch.exp2(theta.detach().cpu().reshape(3).to(torch.float64))
    return float(rates[0]), float(rates[1]), float(rates[2])


def _point_from_probe(
    *,
    selection: FamilySelection,
    plane: tuple[str, str],
    x_offset: float,
    y_offset: float,
    nll_bits: float,
    center_nll_bits: float,
    probe_theta: torch.Tensor,
) -> LandscapePoint:
    rate_D, rate_L, rate_T = theta_rates(probe_theta)
    delta = float(nll_bits - center_nll_bits)
    return LandscapePoint(
        family_index=selection.index,
        family_name=selection.name,
        leaf_count=selection.leaf_count,
        plane=f"{plane[0]}-{plane[1]}",
        x_rate=plane[0],
        y_rate=plane[1],
        x_offset_log2=float(x_offset),
        y_offset_log2=float(y_offset),
        nll_bits=float(nll_bits),
        delta_nll_bits=delta,
        delta_nll_bits_per_leaf=delta / float(selection.leaf_count),
        theta_D=float(probe_theta[0].detach().cpu()),
        theta_L=float(probe_theta[1].detach().cpu()),
        theta_T=float(probe_theta[2].detach().cpu()),
        rate_D=rate_D,
        rate_L=rate_L,
        rate_T=rate_T,
    )


def evaluate_loss_for_theta(model: GeneReconModel, theta: torch.Tensor) -> float:
    probe = theta.detach().to(device=model.theta.device, dtype=model.theta.dtype)
    with torch.no_grad():
        loss = model.full_loss_for_theta(probe)
    return float(loss.detach().cpu())


def optimize_family_anchor(
    model: GeneReconModel,
    *,
    selection: FamilySelection,
    steps: int,
    lr: float,
    min_rate: float,
    max_rate: float | None,
) -> list[OptimizationTracePoint]:
    if steps < 0:
        raise ValueError("optimize steps must be non-negative")
    if steps == 0:
        loss = evaluate_loss_for_theta(model, model.theta.detach())
        theta = model.theta.detach().reshape(3).cpu()
        return [
            OptimizationTracePoint(
                selection.index,
                selection.name,
                0,
                loss,
                float(theta[0]),
                float(theta[1]),
                float(theta[2]),
            )
        ]

    optimizer = torch.optim.Adam([model.theta], lr=lr)
    trace: list[OptimizationTracePoint] = []
    best_loss = math.inf
    best_theta = model.theta.detach().clone()
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = model.full_loss()
        loss.backward()
        loss_value = float(loss.detach().cpu())
        theta = model.theta.detach().reshape(3).cpu()
        trace.append(
            OptimizationTracePoint(
                selection.index,
                selection.name,
                step,
                loss_value,
                float(theta[0]),
                float(theta[1]),
                float(theta[2]),
            )
        )
        if math.isfinite(loss_value) and loss_value < best_loss:
            best_loss = loss_value
            best_theta = model.theta.detach().clone()
        if step == steps:
            break
        if model.theta.grad is None or not torch.isfinite(model.theta.grad).all().item():
            raise RuntimeError(f"non-finite gradient while optimizing {selection.name}")
        optimizer.step()
        model.clamp_theta_(min_rate=min_rate, max_rate=max_rate)

    with torch.no_grad():
        model.theta.copy_(best_theta)
    return trace


def finite_difference_hessian(
    evaluate: Callable[[torch.Tensor], float],
    center_theta: torch.Tensor,
    *,
    step_log2: float,
    scale: float = 1.0,
) -> torch.Tensor:
    if step_log2 <= 0.0:
        raise ValueError("step_log2 must be positive")
    center = center_theta.detach().cpu().reshape(3).to(torch.float64)
    f0 = evaluate(center) * scale
    hessian = torch.zeros((3, 3), dtype=torch.float64)
    for i in range(3):
        ei = torch.zeros(3, dtype=torch.float64)
        ei[i] = step_log2
        fp = evaluate(center + ei) * scale
        fm = evaluate(center - ei) * scale
        hessian[i, i] = (fp - 2.0 * f0 + fm) / (step_log2 * step_log2)
        for j in range(i + 1, 3):
            ej = torch.zeros(3, dtype=torch.float64)
            ej[j] = step_log2
            fpp = evaluate(center + ei + ej) * scale
            fpm = evaluate(center + ei - ej) * scale
            fmp = evaluate(center - ei + ej) * scale
            fmm = evaluate(center - ei - ej) * scale
            value = (fpp - fpm - fmp + fmm) / (4.0 * step_log2 * step_log2)
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def finite_difference_hessian_batched(
    evaluate_many: Callable[[torch.Tensor], torch.Tensor],
    center_theta: torch.Tensor,
    *,
    step_log2: float,
    scale: float = 1.0,
) -> torch.Tensor:
    if step_log2 <= 0.0:
        raise ValueError("step_log2 must be positive")
    center = center_theta.detach().cpu().reshape(3).to(torch.float64)
    probes: list[torch.Tensor] = [center]
    diagonal_indices: list[tuple[int, int, int]] = []
    mixed_indices: list[tuple[int, int, int, int, int]] = []

    for i in range(3):
        ei = torch.zeros(3, dtype=torch.float64)
        ei[i] = step_log2
        start = len(probes)
        probes.extend([center + ei, center - ei])
        diagonal_indices.append((i, start, start + 1))
        for j in range(i + 1, 3):
            ej = torch.zeros(3, dtype=torch.float64)
            ej[j] = step_log2
            start = len(probes)
            probes.extend(
                [
                    center + ei + ej,
                    center + ei - ej,
                    center - ei + ej,
                    center - ei - ej,
                ]
            )
            mixed_indices.append((i, j, start, start + 1, start + 2))

    values = evaluate_many(torch.stack(probes)).detach().cpu().to(torch.float64) * scale
    if int(values.numel()) != len(probes):
        raise RuntimeError("batched Hessian evaluator returned the wrong value count")

    f0 = values[0]
    hessian = torch.zeros((3, 3), dtype=torch.float64)
    for i, plus_idx, minus_idx in diagonal_indices:
        hessian[i, i] = (
            values[plus_idx] - 2.0 * f0 + values[minus_idx]
        ) / (step_log2 * step_log2)
    for i, j, fpp_idx, fpm_idx, fmp_idx in mixed_indices:
        fmm_idx = fmp_idx + 1
        value = (
            values[fpp_idx] - values[fpm_idx] - values[fmp_idx] + values[fmm_idx]
        ) / (4.0 * step_log2 * step_log2)
        hessian[i, j] = value
        hessian[j, i] = value
    return hessian


def evaluate_landscape(
    evaluator: BatchedFamilyProbeEvaluator,
    *,
    selection: FamilySelection,
    center_theta: torch.Tensor,
    grid_points: int,
    radius_log2: float,
    hessian_step_log2: float,
    trace: list[OptimizationTracePoint],
) -> FamilyLandscape:
    center_theta = center_theta.detach().reshape(3).cpu().to(torch.float64)
    center_nll = float(evaluator.evaluate(center_theta.reshape(1, 3))[0])
    offsets = torch.linspace(
        -radius_log2,
        radius_log2,
        steps=grid_points,
        dtype=torch.float64,
    )

    probes: list[torch.Tensor] = []
    probe_specs: list[tuple[tuple[str, str], float, float]] = []
    for plane in PLANES:
        x_col = RATE_TO_COL[plane[0]]
        y_col = RATE_TO_COL[plane[1]]
        for x_offset in offsets.tolist():
            for y_offset in offsets.tolist():
                probe = center_theta.clone()
                probe[x_col] += float(x_offset)
                probe[y_col] += float(y_offset)
                probes.append(probe)
                probe_specs.append((plane, float(x_offset), float(y_offset)))

    losses = evaluator.evaluate(torch.stack(probes))
    if int(losses.numel()) != len(probes):
        raise RuntimeError("landscape evaluator returned the wrong value count")

    points: list[LandscapePoint] = []
    for probe, loss, (plane, x_offset, y_offset) in zip(probes, losses, probe_specs):
        points.append(
            _point_from_probe(
                selection=selection,
                plane=plane,
                x_offset=x_offset,
                y_offset=y_offset,
                nll_bits=float(loss),
                center_nll_bits=center_nll,
                probe_theta=probe,
            )
        )

    hessian = finite_difference_hessian_batched(
        evaluator.evaluate,
        center_theta,
        step_log2=hessian_step_log2,
        scale=1.0 / float(selection.leaf_count),
    )
    eigenvalues = torch.linalg.eigvalsh(hessian)
    return FamilyLandscape(
        selection=selection,
        center_theta=center_theta,
        center_nll_bits=center_nll,
        points=points,
        hessian=hessian,
        hessian_eigenvalues=eigenvalues.detach().cpu(),
        trace=trace,
    )


def contour_levels(landscapes: Sequence[FamilyLandscape], *, count: int = 18):
    values = [
        point.delta_nll_bits_per_leaf
        for landscape in landscapes
        for point in landscape.points
        if math.isfinite(point.delta_nll_bits_per_leaf)
    ]
    if not values:
        return None
    lo = min(0.0, min(values))
    hi = max(values)
    if hi <= lo:
        hi = lo + 1.0
    import numpy as np

    return np.linspace(lo, hi, count)


def plot_family_landscape(
    landscape: FamilyLandscape,
    *,
    out_dir: Path,
    levels,
    dpi: int,
    write_pdf: bool,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    family_dir = out_dir / "plots"
    family_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        len(PLANES),
        figsize=(15, 4.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(PLANES) == 1:
        axes = [axes]

    center = landscape.center_theta.detach().cpu()
    trace_by_col = {
        name: [getattr(point, f"theta_{name}") - float(center[col]) for point in landscape.trace]
        for name, col in RATE_FIELDS
    }

    last_contour = None
    for ax, plane in zip(axes, PLANES):
        plane_name = f"{plane[0]}-{plane[1]}"
        rows = [point for point in landscape.points if point.plane == plane_name]
        xs = sorted({point.x_offset_log2 for point in rows})
        ys = sorted({point.y_offset_log2 for point in rows})
        x_index = {value: idx for idx, value in enumerate(xs)}
        y_index = {value: idx for idx, value in enumerate(ys)}
        z = np.full((len(ys), len(xs)), np.nan, dtype=float)
        for point in rows:
            z[y_index[point.y_offset_log2], x_index[point.x_offset_log2]] = (
                point.delta_nll_bits_per_leaf
            )
        x_grid, y_grid = np.meshgrid(xs, ys)
        last_contour = ax.contourf(x_grid, y_grid, z, levels=levels, cmap="viridis")
        ax.contour(x_grid, y_grid, z, levels=levels, colors="black", linewidths=0.25, alpha=0.35)
        ax.scatter([0.0], [0.0], marker="+", color="white", s=85, linewidths=1.6)
        if landscape.trace:
            ax.plot(
                trace_by_col[plane[0]],
                trace_by_col[plane[1]],
                color="white",
                lw=1.0,
                alpha=0.85,
            )
            ax.scatter(
                [trace_by_col[plane[0]][0]],
                [trace_by_col[plane[1]][0]],
                color="white",
                edgecolor="black",
                s=18,
                zorder=4,
            )
        ax.set_title(plane_name)
        ax.set_xlabel(f"{plane[0]} log2 fold-change")
        ax.set_ylabel(f"{plane[1]} log2 fold-change")
        ax.grid(True, color="white", alpha=0.12, linewidth=0.5)

    if last_contour is not None:
        cbar = fig.colorbar(last_contour, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("Delta NLL bits per leaf")
    fig.suptitle(
        f"{landscape.selection.name} "
        f"(family {landscape.selection.index}, leaves={landscape.selection.leaf_count})"
    )
    safe_name = landscape.selection.name.replace("/", "_")
    paths = [family_dir / f"{landscape.selection.index:04d}_{safe_name}_landscape.png"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    if write_pdf:
        pdf_path = family_dir / f"{landscape.selection.index:04d}_{safe_name}_landscape.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        paths.append(pdf_path)
    plt.close(fig)
    return paths


def write_landscape_values(path: Path, landscapes: Sequence[FamilyLandscape]) -> None:
    fieldnames = list(LandscapePoint.__dataclass_fields__)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for landscape in landscapes:
            for point in landscape.points:
                writer.writerow(point.__dict__)


def write_optimization_trace(path: Path, landscapes: Sequence[FamilyLandscape]) -> None:
    rows = [point for landscape in landscapes for point in landscape.trace]
    if not rows:
        return
    fieldnames = list(OptimizationTracePoint.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _csv_float_list(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def write_family_summary(path: Path, landscapes: Sequence[FamilyLandscape]) -> None:
    fields = [
        "family_index",
        "family_name",
        "leaf_count",
        "center_nll_bits",
        "center_nll_bits_per_leaf",
        "theta_D",
        "theta_L",
        "theta_T",
        "rate_D",
        "rate_L",
        "rate_T",
        "min_delta_nll_bits_per_leaf",
        "max_delta_nll_bits_per_leaf",
        "hessian_eig_min",
        "hessian_eig_max",
        "hessian_negative_curvature_ratio",
        "hessian_condition_abs",
        "hessian_eigenvalues",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for landscape in landscapes:
            values = [point.delta_nll_bits_per_leaf for point in landscape.points]
            eig = landscape.hessian_eigenvalues.to(torch.float64)
            eig_min = float(eig.min())
            eig_max = float(eig.max())
            max_abs = float(eig.abs().max())
            min_abs = float(eig.abs().min())
            theta = landscape.center_theta
            rate_D, rate_L, rate_T = theta_rates(theta)
            writer.writerow(
                {
                    "family_index": landscape.selection.index,
                    "family_name": landscape.selection.name,
                    "leaf_count": landscape.selection.leaf_count,
                    "center_nll_bits": landscape.center_nll_bits,
                    "center_nll_bits_per_leaf": (
                        landscape.center_nll_bits / float(landscape.selection.leaf_count)
                    ),
                    "theta_D": float(theta[0]),
                    "theta_L": float(theta[1]),
                    "theta_T": float(theta[2]),
                    "rate_D": rate_D,
                    "rate_L": rate_L,
                    "rate_T": rate_T,
                    "min_delta_nll_bits_per_leaf": min(values),
                    "max_delta_nll_bits_per_leaf": max(values),
                    "hessian_eig_min": eig_min,
                    "hessian_eig_max": eig_max,
                    "hessian_negative_curvature_ratio": (
                        max(0.0, -eig_min) / max(abs(eig_max), 1e-12)
                    ),
                    "hessian_condition_abs": max_abs / max(min_abs, 1e-12),
                    "hessian_eigenvalues": _csv_float_list(eig.tolist()),
                }
            )


def build_family_model(
    *,
    species_tree: Path,
    families_file: Path,
    selection: FamilySelection,
    device: str,
    dtype: torch.dtype,
    theta_init_rates: tuple[float, float, float],
    fixed_iters_E: int,
    fixed_iters_Pi: int,
    neumann_terms: int,
    max_wave_size: int,
    preprocess_cpu_cores: int | None,
) -> GeneReconModel:
    return GeneReconModel.from_alerax_families(
        str(species_tree),
        families_file,
        mode="global",
        start=selection.index,
        max_families=1,
        device=device,
        dtype=dtype,
        theta_init_rates=theta_init_rates,
        fixed_iters_E=fixed_iters_E,
        fixed_iters_Pi=fixed_iters_Pi,
        neumann_terms=neumann_terms,
        max_wave_size=max_wave_size,
        preprocess_cpu_cores=preprocess_cpu_cores,
    )


def run(args: argparse.Namespace) -> list[FamilyLandscape]:
    names, tree_paths, leaf_maps = read_family_metadata(args.families_file)
    selections = resolve_family_selection(
        names=names,
        tree_paths=tree_paths,
        leaf_maps=leaf_maps,
        family_indices=args.family_indices,
        family_names=args.family_names,
        representative_count=args.representative_count,
    )
    if not selections:
        raise RuntimeError("no families selected")

    anchor_theta = load_anchor_theta(args.theta_final) if args.theta_final else None
    landscapes: list[FamilyLandscape] = []
    for selection in selections:
        print(
            f"[{len(landscapes) + 1}/{len(selections)}] {selection.name} "
            f"(family {selection.index}, leaves={selection.leaf_count})",
            flush=True,
        )
        trace: list[OptimizationTracePoint] = []
        if anchor_theta is not None:
            center = anchor_for_family(
                anchor_theta,
                family_index=selection.index,
                family_count=len(names),
            )
        else:
            model = build_family_model(
                species_tree=args.species_tree,
                families_file=args.families_file,
                selection=selection,
                device=args.device,
                dtype=dtype_from_name(args.dtype),
                theta_init_rates=args.theta_init_rates,
                fixed_iters_E=args.fixed_iters_e,
                fixed_iters_Pi=args.fixed_iters_pi,
                neumann_terms=args.neumann_terms,
                max_wave_size=args.max_wave_size,
                preprocess_cpu_cores=args.preprocess_cpu_cores,
            )
            try:
                trace = optimize_family_anchor(
                    model,
                    selection=selection,
                    steps=args.optimize_steps,
                    lr=args.optimize_lr,
                    min_rate=args.min_rate,
                    max_rate=args.max_rate,
                )
                center = model.theta.detach().reshape(3).cpu().to(torch.float64)
            finally:
                model.close()

        evaluator = BatchedFamilyProbeEvaluator(
            species_tree=args.species_tree,
            family_name=selection.name,
            tree_paths=tree_paths[selection.index],
            leaf_species_map=leaf_maps[selection.index],
            batch_size=args.theta_batch_size,
            device=args.device,
            dtype=dtype_from_name(args.dtype),
            fixed_iters_E=args.fixed_iters_e,
            fixed_iters_Pi=args.fixed_iters_pi,
            neumann_terms=args.neumann_terms,
            max_wave_size=args.max_wave_size,
            preprocess_cpu_cores=args.preprocess_cpu_cores,
        )
        try:
            landscapes.append(
                evaluate_landscape(
                    evaluator,
                    selection=selection,
                    center_theta=center,
                    grid_points=args.grid_points,
                    radius_log2=args.radius_log2,
                    hessian_step_log2=args.hessian_step_log2,
                    trace=trace,
                )
            )
        finally:
            evaluator.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_landscape_values(args.out_dir / "landscape_values.csv", landscapes)
    write_family_summary(args.out_dir / "family_landscape_summary.csv", landscapes)
    write_optimization_trace(args.out_dir / "optimization_trace.csv", landscapes)

    levels = contour_levels(landscapes)
    plot_paths: list[str] = []
    for landscape in landscapes:
        plot_paths.extend(
            str(path)
            for path in plot_family_landscape(
                landscape,
                out_dir=args.out_dir,
                levels=levels,
                dpi=args.dpi,
                write_pdf=args.write_pdf,
            )
        )

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["families"] = [selection.__dict__ for selection in selections]
    config["plots"] = plot_paths
    (args.out_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return landscapes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize gpurec HOGENOM family loss landscapes on D/L/T "
            "log2 fold-change planes."
        )
    )
    parser.add_argument("--species-tree", type=Path, default=SPECIES_TREE)
    parser.add_argument("--families-file", type=Path, default=FAMILIES_FILE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--family-indices", type=int, nargs="+", default=None)
    parser.add_argument("--family-names", nargs="+", default=None)
    parser.add_argument("--representative-count", type=int, default=12)
    parser.add_argument(
        "--theta-final",
        type=Path,
        default=None,
        help="Optional global [3]/[1,3] or genewise [G,3] theta tensor/checkpoint.",
    )
    parser.add_argument("--grid-points", type=validate_grid_points, default=51)
    parser.add_argument("--radius-log2", type=float, default=2.0)
    parser.add_argument("--hessian-step-log2", type=float, default=0.05)
    parser.add_argument(
        "--theta-batch-size",
        type=positive_int_arg("theta-batch-size"),
        default=64,
        help=(
            "Number of theta probes evaluated per synthetic duplicated-family "
            "genewise pass."
        ),
    )
    parser.add_argument("--optimize-steps", type=int, default=200)
    parser.add_argument("--optimize-lr", type=float, default=0.03)
    parser.add_argument(
        "--theta-init-rates",
        type=parse_rate_triple,
        default=(0.05, 0.05, 0.05),
        help="Initial D,L,T natural rates for local optimization.",
    )
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--fixed-iters-e", type=int, default=16)
    parser.add_argument("--fixed-iters-pi", type=int, default=6)
    parser.add_argument("--neumann-terms", type=int, default=3)
    parser.add_argument("--max-wave-size", type=int, default=32768)
    parser.add_argument("--preprocess-cpu-cores", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--write-pdf", dest="write_pdf", action="store_true", default=True)
    parser.add_argument("--no-pdf", dest="write_pdf", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
