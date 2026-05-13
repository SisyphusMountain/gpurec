from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpurec import GeneReconModel  # noqa: E402
from gpurec.core.preprocess_cpp import _load_extension  # noqa: E402


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
OUT_DIR = HOGENOM_DIR / "profile_hogenom_ccp_pass"

FIXED_ITERS_E = 6
FIXED_ITERS_PI = 6
NEUMANN_TERMS = 6
INITIAL_RATES = (0.05, 0.05, 0.05)


class DatasetConfig:
    def __init__(
        self,
        *,
        species_tree: Path,
        families_file: Path,
        preprocess_cache: Path,
        device: str,
        dtype: torch.dtype,
        max_families: int | None,
        family_chunk_size: int,
        clade_budget: int | None,
        batch_packing: str,
        fixed_iters_E: int,
        fixed_iters_Pi: int,
        neumann_terms: int,
        use_pruning: bool,
    ) -> None:
        self.species_tree = species_tree
        self.families_file = families_file
        self.preprocess_cache = preprocess_cache
        self.device = device
        self.dtype = dtype
        self.max_families = max_families
        self.family_chunk_size = family_chunk_size
        self.clade_budget = clade_budget
        self.batch_packing = batch_packing
        self.fixed_iters_E = fixed_iters_E
        self.fixed_iters_Pi = fixed_iters_Pi
        self.neumann_terms = neumann_terms
        self.use_pruning = use_pruning


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_profiler_start(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        synchronize()
        torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        synchronize()
        torch.cuda.cudart().cudaProfilerStop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile one HOGENOM CCP forward/backward pass."
    )
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument(
        "--batch-packing",
        choices=("sequential", "clade_first_fit"),
        default="sequential",
        help="Resident batch packing policy.",
    )
    parser.add_argument(
        "--clade-budget",
        type=int,
        default=None,
        help=(
            "Optional sequential resident-batch clade budget. Use "
            "--chunk-size 0 to make the clade budget the only batch cap."
        ),
    )
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--profile-runs", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=("stream-batches", "full", "active-batch", "largest-batch"),
        default="stream-batches",
        help=(
            "stream-batches profiles model()/backward() over every resident "
            "batch; full profiles full_loss(), whose backward is intentionally "
            "cheap because the streamed gradient is computed inside full_loss()."
        ),
    )
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument(
        "--cuda-profiler-api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bracket measured passes with cudaProfilerStart/Stop.",
    )
    parser.add_argument(
        "--pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable backward active-row pruning.",
    )
    return parser.parse_args()


def dataset_config(args: argparse.Namespace) -> DatasetConfig:
    return DatasetConfig(
        species_tree=SPECIES_TREE,
        families_file=FAMILIES_FILE,
        preprocess_cache=PREPROCESS_CACHE,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        max_families=args.max_families,
        family_chunk_size=args.chunk_size,
        clade_budget=args.clade_budget,
        batch_packing=args.batch_packing,
        fixed_iters_E=FIXED_ITERS_E,
        fixed_iters_Pi=FIXED_ITERS_PI,
        neumann_terms=NEUMANN_TERMS,
        use_pruning=args.pruning,
    )


def load_species_names(species_tree: Path) -> list[str]:
    ext = _load_extension()
    raw = ext.preprocess_multiple_families(
        str(species_tree),
        {},
        include_species_matrices=False,
    )
    return [str(x) for x in raw["species"]["names"]]


def uniform_origination_probs(
    n_species: int,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.full((n_species,), 1.0 / n_species, device=device, dtype=dtype)


def build_model(
    config: DatasetConfig,
    origination_probs: torch.Tensor,
) -> GeneReconModel:
    return GeneReconModel.from_alerax_families(
        str(config.species_tree),
        config.families_file,
        mode="specieswise",
        start=0,
        max_families=config.max_families,
        device=config.device,
        dtype=config.dtype,
        theta_init_rates=INITIAL_RATES,
        preprocess_cache_dir=config.preprocess_cache,
        fixed_iters_E=config.fixed_iters_E,
        fixed_iters_Pi=config.fixed_iters_Pi,
        neumann_terms=config.neumann_terms,
        use_pruning=config.use_pruning,
        family_chunk_size=config.family_chunk_size,
        clade_budget=config.clade_budget,
        batch_packing=config.batch_packing,
        lazy_preprocess=True,
        prefetch_batches="all",
        origination_probs=origination_probs,
    )


def activate_batch(model, batch_index: int) -> None:
    model.clear()
    model._current_batch_index = batch_index
    model._ensure_batch_static(batch_index)
    model._schedule_prefetch()


def choose_batch_index(model, mode: str, batch_index: int) -> int:
    if mode == "active-batch":
        if batch_index < 0 or batch_index >= len(model.batch_metadata):
            raise ValueError(
                f"batch index {batch_index} outside [0, {len(model.batch_metadata)})"
            )
        return batch_index
    if mode == "largest-batch":
        return max(
            range(len(model.batch_metadata)),
            key=lambda idx: model.batch_metadata[idx].clade_count,
        )
    raise ValueError(f"batch selection is not valid in mode={mode!r}")


def run_pass(model, mode: str, batch_index: int | None) -> dict[str, float | int | str]:
    model.theta.grad = None
    model.clear()

    if mode == "stream-batches":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        synchronize()
        t0 = time.perf_counter()
        total_loss = model.theta.new_zeros(())
        event_rows = []
        for idx in range(len(model.batch_metadata)):
            activate_batch(model, idx)
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)
            with nvtx_range(f"gpurec_profile.stream_batches.forward.batch_{idx:04d}"):
                fwd_start.record()
                loss_i = model()
                fwd_end.record()
            with nvtx_range(f"gpurec_profile.stream_batches.backward.batch_{idx:04d}"):
                loss_i.backward()
                bwd_end.record()
            total_loss = total_loss + loss_i.detach()
            event_rows.append((fwd_start, fwd_end, bwd_end))
        synchronize()
        t1 = time.perf_counter()
        grad = model.theta.grad.detach()
        forward_s = sum(
            fwd_start.elapsed_time(fwd_end) for fwd_start, fwd_end, _bwd_end in event_rows
        ) / 1000.0
        backward_s = sum(
            fwd_end.elapsed_time(bwd_end) for _fwd_start, fwd_end, bwd_end in event_rows
        ) / 1000.0
        result: dict[str, float | int | str] = {
            "mode": mode,
            "loss_bits": float(total_loss.detach().cpu()),
            "forward_s": forward_s,
            "backward_s": backward_s,
            "forward_backward_s": forward_s + backward_s,
            "wall_s": t1 - t0,
            "grad_inf": float(grad.abs().amax().cpu()),
            "grad_norm": float(torch.linalg.vector_norm(grad).cpu()),
            "streamed_batches": len(model.batch_metadata),
        }
        if torch.cuda.is_available():
            result.update(
                {
                    "peak_allocated_gib": torch.cuda.max_memory_allocated()
                    / (1024.0**3),
                    "peak_reserved_gib": torch.cuda.max_memory_reserved()
                    / (1024.0**3),
                }
            )
        return result

    if mode != "full":
        assert batch_index is not None
        activate_batch(model, batch_index)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    synchronize()

    t0 = time.perf_counter()
    with nvtx_range(f"gpurec_profile.{mode}.forward"):
        loss = model.full_loss() if mode == "full" else model()
    synchronize()
    t1 = time.perf_counter()

    with nvtx_range(f"gpurec_profile.{mode}.backward"):
        loss.backward()
    synchronize()
    t2 = time.perf_counter()

    grad = model.theta.grad.detach()
    result: dict[str, float | int | str] = {
        "mode": mode,
        "loss_bits": float(loss.detach().cpu()),
        "forward_s": t1 - t0,
        "backward_s": t2 - t1,
        "forward_backward_s": t2 - t0,
        "grad_inf": float(grad.abs().amax().cpu()),
        "grad_norm": float(torch.linalg.vector_norm(grad).cpu()),
    }
    if batch_index is not None:
        meta = model.batch_metadata[batch_index]
        result.update(
            {
                "batch_index": batch_index,
                "batch_families": meta.family_count,
                "batch_clades": meta.clade_count,
                "batch_splits": meta.split_count,
                "batch_waves": meta.wave_count,
                "batch_max_wave_size": meta.max_wave_size,
            }
        )
    if torch.cuda.is_available():
        result.update(
            {
                "peak_allocated_gib": torch.cuda.max_memory_allocated()
                / (1024.0**3),
                "peak_reserved_gib": torch.cuda.max_memory_reserved()
                / (1024.0**3),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Nsight profiling")
    if not SPECIES_TREE.exists():
        raise FileNotFoundError(SPECIES_TREE)
    if not FAMILIES_FILE.exists():
        raise FileNotFoundError(FAMILIES_FILE)

    config = dataset_config(args)
    print(
        json.dumps(
            {
                "event": "config",
                "species_tree": str(config.species_tree),
                "families_file": str(config.families_file),
                "device": config.device,
                "dtype": str(config.dtype),
                "chunk_size": config.family_chunk_size,
                "clade_budget": config.clade_budget,
                "batch_packing": config.batch_packing,
                "max_families": config.max_families,
                "fixed_iters_E": config.fixed_iters_E,
                "fixed_iters_Pi": config.fixed_iters_Pi,
                "neumann_terms": config.neumann_terms,
                "use_pruning": config.use_pruning,
                "mode": args.mode,
            }
        ),
        flush=True,
    )

    with nvtx_range("gpurec_profile.build_model"):
        species_names = load_species_names(config.species_tree)
        origination_probs = uniform_origination_probs(
            len(species_names),
            device=config.device,
            dtype=config.dtype,
        )
        model = build_model(config, origination_probs)

    batch_index: int | None = None
    if args.mode not in ("full", "stream-batches"):
        batch_index = choose_batch_index(model, args.mode, args.batch_index)

    print(
        json.dumps(
            {
                "event": "model",
                "families": sum(meta.family_count for meta in model.batch_metadata),
                "species": model.n_species,
                "batches": len(model.batch_metadata),
                "chunk_size": model.family_chunk_size,
                "selected_batch": batch_index,
            }
        ),
        flush=True,
    )

    for warmup_idx in range(args.warmup_runs):
        with torch.enable_grad(), nvtx_range("gpurec_profile.warmup"):
            row = run_pass(model, args.mode, batch_index)
        print(json.dumps({"event": "warmup", "idx": warmup_idx, **row}), flush=True)

    measured: list[dict[str, float | int | str]] = []
    cuda_profiler_start(args.cuda_profiler_api)
    try:
        for profile_idx in range(args.profile_runs):
            with torch.enable_grad(), nvtx_range("gpurec_profile.measured_pass"):
                row = run_pass(model, args.mode, batch_index)
            measured.append(row)
            print(
                json.dumps({"event": "measured", "idx": profile_idx, **row}),
                flush=True,
            )
    finally:
        cuda_profiler_stop(args.cuda_profiler_api)
        model.close()

    if measured:
        print(
            json.dumps(
                {
                    "event": "summary",
                    "profile_runs": len(measured),
                    "median_forward_s": float(
                        torch.median(
                            torch.tensor([row["forward_s"] for row in measured])
                        )
                    ),
                    "median_backward_s": float(
                        torch.median(
                            torch.tensor([row["backward_s"] for row in measured])
                        )
                    ),
                    "median_forward_backward_s": float(
                        torch.median(
                            torch.tensor(
                                [row["forward_backward_s"] for row in measured]
                            )
                        )
                    ),
                    "max_peak_allocated_gib": max(
                        float(row.get("peak_allocated_gib", math.nan))
                        for row in measured
                    ),
                    "max_peak_reserved_gib": max(
                        float(row.get("peak_reserved_gib", math.nan))
                        for row in measured
                    ),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
