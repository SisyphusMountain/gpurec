from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch

from gpurec.core.batch_planning import (
    normalize_batch_packing as _normalize_batch_packing,
    normalize_family_chunk_size as _normalize_family_chunk_size,
)


def _default_device() -> str:
    return "cuda"


def dtype_from_name(name: str) -> torch.dtype:
    text = str(name).lower().replace("torch.", "")
    if text in {"float32", "fp32", "single"}:
        return torch.float32
    if text in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError(f"unsupported dtype {name!r}; expected float32 or float64")


def _normalize_optional_positive_int(name: str, value: int | str | None) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive when provided")
    return number


def _finite_float(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    return value


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


@dataclass
class RunConfig:
    species_tree: Path
    families_file: Path
    out_dir: Path
    mode: str = "genewise"
    device: str = ""
    dtype: str = "float32"

    start: int = 0
    max_families: int | None = None
    preprocess_cache: Path | None = None
    refresh_preprocess_cache: bool = False

    family_chunk_size: int | str | None = 0
    clade_budget: int | None = 305_000
    batch_packing: str = "depth_first_fit"
    max_wave_size: int | None = 8192

    fixed_iters_e: int | None = None
    max_iters_e: int = 2000
    tol_e: float = 1e-8
    fixed_iters_pi: int = 64
    neumann_terms: int = 64
    adaptive_iters: bool = True
    convergence_check_interval: int = 4
    e_logsumexp_tol: float = 1e-5
    pi_max_diff_tol: float = 1e-5
    gradient_change_tol: float = 1e-4
    gradient_change_rtol: float = 1e-4

    theta_init_d: float = 0.05
    theta_init_l: float = 0.05
    theta_init_t: float = 0.05
    min_rate: float = 1e-10
    max_rate: float = 1e9

    optimizer: str = "adam"
    steps: int = 5000
    lr: float = 0.01
    adam_warmup_steps: int = 100
    lbfgs_lr: float = 0.1
    lbfgs_history_size: int = 20
    lbfgs_max_iter: int = 1
    lbfgs_line_search: str = "none"

    grad_inf_tol: float = 1e-3
    loss_change_tol: float = 1e-5
    loss_patience: int = 20
    best_likelihood_patience: int = 20
    best_likelihood_min_delta: float = 0.0

    checkpoint_every: int = 1
    log_every: int = 1
    resume_from: Path | None = None

    def __post_init__(self) -> None:
        self.species_tree = _resolve_path(self.species_tree)
        self.families_file = _resolve_path(self.families_file)
        self.out_dir = _resolve_path(self.out_dir)
        if self.preprocess_cache is not None:
            self.preprocess_cache = _resolve_path(self.preprocess_cache)
        if self.resume_from is not None:
            self.resume_from = _resolve_path(self.resume_from)
        self.family_chunk_size = int(_normalize_family_chunk_size(self.family_chunk_size))
        self.clade_budget = _normalize_optional_positive_int(
            "clade_budget",
            self.clade_budget,
        )
        self.max_wave_size = _normalize_optional_positive_int(
            "max_wave_size",
            self.max_wave_size,
        )
        self.batch_packing = _normalize_batch_packing(self.batch_packing)
        if not self.device:
            self.device = _default_device()
        self.validate()

    def validate(self) -> None:
        if self.mode not in {"global", "specieswise", "genewise"}:
            raise ValueError("mode must be 'global', 'specieswise', or 'genewise'")
        dtype_from_name(self.dtype)
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.max_families is not None and self.max_families < 1:
            raise ValueError("max_families must be positive when provided")
        if self.batch_packing != "sequential" and self.clade_budget is None:
            raise ValueError(f"batch_packing={self.batch_packing!r} requires clade_budget")
        if self.fixed_iters_e is not None and self.fixed_iters_e < 1:
            raise ValueError("fixed_iters_e must be positive when provided")
        if self.max_iters_e < 1:
            raise ValueError("max_iters_e must be positive")
        if self.fixed_iters_pi < 1 or self.fixed_iters_pi % 2 != 0:
            raise ValueError("fixed_iters_pi must be a positive even integer")
        if self.neumann_terms < 1:
            raise ValueError("neumann_terms must be positive")
        if self.convergence_check_interval < 1:
            raise ValueError("convergence_check_interval must be positive")
        if self.adaptive_iters and self.convergence_check_interval % 2 != 0:
            raise ValueError("adaptive convergence_check_interval must be even")
        for name in (
            "tol_e",
            "e_logsumexp_tol",
            "pi_max_diff_tol",
            "gradient_change_tol",
            "gradient_change_rtol",
            "grad_inf_tol",
            "loss_change_tol",
            "best_likelihood_min_delta",
        ):
            if _finite_float(name, getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        min_rate = _finite_float("min_rate", self.min_rate)
        max_rate = _finite_float("max_rate", self.max_rate)
        if min_rate <= 0.0 or max_rate <= min_rate:
            raise ValueError("rate bounds must satisfy 0 < min_rate < max_rate")
        theta_init_d = _finite_float("theta_init_d", self.theta_init_d)
        theta_init_l = _finite_float("theta_init_l", self.theta_init_l)
        theta_init_t = _finite_float("theta_init_t", self.theta_init_t)
        if (
            theta_init_d <= 0.0
            or theta_init_l <= 0.0
            or theta_init_t <= 0.0
        ):
            raise ValueError("theta_init_d/l/t must be strictly positive")
        if self.optimizer not in {"adam", "adagrad", "lbfgs", "adam-lbfgs"}:
            raise ValueError("optimizer must be adam, adagrad, lbfgs, or adam-lbfgs")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        lr = _finite_float("lr", self.lr)
        lbfgs_lr = _finite_float("lbfgs_lr", self.lbfgs_lr)
        if lr <= 0.0 or lbfgs_lr <= 0.0:
            raise ValueError("optimizer learning rates must be positive")
        if self.adam_warmup_steps < 0:
            raise ValueError("adam_warmup_steps must be non-negative")
        if self.lbfgs_history_size < 1 or self.lbfgs_max_iter < 1:
            raise ValueError("LBFGS history size and max_iter must be positive")
        if self.lbfgs_line_search not in {"none", "strong_wolfe"}:
            raise ValueError("lbfgs_line_search must be 'none' or 'strong_wolfe'")
        if self.loss_patience < 0 or self.best_likelihood_patience < 0:
            raise ValueError("patience values must be non-negative")
        if self.checkpoint_every < 0 or self.log_every < 1:
            raise ValueError("checkpoint_every must be >= 0 and log_every must be positive")

    @property
    def torch_dtype(self) -> torch.dtype:
        return dtype_from_name(self.dtype)

    @property
    def theta_init_rates(self) -> tuple[float, float, float]:
        return (self.theta_init_d, self.theta_init_l, self.theta_init_t)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        if not isinstance(data, dict):
            raise ValueError("RunConfig data must be a JSON object")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(str(key) for key in data if key not in allowed)
        if unknown:
            raise ValueError(f"unknown RunConfig field(s): {', '.join(unknown)}")
        return cls(**dict(data))

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@dataclass
class SamplingConfig:
    checkpoint: Path
    out_dir: Path | None = None
    samples: int = 100
    seed: int = 0
    family_start: int = 0
    max_families: int | None = None
    max_events: int | None = 100_000
    backtrack_binary: Path | None = None

    def __post_init__(self) -> None:
        self.checkpoint = _resolve_path(self.checkpoint)
        if self.out_dir is not None:
            self.out_dir = _resolve_path(self.out_dir)
        if self.backtrack_binary is not None:
            self.backtrack_binary = _resolve_path(self.backtrack_binary)
        if self.samples < 1:
            raise ValueError("samples must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.family_start < 0:
            raise ValueError("family_start must be non-negative")
        if self.max_families is not None and self.max_families < 1:
            raise ValueError("max_families must be positive when provided")
        if self.max_events is not None and self.max_events < 1:
            raise ValueError("max_events must be positive when provided")
