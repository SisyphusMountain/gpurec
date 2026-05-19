from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from gpurec.core.batch_planning import normalize_batch_packing as _normalize_batch_packing


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype_from_name(name: str) -> torch.dtype:
    text = str(name).lower().replace("torch.", "")
    if text in {"float32", "fp32", "single"}:
        return torch.float32
    if text in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError(f"unsupported dtype {name!r}; expected float32 or float64")


def _normalize_family_chunk_size(value: int | str | None) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "all", "none", "null"}:
            return 0
        if text == "auto":
            raise ValueError(
                "family_chunk_size='auto' is not supported by gpurec.workflow; "
                "use 0 for one resident batch or a positive integer"
            )
        value = int(text)
    size = int(value)
    if size < 0:
        raise ValueError("family_chunk_size must be non-negative")
    return size


def _normalize_optional_positive_int(name: str, value: int | str | None) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive when provided")
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
        self.family_chunk_size = _normalize_family_chunk_size(self.family_chunk_size)
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
        if self.min_rate <= 0.0 or self.max_rate <= self.min_rate:
            raise ValueError("rate bounds must satisfy 0 < min_rate < max_rate")
        if self.optimizer not in {"adam", "adagrad", "lbfgs", "adam-lbfgs"}:
            raise ValueError("optimizer must be adam, adagrad, lbfgs, or adam-lbfgs")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.lr <= 0.0 or self.lbfgs_lr <= 0.0:
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
        if self.family_start < 0:
            raise ValueError("family_start must be non-negative")
        if self.max_families is not None and self.max_families < 1:
            raise ValueError("max_families must be positive when provided")
