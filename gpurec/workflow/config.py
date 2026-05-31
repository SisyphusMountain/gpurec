from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch

from . import _route_defaults
from ._route_defaults import (
    DEFAULT_ADAGRAD_RESTART_FINAL_CHECK_ITERS,
    DEFAULT_ADAGRAD_RESTART_SCHEDULE,
    DEFAULT_CLADE_BUDGET,
    SpecieswiseRouteDefaults,
)
from ._schedules import (
    DEFAULT_ADAGRAD_RESTART_TOTAL_STEPS,
    DEFAULT_NORMALIZED_ADAGRAD_RESTART_SCHEDULE,
    AdagradRestartPhase as AdagradRestartPhase,
    LossStopPhase as LossStopPhase,
    _normalize_adagrad_restart_schedule,
    _normalize_optional_loss_stop_schedule,
    adagrad_restart_schedule_specs as adagrad_restart_schedule_specs,
    adagrad_restart_schedule_total_steps,
    loss_stop_schedule_specs as loss_stop_schedule_specs,
)
from ._config_io import (
    _JSON_BOOL_FIELDS,
    _JSON_FLOAT_FIELDS,
    _RUN_CONFIG_LEGACY_FIELDS,
    _RUN_CONFIG_REQUIRED_PATH_FIELDS,
    _validate_json_scalar_types,
    load_json_object as load_json_object,
    load_json_object_text as load_json_object_text,
    load_run_config_data,
    load_run_config_text as load_run_config_text,
)
from gpurec._validation import (
    bool_value,
    disabled_adaptive_neumann_terms_value,
    finite_float,
    integer_value,
    nonnegative_int,
    optional_nonnegative_int,
    optional_positive_even_int,
    optional_positive_int,
    positive_even_int,
    positive_int,
)
from gpurec.core.batch_planning import (
    normalize_batch_packing as _normalize_batch_packing,
    normalize_family_chunk_size as _normalize_family_chunk_size,
)


def _default_device() -> str:
    return "cuda"


UINT64_MAX = (1 << 64) - 1
MODE_DEFAULT_OPTIMIZERS = _route_defaults.MODE_DEFAULT_OPTIMIZERS


def dtype_from_name(name: str) -> torch.dtype:
    text = str(name).lower().replace("torch.", "")
    if text in {"float32", "fp32", "single"}:
        return torch.float32
    if text in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError(f"unsupported dtype {name!r}; expected float32 or float64")


def dtype_name_from_name(name: str) -> str:
    return str(dtype_from_name(name)).removeprefix("torch.")


def _normalize_mode(mode: str) -> str:
    return _route_defaults.normalize_mode_name(mode)


def normalize_mode_name(mode: str) -> str:
    return _normalize_mode(mode)


def default_optimizer_for_mode(mode: str) -> str:
    return _route_defaults.default_optimizer_for_mode(mode)


def _normalize_int(name: str, value: int | float | str) -> int:
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    return integer_value(name, value)


def _normalize_positive_int(name: str, value: int | float | str) -> int:
    return positive_int(name, _normalize_int(name, value))


def _normalize_positive_even_int(name: str, value: int | float | str) -> int:
    return positive_even_int(name, _normalize_int(name, value))


def _normalize_nonnegative_int(name: str, value: int | float | str) -> int:
    return nonnegative_int(name, _normalize_int(name, value))


def _normalize_uint64(name: str, value: int | float | str) -> int:
    number = _normalize_nonnegative_int(name, value)
    if number > UINT64_MAX:
        raise ValueError(f"{name} must be <= {UINT64_MAX}")
    return number


def _normalize_optional_positive_int(
    name: str,
    value: int | float | str | None,
) -> int | None:
    return optional_positive_int(
        name,
        None if value is None else _normalize_int(name, value),
    )


def _normalize_optional_positive_even_int(
    name: str,
    value: int | float | str | None,
) -> int | None:
    return optional_positive_even_int(
        name,
        None if value is None else _normalize_int(name, value),
    )


def _normalize_optional_nonnegative_int(
    name: str,
    value: int | float | str | None,
) -> int | None:
    return optional_nonnegative_int(
        name,
        None if value is None else _normalize_int(name, value),
    )


def _normalize_finite_float(name: str, value: float | int | str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError(f"{name} must be a number")
        return finite_float(name, value)
    return finite_float(name, value)


def _normalize_bool(name: str, value: bool) -> bool:
    return bool_value(name, value)


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


def _normalize_path(name: str, value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be a path string")
    return _resolve_path(value)


def _normalize_optional_path(name: str, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return _normalize_path(name, value)


def _normalize_device(value: str | None) -> str:
    if value is None or value == "":
        value = _default_device()
    if not isinstance(value, str):
        raise ValueError("device must be a device string")
    try:
        torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"device must be a valid torch device string: {value!r}"
        ) from exc
    return value


def _normalize_optimizer(mode: str, value: str) -> str:
    return _route_defaults.normalize_optimizer_for_mode(mode, value)


def normalize_optimizer_name(value: str) -> str:
    return _route_defaults.normalize_optimizer_name(value)


def normalize_optimizer_for_mode(mode: str, value: str) -> str:
    return _normalize_optimizer(mode, value)


def effective_optimizer_step_cap(config: RunConfig) -> tuple[int, str]:
    if config.optimizer == "adagrad-restarts":
        schedule_steps = adagrad_restart_schedule_total_steps(
            config.adagrad_restart_schedule
        )
        if schedule_steps <= config.steps:
            return schedule_steps, "adagrad_restart_schedule"
    return config.steps, "configured_steps"


def effective_final_check_iters(config: RunConfig) -> int:
    """Return the solver budget used by final likelihood/gradient validation."""

    if config.optimizer == "adagrad-restarts":
        return int(config.adagrad_restart_final_check_iters)
    return int(config.final_check_iters)


def effective_final_check_iters_e(config: RunConfig) -> int | None:
    """Return the E-solver budget used by final likelihood/gradient validation."""

    check_iters = effective_final_check_iters(config)
    if check_iters <= 0:
        return 0
    if config.optimizer == "adagrad-restarts":
        return int(check_iters)
    if config.mode == "specieswise" and check_iters > 16:
        if config.fixed_iters_e is None:
            return int(check_iters)
        return max(int(config.fixed_iters_e), int(check_iters))
    return None if config.fixed_iters_e is None else int(config.fixed_iters_e)


def _normalize_workflow_batch_packing(value: str | None) -> str:
    if value is None:
        raise ValueError("batch_packing must be provided as a string")
    return _normalize_batch_packing(value)


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
    preprocess_cpu_cores: int | None = None

    family_chunk_size: int | str | None = 0
    clade_budget: int | None = DEFAULT_CLADE_BUDGET
    batch_packing: str = "depth_first_fit"
    max_wave_size: int | None = 8192
    small_family_max_leaves: int = 0

    fixed_iters_e: int | None = None
    max_iters_e: int = 2000
    tol_e: float = 1e-8
    fixed_iters_pi: int = 16
    neumann_terms: int = 16
    solver_warmup_iters: int = 4
    solver_warmup_loss_patience: int = 2
    adaptive_iters: bool = True
    adaptive_neumann_terms: bool = False
    final_check_iters: int = 32
    convergence_check_interval: int = 4
    e_logsumexp_tol: float = 1e-5
    pi_max_diff_tol: float = 1e-5
    gradient_change_tol: float = 1e-4
    gradient_change_rtol: float = 1e-4

    theta_init_d: float = 0.05
    theta_init_l: float = 0.05
    theta_init_t: float = 0.05
    min_rate: float = 2.0 ** -30
    max_rate: float = 2.0

    optimizer: str = "auto"
    steps: int = 5000
    lr: float = 0.01
    adam_warmup_steps: int = 100
    fd_adam_warmup_steps: int = 3
    fd_hessian_refresh_steps: int = 16
    hessian_sgd_normal_fixed_iters_pi: int | None = None
    hessian_sgd_normal_neumann_terms: int | None = None
    hessian_sgd_pi_adjoint_warmstart: bool = False
    pi_fixed_point_relaxation: float = 1.0
    hessian_sgd_validation_interval: int = 0
    hessian_sgd_validation_fixed_iters_pi: int | None = None
    hessian_sgd_validation_neumann_terms: int | None = None
    adagrad_restart_schedule: str = DEFAULT_ADAGRAD_RESTART_SCHEDULE
    adagrad_restart_final_check_iters: int = DEFAULT_ADAGRAD_RESTART_FINAL_CHECK_ITERS
    adagrad_restart_phase_loss_patience: int = 0
    lbfgs_lr: float = 0.1
    lbfgs_history_size: int = 20
    lbfgs_max_iter: int = 1
    lbfgs_max_ls: int = 8
    lbfgs_line_search: str = "none"
    lbfgsb_high_kkt_stop_patience: int = 0
    lbfgsb_high_kkt_stop_min_fallbacks: int = 1
    lbfgsb_fallback_max_coordinates: int = 16
    lbfgsb_fallback_max_loss_evals: int | None = None
    lbfgsb_fallback_resolution_competition_factor: float = 0.0
    lbfgsb_best_retry_attempts: int = 0
    lbfgsb_loss_change_tol_schedule: str | None = None
    lbfgsb_loss_schedule_force_fallback: bool = False
    fd_hessian_epsilon: float = 1e-3
    fd_newton_damping: float = 1e-3
    adaptive_rebatch: bool = False
    adaptive_rebatch_fraction: float = 0.5
    adaptive_rebatch_check_interval: int = 1
    adaptive_rebatch_min_remaining_families: int = 2

    loss_change_tol: float = 3e-3
    loss_patience: int = 1
    best_likelihood_patience: int = 1
    best_likelihood_min_delta: float = 0.0
    projected_grad_tol: float = 1e-3
    loss_stop_projected_grad_gate: bool = True
    projected_lbfgs_min_lr: float = 1e-8

    checkpoint_every: int = 1
    # History rows are recorded every optimizer step; this only gates stdout.
    log_every: int = 1
    resume_from: Path | None = None

    def __post_init__(self) -> None:
        self.species_tree = _normalize_path("species_tree", self.species_tree)
        self.families_file = _normalize_path("families_file", self.families_file)
        self.out_dir = _normalize_path("out_dir", self.out_dir)
        self.resume_from = _normalize_optional_path("resume_from", self.resume_from)
        self.mode = _normalize_mode(self.mode)
        for name in _JSON_BOOL_FIELDS:
            setattr(self, name, _normalize_bool(name, getattr(self, name)))
        self.adaptive_neumann_terms = disabled_adaptive_neumann_terms_value(
            self.adaptive_neumann_terms
        )
        self.start = _normalize_nonnegative_int("start", self.start)
        self.max_families = _normalize_optional_positive_int(
            "max_families",
            self.max_families,
        )
        self.preprocess_cpu_cores = _normalize_optional_positive_int(
            "preprocess_cpu_cores",
            self.preprocess_cpu_cores,
        )
        self.family_chunk_size = int(_normalize_family_chunk_size(self.family_chunk_size))
        self.clade_budget = _normalize_optional_positive_int(
            "clade_budget",
            self.clade_budget,
        )
        self.max_wave_size = _normalize_optional_positive_int(
            "max_wave_size",
            self.max_wave_size,
        )
        self.small_family_max_leaves = _normalize_nonnegative_int(
            "small_family_max_leaves",
            self.small_family_max_leaves,
        )
        self.batch_packing = _normalize_workflow_batch_packing(self.batch_packing)
        if self.fixed_iters_e is not None:
            self.fixed_iters_e = _normalize_positive_int(
                "fixed_iters_e",
                self.fixed_iters_e,
            )
        self.max_iters_e = _normalize_positive_int("max_iters_e", self.max_iters_e)
        self.fixed_iters_pi = _normalize_positive_even_int(
            "fixed_iters_pi",
            self.fixed_iters_pi,
        )
        if self.mode == "specieswise" and self.fixed_iters_pi > 16:
            self.fixed_iters_e = max(
                self.fixed_iters_pi,
                0 if self.fixed_iters_e is None else int(self.fixed_iters_e),
            )
        self.neumann_terms = _normalize_positive_int(
            "neumann_terms",
            self.neumann_terms,
        )
        self.final_check_iters = _normalize_nonnegative_int(
            "final_check_iters",
            self.final_check_iters,
        )
        self.solver_warmup_iters = _normalize_nonnegative_int(
            "solver_warmup_iters",
            self.solver_warmup_iters,
        )
        self.solver_warmup_loss_patience = _normalize_nonnegative_int(
            "solver_warmup_loss_patience",
            self.solver_warmup_loss_patience,
        )
        self.convergence_check_interval = _normalize_positive_int(
            "convergence_check_interval",
            self.convergence_check_interval,
        )
        self.steps = _normalize_positive_int("steps", self.steps)
        self.adam_warmup_steps = _normalize_nonnegative_int(
            "adam_warmup_steps",
            self.adam_warmup_steps,
        )
        self.fd_adam_warmup_steps = _normalize_nonnegative_int(
            "fd_adam_warmup_steps",
            self.fd_adam_warmup_steps,
        )
        self.fd_hessian_refresh_steps = _normalize_positive_int(
            "fd_hessian_refresh_steps",
            self.fd_hessian_refresh_steps,
        )
        self.hessian_sgd_normal_fixed_iters_pi = _normalize_optional_positive_even_int(
            "hessian_sgd_normal_fixed_iters_pi",
            self.hessian_sgd_normal_fixed_iters_pi,
        )
        self.hessian_sgd_normal_neumann_terms = _normalize_optional_positive_int(
            "hessian_sgd_normal_neumann_terms",
            self.hessian_sgd_normal_neumann_terms,
        )
        self.hessian_sgd_validation_interval = _normalize_nonnegative_int(
            "hessian_sgd_validation_interval",
            self.hessian_sgd_validation_interval,
        )
        self.hessian_sgd_validation_fixed_iters_pi = (
            _normalize_optional_positive_even_int(
                "hessian_sgd_validation_fixed_iters_pi",
                self.hessian_sgd_validation_fixed_iters_pi,
            )
        )
        self.hessian_sgd_validation_neumann_terms = _normalize_optional_positive_int(
            "hessian_sgd_validation_neumann_terms",
            self.hessian_sgd_validation_neumann_terms,
        )
        self.adagrad_restart_schedule = _normalize_adagrad_restart_schedule(
            self.adagrad_restart_schedule,
        )
        self.adagrad_restart_final_check_iters = _normalize_nonnegative_int(
            "adagrad_restart_final_check_iters",
            self.adagrad_restart_final_check_iters,
        )
        self.adagrad_restart_phase_loss_patience = _normalize_nonnegative_int(
            "adagrad_restart_phase_loss_patience",
            self.adagrad_restart_phase_loss_patience,
        )
        self.adaptive_rebatch_check_interval = _normalize_positive_int(
            "adaptive_rebatch_check_interval",
            self.adaptive_rebatch_check_interval,
        )
        self.adaptive_rebatch_min_remaining_families = _normalize_positive_int(
            "adaptive_rebatch_min_remaining_families",
            self.adaptive_rebatch_min_remaining_families,
        )
        self.lbfgs_history_size = _normalize_positive_int(
            "lbfgs_history_size",
            self.lbfgs_history_size,
        )
        self.lbfgs_max_iter = _normalize_positive_int(
            "lbfgs_max_iter",
            self.lbfgs_max_iter,
        )
        self.lbfgs_max_ls = _normalize_positive_int(
            "lbfgs_max_ls",
            self.lbfgs_max_ls,
        )
        self.lbfgsb_high_kkt_stop_patience = _normalize_nonnegative_int(
            "lbfgsb_high_kkt_stop_patience",
            self.lbfgsb_high_kkt_stop_patience,
        )
        self.lbfgsb_high_kkt_stop_min_fallbacks = _normalize_nonnegative_int(
            "lbfgsb_high_kkt_stop_min_fallbacks",
            self.lbfgsb_high_kkt_stop_min_fallbacks,
        )
        self.lbfgsb_fallback_max_coordinates = _normalize_nonnegative_int(
            "lbfgsb_fallback_max_coordinates",
            self.lbfgsb_fallback_max_coordinates,
        )
        self.lbfgsb_fallback_max_loss_evals = _normalize_optional_positive_int(
            "lbfgsb_fallback_max_loss_evals",
            self.lbfgsb_fallback_max_loss_evals,
        )
        self.lbfgsb_best_retry_attempts = _normalize_nonnegative_int(
            "lbfgsb_best_retry_attempts",
            self.lbfgsb_best_retry_attempts,
        )
        self.lbfgsb_loss_change_tol_schedule = _normalize_optional_loss_stop_schedule(
            self.lbfgsb_loss_change_tol_schedule,
        )
        self.loss_patience = _normalize_nonnegative_int(
            "loss_patience",
            self.loss_patience,
        )
        self.best_likelihood_patience = _normalize_nonnegative_int(
            "best_likelihood_patience",
            self.best_likelihood_patience,
        )
        self.checkpoint_every = _normalize_nonnegative_int(
            "checkpoint_every",
            self.checkpoint_every,
        )
        self.log_every = _normalize_positive_int("log_every", self.log_every)
        for name in _JSON_FLOAT_FIELDS:
            setattr(self, name, _normalize_finite_float(name, getattr(self, name)))
        self.device = _normalize_device(self.device)
        self.dtype = dtype_name_from_name(self.dtype)
        self.optimizer = _normalize_optimizer(self.mode, self.optimizer)
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
        _normalize_positive_even_int("fixed_iters_pi", self.fixed_iters_pi)
        if self.neumann_terms < 1:
            raise ValueError("neumann_terms must be positive")
        if self.final_check_iters < 0:
            raise ValueError("final_check_iters must be non-negative")
        if self.final_check_iters > 0:
            _normalize_positive_even_int("final_check_iters", self.final_check_iters)
        if self.solver_warmup_iters < 0:
            raise ValueError("solver_warmup_iters must be non-negative")
        if self.solver_warmup_loss_patience < 0:
            raise ValueError("solver_warmup_loss_patience must be non-negative")
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
            "loss_change_tol",
            "best_likelihood_min_delta",
            "projected_grad_tol",
            "lbfgsb_fallback_resolution_competition_factor",
        ):
            if _normalize_finite_float(name, getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if _normalize_finite_float("projected_lbfgs_min_lr", self.projected_lbfgs_min_lr) <= 0.0:
            raise ValueError("projected_lbfgs_min_lr must be positive")
        min_rate = _normalize_finite_float("min_rate", self.min_rate)
        max_rate = _normalize_finite_float("max_rate", self.max_rate)
        if min_rate <= 0.0 or max_rate <= min_rate:
            raise ValueError("rate bounds must satisfy 0 < min_rate < max_rate")
        theta_init_d = _normalize_finite_float("theta_init_d", self.theta_init_d)
        theta_init_l = _normalize_finite_float("theta_init_l", self.theta_init_l)
        theta_init_t = _normalize_finite_float("theta_init_t", self.theta_init_t)
        if (
            theta_init_d <= 0.0
            or theta_init_l <= 0.0
            or theta_init_t <= 0.0
        ):
            raise ValueError("theta_init_d/l/t must be strictly positive")
        if self.optimizer not in {
            "adam",
            "adagrad",
            "projected-sgd",
            "lbfgs",
            "adam-lbfgs",
            "projected-lbfgs",
            "lbfgsb",
            "batched-lbfgs",
            "adam-fd-newton",
            "hessian-sgd",
            "adagrad-restarts",
            "adagrad-restarts-lbfgsb",
        }:
            raise ValueError(
                "optimizer must be auto, adam, adagrad, projected-sgd, "
                "lbfgs, adam-lbfgs, projected-lbfgs, lbfgsb, batched-lbfgs, "
                "adam-fd-newton, hessian-sgd, adagrad-restarts, or "
                "adagrad-restarts-lbfgsb"
            )
        if self.optimizer == "batched-lbfgs" and self.mode != "genewise":
            raise ValueError("batched-lbfgs optimizer requires genewise mode")
        if self.optimizer == "adam-fd-newton" and self.mode != "genewise":
            raise ValueError("adam-fd-newton optimizer requires genewise mode")
        if self.optimizer == "hessian-sgd" and self.mode != "genewise":
            raise ValueError("hessian-sgd optimizer requires genewise mode")
        hessian_sgd_normal_configured = (
            self.hessian_sgd_normal_fixed_iters_pi is not None
            or self.hessian_sgd_normal_neumann_terms is not None
        )
        if hessian_sgd_normal_configured and self.optimizer != "hessian-sgd":
            raise ValueError(
                "hessian_sgd_normal solver controls require genewise "
                "hessian-sgd optimizer"
            )
        if self.hessian_sgd_pi_adjoint_warmstart and self.optimizer != "hessian-sgd":
            raise ValueError(
                "hessian_sgd_pi_adjoint_warmstart requires genewise "
                "hessian-sgd optimizer"
            )
        if self.pi_fixed_point_relaxation <= 0.0:
            raise ValueError("pi_fixed_point_relaxation must be positive")
        if (
            self.pi_fixed_point_relaxation != 1.0
            and (
                self.optimizer != "hessian-sgd"
                or not self.hessian_sgd_pi_adjoint_warmstart
            )
        ):
            raise ValueError(
                "pi_fixed_point_relaxation requires "
                "hessian_sgd_pi_adjoint_warmstart with genewise hessian-sgd"
            )
        hessian_sgd_validation_configured = (
            self.hessian_sgd_validation_interval > 0
            or self.hessian_sgd_validation_fixed_iters_pi is not None
            or self.hessian_sgd_validation_neumann_terms is not None
        )
        if hessian_sgd_validation_configured and self.optimizer != "hessian-sgd":
            raise ValueError(
                "hessian_sgd_validation controls require genewise hessian-sgd "
                "optimizer"
            )
        if (
            self.hessian_sgd_validation_interval == 0
            and (
                self.hessian_sgd_validation_fixed_iters_pi is not None
                or self.hessian_sgd_validation_neumann_terms is not None
            )
        ):
            raise ValueError(
                "hessian_sgd_validation_interval must be positive when "
                "validation budgets are provided"
            )
        if self.optimizer == "adagrad-restarts" and self.mode != "specieswise":
            raise ValueError("adagrad-restarts optimizer requires specieswise mode")
        if (
            self.optimizer == "adagrad-restarts-lbfgsb"
            and self.mode != "specieswise"
        ):
            raise ValueError(
                "adagrad-restarts-lbfgsb optimizer requires specieswise mode"
            )
        adagrad_restart_configured = (
            self.adagrad_restart_schedule
            != DEFAULT_NORMALIZED_ADAGRAD_RESTART_SCHEDULE
            or self.adagrad_restart_final_check_iters
            != DEFAULT_ADAGRAD_RESTART_FINAL_CHECK_ITERS
        )
        if (
            adagrad_restart_configured
            and self.optimizer
            not in {"adagrad-restarts", "adagrad-restarts-lbfgsb"}
        ):
            raise ValueError(
                "adagrad_restart controls require specieswise "
                "adagrad-restarts"
            )
        if self.adagrad_restart_final_check_iters > 0:
            _normalize_positive_even_int(
                "adagrad_restart_final_check_iters",
                self.adagrad_restart_final_check_iters,
            )
        if self.steps < 1:
            raise ValueError("steps must be positive")
        lr = _normalize_finite_float("lr", self.lr)
        lbfgs_lr = _normalize_finite_float("lbfgs_lr", self.lbfgs_lr)
        if lr <= 0.0 or lbfgs_lr <= 0.0:
            raise ValueError("optimizer learning rates must be positive")
        if self.fd_hessian_epsilon <= 0.0:
            raise ValueError("fd_hessian_epsilon must be positive")
        if self.fd_newton_damping <= 0.0:
            raise ValueError("fd_newton_damping must be positive")
        if not (0.0 < self.adaptive_rebatch_fraction <= 1.0):
            raise ValueError("adaptive_rebatch_fraction must be in (0, 1]")
        if self.adaptive_rebatch and (
            self.mode != "genewise"
            or self.optimizer not in {"batched-lbfgs", "adam-fd-newton", "hessian-sgd"}
        ):
            raise ValueError(
                "adaptive_rebatch requires genewise mode with batched-lbfgs "
                "adam-fd-newton, or hessian-sgd"
            )
        if self.adam_warmup_steps < 0:
            raise ValueError("adam_warmup_steps must be non-negative")
        if self.fd_adam_warmup_steps < 0:
            raise ValueError("fd_adam_warmup_steps must be non-negative")
        if self.fd_hessian_refresh_steps < 1:
            raise ValueError("fd_hessian_refresh_steps must be positive")
        if (
            self.lbfgs_history_size < 1
            or self.lbfgs_max_iter < 1
            or self.lbfgs_max_ls < 1
        ):
            raise ValueError("LBFGS history size, max_iter, and max_ls must be positive")
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
        unknown = sorted(
            str(key)
            for key in data
            if key not in allowed and key not in _RUN_CONFIG_LEGACY_FIELDS
        )
        if unknown:
            raise ValueError(f"unknown RunConfig field(s): {', '.join(unknown)}")
        data = {
            key: value
            for key, value in data.items()
            if key not in _RUN_CONFIG_LEGACY_FIELDS
        }
        missing = [
            name for name in _RUN_CONFIG_REQUIRED_PATH_FIELDS if data.get(name) is None
        ]
        if missing:
            raise ValueError(
                f"missing required RunConfig field(s): {', '.join(missing)}"
            )
        _validate_json_scalar_types(data)
        return cls(**dict(data))

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        return cls.from_dict(load_run_config_data(path))

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _specieswise_route_defaults() -> SpecieswiseRouteDefaults:
    return SpecieswiseRouteDefaults(
        normalized_adagrad_restart_schedule=(
            DEFAULT_NORMALIZED_ADAGRAD_RESTART_SCHEDULE
        ),
        adagrad_restart_total_steps=DEFAULT_ADAGRAD_RESTART_TOTAL_STEPS,
        adagrad_restart_final_check_iters=DEFAULT_ADAGRAD_RESTART_FINAL_CHECK_ITERS,
    )


def production_default_route_contract() -> dict[str, Any]:
    """Return the shipped likelihood/gradient route contract fields."""
    return _route_defaults.production_default_route_contract()


def production_default_route_contract_fields() -> tuple[str, ...]:
    """Return the required shipped likelihood/gradient route field names."""
    return _route_defaults.production_default_route_contract_fields()


def production_default_optimizer_config_overrides(mode: str) -> dict[str, Any]:
    """Return RunConfig overrides for the shipped optimizer profile."""
    return _route_defaults.production_default_optimizer_config_overrides(mode)


def production_default_optimizer_setting_mismatches_from_route(
    route: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(missing, mismatched)`` audit fields for a route dictionary."""
    return _route_defaults.production_default_optimizer_setting_mismatches_from_route(
        route,
        specieswise_defaults=_specieswise_route_defaults(),
        normalize_adagrad_restart_schedule=_normalize_adagrad_restart_schedule,
    )


def production_default_route_mismatches_from_route(
    route: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return missing/mismatched fields for the shipped route.

    The route includes likelihood/gradient fields, resident batch route fields,
    and optimizer evidence.
    """
    return _route_defaults.production_default_route_mismatches_from_route(
        route,
        specieswise_defaults=_specieswise_route_defaults(),
        normalize_adagrad_restart_schedule=_normalize_adagrad_restart_schedule,
    )


def production_default_optimizer_setting_mismatches(
    config: RunConfig,
) -> tuple[str, ...]:
    return _route_defaults.production_default_optimizer_setting_mismatches(
        config,
        effective_optimizer_step_cap=effective_optimizer_step_cap,
        effective_final_check_iters=effective_final_check_iters,
        effective_final_check_iters_e=effective_final_check_iters_e,
        adagrad_restart_schedule_total_steps=adagrad_restart_schedule_total_steps,
        specieswise_defaults=_specieswise_route_defaults(),
        normalize_adagrad_restart_schedule=_normalize_adagrad_restart_schedule,
    )


def effective_route_metadata(config: RunConfig) -> dict[str, Any]:
    return _route_defaults.effective_route_metadata(
        config,
        effective_optimizer_step_cap=effective_optimizer_step_cap,
        effective_final_check_iters=effective_final_check_iters,
        effective_final_check_iters_e=effective_final_check_iters_e,
        adagrad_restart_schedule_total_steps=adagrad_restart_schedule_total_steps,
        specieswise_defaults=_specieswise_route_defaults(),
        normalize_adagrad_restart_schedule=_normalize_adagrad_restart_schedule,
        jsonable=_jsonable,
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
        self.checkpoint = _normalize_path("checkpoint", self.checkpoint)
        self.out_dir = _normalize_optional_path("out_dir", self.out_dir)
        self.backtrack_binary = _normalize_optional_path(
            "backtrack_binary",
            self.backtrack_binary,
        )
        self.samples = _normalize_positive_int("samples", self.samples)
        self.seed = _normalize_uint64("seed", self.seed)
        self.family_start = _normalize_nonnegative_int(
            "family_start",
            self.family_start,
        )
        self.max_families = _normalize_optional_positive_int(
            "max_families",
            self.max_families,
        )
        self.max_events = _normalize_optional_positive_int(
            "max_events",
            self.max_events,
        )
