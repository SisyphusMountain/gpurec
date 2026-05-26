from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from numbers import Real
from pathlib import Path
from typing import Any

import torch

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
DEFAULT_ADAGRAD_RESTART_SCHEDULE = "8:1.0:60,16:0.5:35,32:0.5:30"
DEFAULT_ADAGRAD_RESTART_FINAL_CHECK_ITERS = 128
MODE_DEFAULT_OPTIMIZERS = {
    "genewise": "hessian-sgd",
    "specieswise": "adagrad-restarts",
    "global": "adam",
}
_RUN_CONFIG_MODES = frozenset(MODE_DEFAULT_OPTIMIZERS)


@dataclass(frozen=True)
class AdagradRestartPhase:
    fixed_iters_e: int
    fixed_iters_pi: int
    neumann_terms: int
    lr: float
    steps: int

    @property
    def budget(self) -> int:
        return self.fixed_iters_pi

    @property
    def is_tied_budget(self) -> bool:
        return (
            self.fixed_iters_e == self.fixed_iters_pi
            and self.neumann_terms == self.fixed_iters_pi
        )

    def budget_label(self) -> str:
        if self.is_tied_budget:
            return f"fixed{self.fixed_iters_pi}"
        label = f"E{self.fixed_iters_e}_Pi{self.fixed_iters_pi}"
        if self.neumann_terms != self.fixed_iters_pi:
            label += f"_N{self.neumann_terms}"
        return label

    def budget_spec(self) -> str:
        if self.is_tied_budget:
            return str(self.fixed_iters_pi)
        if self.neumann_terms == self.fixed_iters_pi:
            return f"{self.fixed_iters_e}/{self.fixed_iters_pi}"
        return f"{self.fixed_iters_e}/{self.fixed_iters_pi}/{self.neumann_terms}"


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
    if not isinstance(mode, str):
        raise ValueError("mode must be 'global', 'specieswise', or 'genewise'")
    normalized = mode.strip().lower()
    if normalized not in _RUN_CONFIG_MODES:
        raise ValueError("mode must be 'global', 'specieswise', or 'genewise'")
    return normalized


def normalize_mode_name(mode: str) -> str:
    return _normalize_mode(mode)


def default_optimizer_for_mode(mode: str) -> str:
    return MODE_DEFAULT_OPTIMIZERS[normalize_mode_name(mode)]


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
    mode = normalize_mode_name(mode)
    normalized = normalize_optimizer_name(value)
    if normalized == "auto":
        return default_optimizer_for_mode(mode)
    return normalized


def normalize_optimizer_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            "optimizer must be auto, adam, adagrad, projected-sgd, lbfgs, "
            "adam-lbfgs, projected-lbfgs, lbfgsb, batched-lbfgs, "
            "adam-fd-newton, hessian-sgd, or adagrad-restarts"
        )
    return value.strip().lower().replace("_", "-")


def normalize_optimizer_for_mode(mode: str, value: str) -> str:
    return _normalize_optimizer(mode, value)


def adagrad_restart_schedule_specs(value: str) -> tuple[AdagradRestartPhase, ...]:
    if not isinstance(value, str):
        raise ValueError("adagrad_restart_schedule must be a string")
    text = value.strip()
    if not text:
        raise ValueError("adagrad_restart_schedule must not be empty")
    phases: list[AdagradRestartPhase] = []
    for position, raw_entry in enumerate(text.split(","), start=1):
        entry = raw_entry.strip()
        if not entry:
            raise ValueError(
                "adagrad_restart_schedule entries must be budget:lr:steps "
                "or E/Pi[/Neumann]:lr:steps"
            )
        pieces = [piece.strip() for piece in entry.split(":")]
        if len(pieces) != 3:
            raise ValueError(
                "adagrad_restart_schedule entries must be budget:lr:steps "
                "or E/Pi[/Neumann]:lr:steps"
            )
        budget_pieces = [piece.strip() for piece in pieces[0].split("/")]
        if len(budget_pieces) == 1:
            fixed_iters_pi = _normalize_positive_even_int(
                f"adagrad_restart_schedule entry {position} budget",
                budget_pieces[0],
            )
            fixed_iters_e = fixed_iters_pi
            neumann_terms = fixed_iters_pi
        elif len(budget_pieces) in {2, 3}:
            fixed_iters_e = _normalize_positive_int(
                f"adagrad_restart_schedule entry {position} E budget",
                budget_pieces[0],
            )
            fixed_iters_pi = _normalize_positive_even_int(
                f"adagrad_restart_schedule entry {position} Pi budget",
                budget_pieces[1],
            )
            neumann_terms = (
                fixed_iters_pi
                if len(budget_pieces) == 2
                else _normalize_positive_int(
                    f"adagrad_restart_schedule entry {position} Neumann budget",
                    budget_pieces[2],
                )
            )
        else:
            raise ValueError(
                "adagrad_restart_schedule budget entries must be budget "
                "or E/Pi[/Neumann]"
            )
        lr = _normalize_finite_float(
            f"adagrad_restart_schedule entry {position} lr",
            pieces[1],
        )
        if lr <= 0.0:
            raise ValueError("adagrad_restart_schedule learning rates must be positive")
        steps = _normalize_positive_int(
            f"adagrad_restart_schedule entry {position} steps",
            pieces[2],
        )
        phases.append(
            AdagradRestartPhase(
                fixed_iters_e=fixed_iters_e,
                fixed_iters_pi=fixed_iters_pi,
                neumann_terms=neumann_terms,
                lr=lr,
                steps=steps,
            )
        )
        if len(phases) > 1:
            previous = phases[-2]
            current = phases[-1]
            if (
                current.fixed_iters_e < previous.fixed_iters_e
                or current.fixed_iters_pi < previous.fixed_iters_pi
                or current.neumann_terms < previous.neumann_terms
            ):
                raise ValueError(
                    "adagrad_restart_schedule phases must not decrease "
                    "fixed_iters_E, fixed_iters_Pi, or neumann_terms"
                )
    return tuple(phases)


def adagrad_restart_schedule_total_steps(value: str) -> int:
    return sum(phase.steps for phase in adagrad_restart_schedule_specs(value))


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


def _normalize_adagrad_restart_schedule(value: str) -> str:
    return ",".join(
        f"{phase.budget_spec()}:{phase.lr:.12g}:{phase.steps}"
        for phase in adagrad_restart_schedule_specs(value)
    )


DEFAULT_NORMALIZED_ADAGRAD_RESTART_SCHEDULE = _normalize_adagrad_restart_schedule(
    DEFAULT_ADAGRAD_RESTART_SCHEDULE
)
DEFAULT_ADAGRAD_RESTART_TOTAL_STEPS = adagrad_restart_schedule_total_steps(
    DEFAULT_ADAGRAD_RESTART_SCHEDULE
)


def _normalize_workflow_batch_packing(value: str | None) -> str:
    if value is None:
        raise ValueError("batch_packing must be provided as a string")
    return _normalize_batch_packing(value)


def _resolve_run_config_path_fields(
    data: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    resolved = dict(data)
    for name in _RUN_CONFIG_PATH_FIELDS:
        value = resolved.get(name)
        if value is None or not isinstance(value, (str, Path)):
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        resolved[name] = path.resolve()
    return resolved


def _reject_json_constant(constant: str) -> None:
    raise ValueError(f"invalid JSON numeric constant {constant}")


def load_json_object(path: str | Path, *, description: str = "config") -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise ValueError(f"could not read {description} {path}: {detail}") from exc
    try:
        data = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON {description} {path}: {exc.msg}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid JSON {description} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{description} {path} must contain a JSON object")
    return data


def load_run_config_data(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    data = load_json_object(path)
    return _resolve_run_config_path_fields(
        data,
        base_dir=path.parent,
    )


_JSON_INT_FIELDS = {
    "start",
    "max_families",
    "clade_budget",
    "max_wave_size",
    "small_family_max_leaves",
    "fixed_iters_e",
    "max_iters_e",
    "fixed_iters_pi",
    "neumann_terms",
    "final_check_iters",
    "solver_warmup_iters",
    "solver_warmup_loss_patience",
    "convergence_check_interval",
    "preprocess_cpu_cores",
    "steps",
    "adam_warmup_steps",
    "fd_adam_warmup_steps",
    "fd_hessian_refresh_steps",
    "hessian_sgd_normal_fixed_iters_pi",
    "hessian_sgd_normal_neumann_terms",
    "hessian_sgd_validation_interval",
    "hessian_sgd_validation_fixed_iters_pi",
    "hessian_sgd_validation_neumann_terms",
    "adagrad_restart_final_check_iters",
    "adaptive_rebatch_check_interval",
    "adaptive_rebatch_min_remaining_families",
    "lbfgs_history_size",
    "lbfgs_max_iter",
    "lbfgs_max_ls",
    "loss_patience",
    "best_likelihood_patience",
    "checkpoint_every",
    "log_every",
}
_JSON_FLOAT_FIELDS = {
    "tol_e",
    "e_logsumexp_tol",
    "pi_max_diff_tol",
    "gradient_change_tol",
    "gradient_change_rtol",
    "theta_init_d",
    "theta_init_l",
    "theta_init_t",
    "min_rate",
    "max_rate",
    "lr",
    "lbfgs_lr",
    "fd_hessian_epsilon",
    "fd_newton_damping",
    "adaptive_rebatch_fraction",
    "loss_change_tol",
    "best_likelihood_min_delta",
    "projected_grad_tol",
    "projected_lbfgs_min_lr",
    "pi_fixed_point_relaxation",
}
_JSON_BOOL_FIELDS = {
    "adaptive_iters",
    "adaptive_neumann_terms",
    "adaptive_rebatch",
    "hessian_sgd_pi_adjoint_warmstart",
}
_RUN_CONFIG_REQUIRED_PATH_FIELDS = ("species_tree", "families_file", "out_dir")
_RUN_CONFIG_PATH_FIELDS = _RUN_CONFIG_REQUIRED_PATH_FIELDS + (
    "resume_from",
)
_RUN_CONFIG_LEGACY_FIELDS = frozenset(
    {
        "fd_newton_max_step",
        "grad_inf_tol",
        "hessian_sgd_polish_max_steps",
        "hessian_sgd_polish_refresh_steps",
        "hessian_sgd_polish_max_ls",
        "solver_warmup_grad_inf_tol",
    }
)


def _validate_json_scalar_types(data: dict[str, Any]) -> None:
    for name in _JSON_INT_FIELDS:
        if name not in data or data[name] is None:
            continue
        if isinstance(data[name], bool) or not isinstance(data[name], int):
            raise ValueError(f"{name} must be an integer")
    for name in _JSON_FLOAT_FIELDS:
        if name not in data:
            continue
        if isinstance(data[name], bool) or not isinstance(data[name], (int, float)):
            raise ValueError(f"{name} must be a number")
    for name in _JSON_BOOL_FIELDS:
        if name not in data:
            continue
        if not isinstance(data[name], bool):
            raise ValueError(f"{name} must be true or false")
    if (
        "device" in data
        and data["device"] is not None
        and not isinstance(data["device"], str)
    ):
        raise ValueError("device must be a device string")
    if (
        "mode" in data
        and data["mode"] is not None
        and not isinstance(data["mode"], str)
    ):
        raise ValueError("mode must be 'global', 'specieswise', or 'genewise'")
    for name in _RUN_CONFIG_PATH_FIELDS:
        if name not in data or data[name] is None:
            continue
        if not isinstance(data[name], (str, Path)):
            raise ValueError(f"{name} must be a path string")


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
    clade_budget: int | None = 500_000
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
    lbfgs_lr: float = 0.1
    lbfgs_history_size: int = 20
    lbfgs_max_iter: int = 1
    lbfgs_max_ls: int = 8
    lbfgs_line_search: str = "none"
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
        }:
            raise ValueError(
                "optimizer must be auto, adam, adagrad, projected-sgd, "
                "lbfgs, adam-lbfgs, projected-lbfgs, lbfgsb, batched-lbfgs, "
                "adam-fd-newton, hessian-sgd, or adagrad-restarts"
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
        adagrad_restart_configured = (
            self.adagrad_restart_schedule
            != DEFAULT_NORMALIZED_ADAGRAD_RESTART_SCHEDULE
            or self.adagrad_restart_final_check_iters
            != DEFAULT_ADAGRAD_RESTART_FINAL_CHECK_ITERS
        )
        if adagrad_restart_configured and self.optimizer != "adagrad-restarts":
            raise ValueError(
                "adagrad_restart controls require specieswise "
                "adagrad-restarts optimizer"
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


_PRODUCTION_DEFAULT_GENEWISE_OPTIMIZER_SETTINGS = {
    "final_check_iters": 32,
    "final_check_iters_e": None,
    "solver_warmup_iters": 4,
    "fd_adam_warmup_steps": 3,
    "fd_hessian_refresh_steps": 16,
    "hessian_sgd_normal_fixed_iters_pi": None,
    "hessian_sgd_normal_neumann_terms": None,
    "hessian_sgd_pi_adjoint_warmstart": False,
    "pi_fixed_point_relaxation": 1.0,
    "hessian_sgd_validation_interval": 0,
    "hessian_sgd_validation_fixed_iters_pi": None,
    "hessian_sgd_validation_neumann_terms": None,
}
_PRODUCTION_DEFAULT_SPECIESWISE_OPTIMIZER_SETTINGS = {
    "final_check_iters": DEFAULT_ADAGRAD_RESTART_FINAL_CHECK_ITERS,
    "final_check_iters_e": DEFAULT_ADAGRAD_RESTART_FINAL_CHECK_ITERS,
    "optimizer_step_cap": DEFAULT_ADAGRAD_RESTART_TOTAL_STEPS,
    "optimizer_step_cap_reason": "adagrad_restart_schedule",
    "adagrad_restart_schedule": DEFAULT_NORMALIZED_ADAGRAD_RESTART_SCHEDULE,
    "adagrad_restart_total_steps": DEFAULT_ADAGRAD_RESTART_TOTAL_STEPS,
    "adagrad_restart_final_check_iters": DEFAULT_ADAGRAD_RESTART_FINAL_CHECK_ITERS,
}
_PRODUCTION_DEFAULT_OPTIMIZER_CONFIG_FIELDS = {
    "genewise": tuple(
        name
        for name in _PRODUCTION_DEFAULT_GENEWISE_OPTIMIZER_SETTINGS
        if name != "final_check_iters_e"
    ),
    "specieswise": (
        "adagrad_restart_schedule",
        "adagrad_restart_final_check_iters",
    ),
    "global": (),
}
_PRODUCTION_DEFAULT_ROUTE_CONTRACT = {
    "objective": "negative_log_likelihood_bits",
    "gradient_route": "implicit_first_order_adjoint",
    "rate_parameterization": "base2_log_dlt_rates",
    "production_default_basis": "hogenom_and_test_trees_1000",
}


def production_default_route_contract() -> dict[str, Any]:
    """Return the shipped likelihood/gradient route contract fields."""
    return dict(_PRODUCTION_DEFAULT_ROUTE_CONTRACT)


def production_default_route_contract_fields() -> tuple[str, ...]:
    """Return the required shipped likelihood/gradient route field names."""
    return tuple(_PRODUCTION_DEFAULT_ROUTE_CONTRACT)


def production_default_optimizer_config_overrides(mode: str) -> dict[str, Any]:
    """Return RunConfig overrides for the shipped optimizer profile."""
    mode_text = normalize_mode_name(mode)
    if mode_text == "specieswise":
        settings = {
            "adagrad_restart_schedule": DEFAULT_ADAGRAD_RESTART_SCHEDULE,
            "adagrad_restart_final_check_iters": (
                DEFAULT_ADAGRAD_RESTART_FINAL_CHECK_ITERS
            ),
        }
    else:
        settings = _production_default_optimizer_expected_settings(mode_text)
    return {
        name: settings[name]
        for name in _PRODUCTION_DEFAULT_OPTIMIZER_CONFIG_FIELDS[mode_text]
    }


def _production_default_optimizer_expected_settings(mode: str) -> dict[str, Any]:
    if mode == "genewise":
        return _PRODUCTION_DEFAULT_GENEWISE_OPTIMIZER_SETTINGS
    if mode == "specieswise":
        return _PRODUCTION_DEFAULT_SPECIESWISE_OPTIMIZER_SETTINGS
    if mode == "global":
        return {}
    raise ValueError("mode must be 'global', 'specieswise', or 'genewise'")


def _route_setting_matches(name: str, actual: Any, expected: Any) -> bool:
    if name == "adagrad_restart_schedule" and actual is not None:
        try:
            actual = _normalize_adagrad_restart_schedule(str(actual))
        except ValueError:
            return False
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        try:
            return integer_value(name, actual) == expected
        except ValueError:
            return False
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, Real):
            return False
        try:
            return finite_float(name, actual) == expected
        except ValueError:
            return False
    return actual == expected


def production_default_optimizer_setting_mismatches_from_route(
    route: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(missing, mismatched)`` audit fields for a route dictionary."""
    missing: list[str] = []
    mismatched: list[str] = []
    mode = route.get("mode")
    optimizer = route.get("optimizer")
    if mode is None:
        missing.append("mode")
    if optimizer is None:
        missing.append("optimizer")
    if missing:
        return tuple(missing), tuple(mismatched)
    try:
        mode_text = normalize_mode_name(str(mode))
        mode_default_optimizer = default_optimizer_for_mode(mode_text)
    except ValueError:
        mismatched.append("mode")
        return tuple(missing), tuple(mismatched)
    try:
        optimizer_text = normalize_optimizer_for_mode(mode_text, optimizer)
    except ValueError:
        mismatched.append("optimizer")
        return tuple(missing), tuple(mismatched)
    if optimizer_text != mode_default_optimizer:
        mismatched.append("optimizer")
        return tuple(missing), tuple(mismatched)
    if mode_text == "global":
        mismatched.append("mode")
        return tuple(missing), tuple(mismatched)
    expected_settings = _production_default_optimizer_expected_settings(mode_text)
    for name, expected in expected_settings.items():
        if name not in route:
            missing.append(name)
            continue
        if not _route_setting_matches(name, route[name], expected):
            mismatched.append(name)
    return tuple(missing), tuple(mismatched)


def production_default_route_mismatches_from_route(
    route: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return missing/mismatched fields for the shipped route.

    The route includes likelihood/gradient fields and optimizer evidence.
    """
    missing: list[str] = []
    mismatched: list[str] = []
    for name, expected in _PRODUCTION_DEFAULT_ROUTE_CONTRACT.items():
        if name not in route:
            missing.append(name)
        elif route[name] != expected:
            mismatched.append(name)
    setting_missing, setting_mismatches = (
        production_default_optimizer_setting_mismatches_from_route(route)
    )
    missing.extend(setting_missing)
    mismatched.extend(setting_mismatches)
    return tuple(missing), tuple(mismatched)


def production_default_optimizer_setting_mismatches(
    config: RunConfig,
) -> tuple[str, ...]:
    optimizer_step_cap, optimizer_step_cap_reason = effective_optimizer_step_cap(
        config
    )
    route = {
        "mode": config.mode,
        "optimizer": config.optimizer,
        "final_check_iters": effective_final_check_iters(config),
        "final_check_iters_e": effective_final_check_iters_e(config),
        "optimizer_step_cap": optimizer_step_cap,
        "optimizer_step_cap_reason": optimizer_step_cap_reason,
        "solver_warmup_iters": config.solver_warmup_iters,
        "fd_adam_warmup_steps": config.fd_adam_warmup_steps,
        "fd_hessian_refresh_steps": config.fd_hessian_refresh_steps,
        "hessian_sgd_normal_fixed_iters_pi": (
            config.hessian_sgd_normal_fixed_iters_pi
        ),
        "hessian_sgd_normal_neumann_terms": config.hessian_sgd_normal_neumann_terms,
        "hessian_sgd_pi_adjoint_warmstart": (
            config.hessian_sgd_pi_adjoint_warmstart
        ),
        "pi_fixed_point_relaxation": config.pi_fixed_point_relaxation,
        "hessian_sgd_validation_interval": config.hessian_sgd_validation_interval,
        "hessian_sgd_validation_fixed_iters_pi": (
            config.hessian_sgd_validation_fixed_iters_pi
        ),
        "hessian_sgd_validation_neumann_terms": (
            config.hessian_sgd_validation_neumann_terms
        ),
        "adagrad_restart_schedule": config.adagrad_restart_schedule,
        "adagrad_restart_total_steps": adagrad_restart_schedule_total_steps(
            config.adagrad_restart_schedule
        ),
        "adagrad_restart_final_check_iters": (
            config.adagrad_restart_final_check_iters
        ),
    }
    missing, mismatches = production_default_optimizer_setting_mismatches_from_route(
        route
    )
    if missing:
        raise RuntimeError(
            "internal error: complete RunConfig route is missing production "
            f"default optimizer setting field(s): {', '.join(missing)}"
        )
    return mismatches


def effective_route_metadata(config: RunConfig) -> dict[str, Any]:
    optimizer_step_cap, optimizer_step_cap_reason = effective_optimizer_step_cap(
        config
    )
    mode_default_optimizer = default_optimizer_for_mode(config.mode)
    default_setting_mismatches = production_default_optimizer_setting_mismatches(
        config
    )
    route: dict[str, Any] = {
        **_PRODUCTION_DEFAULT_ROUTE_CONTRACT,
        "mode": config.mode,
        "optimizer": config.optimizer,
        "mode_default_optimizer": mode_default_optimizer,
        "uses_mode_default_optimizer": config.optimizer == mode_default_optimizer,
        "uses_production_default_optimizer_settings": (
            len(default_setting_mismatches) == 0
        ),
        "production_default_optimizer_setting_mismatches": list(
            default_setting_mismatches
        ),
        "batch_packing": config.batch_packing,
        "family_chunk_size": config.family_chunk_size,
        "clade_budget": config.clade_budget,
        "fixed_iters_e": config.fixed_iters_e,
        "fixed_iters_pi": config.fixed_iters_pi,
        "neumann_terms": config.neumann_terms,
        "final_check_iters": effective_final_check_iters(config),
        "final_check_iters_e": effective_final_check_iters_e(config),
        "configured_steps": config.steps,
        "optimizer_step_cap": optimizer_step_cap,
        "optimizer_step_cap_reason": optimizer_step_cap_reason,
    }
    if config.optimizer == "hessian-sgd":
        route.update(
            {
                "solver_warmup_iters": config.solver_warmup_iters,
                "fd_adam_warmup_steps": config.fd_adam_warmup_steps,
                "fd_hessian_refresh_steps": config.fd_hessian_refresh_steps,
                "hessian_sgd_normal_fixed_iters_pi": (
                    config.hessian_sgd_normal_fixed_iters_pi
                ),
                "hessian_sgd_normal_neumann_terms": (
                    config.hessian_sgd_normal_neumann_terms
                ),
                "hessian_sgd_pi_adjoint_warmstart": (
                    config.hessian_sgd_pi_adjoint_warmstart
                ),
                "pi_fixed_point_relaxation": config.pi_fixed_point_relaxation,
                "hessian_sgd_validation_interval": (
                    config.hessian_sgd_validation_interval
                ),
                "hessian_sgd_validation_fixed_iters_pi": (
                    config.hessian_sgd_validation_fixed_iters_pi
                ),
                "hessian_sgd_validation_neumann_terms": (
                    config.hessian_sgd_validation_neumann_terms
                ),
            }
        )
    elif config.optimizer == "adagrad-restarts":
        route.update(
            {
                "adagrad_restart_schedule": config.adagrad_restart_schedule,
                "adagrad_restart_total_steps": (
                    adagrad_restart_schedule_total_steps(
                        config.adagrad_restart_schedule
                    )
                ),
                "adagrad_restart_final_check_iters": (
                    config.adagrad_restart_final_check_iters
                ),
            }
        )
    default_route_missing, default_route_mismatches = (
        production_default_route_mismatches_from_route(route)
    )
    if default_route_missing:
        raise RuntimeError(
            "internal error: complete RunConfig route is missing production "
            f"default route field(s): {', '.join(default_route_missing)}"
        )
    route["uses_production_default_route"] = len(default_route_mismatches) == 0
    route["production_default_route_mismatches"] = list(default_route_mismatches)
    return _jsonable(route)


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
