"""Private workflow schedule parsers.

Public callers should keep importing these names from ``gpurec.workflow.config``.
"""

from __future__ import annotations

from dataclasses import dataclass

from gpurec._validation import (
    finite_float,
    integer_value,
    positive_even_int,
    positive_int,
)

from ._route_defaults import DEFAULT_ADAGRAD_RESTART_SCHEDULE


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


@dataclass(frozen=True)
class LossStopPhase:
    loss_change_tol: float
    loss_patience: int

    def spec(self) -> str:
        return f"{self.loss_change_tol:.12g}:{self.loss_patience}"


# Keep historical public class paths for pickle/introspection compatibility.
AdagradRestartPhase.__module__ = "gpurec.workflow.config"
LossStopPhase.__module__ = "gpurec.workflow.config"


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


def _normalize_finite_float(name: str, value: float | int | str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError(f"{name} must be a number")
        return finite_float(name, value)
    return finite_float(name, value)


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


def loss_stop_schedule_specs(value: str) -> tuple[LossStopPhase, ...]:
    if not isinstance(value, str):
        raise ValueError("lbfgsb_loss_change_tol_schedule must be a string")
    text = value.strip()
    if not text:
        raise ValueError("lbfgsb_loss_change_tol_schedule must not be empty")
    phases: list[LossStopPhase] = []
    for position, raw_entry in enumerate(text.split(","), start=1):
        entry = raw_entry.strip()
        if not entry:
            raise ValueError(
                "lbfgsb_loss_change_tol_schedule entries must be "
                "loss_change_tol:loss_patience"
            )
        pieces = [piece.strip() for piece in entry.split(":")]
        if len(pieces) != 2:
            raise ValueError(
                "lbfgsb_loss_change_tol_schedule entries must be "
                "loss_change_tol:loss_patience"
            )
        loss_change_tol = _normalize_finite_float(
            f"lbfgsb_loss_change_tol_schedule entry {position} loss_change_tol",
            pieces[0],
        )
        if loss_change_tol < 0.0:
            raise ValueError(
                "lbfgsb_loss_change_tol_schedule loss_change_tol values must "
                "be non-negative"
            )
        loss_patience = _normalize_positive_int(
            f"lbfgsb_loss_change_tol_schedule entry {position} loss_patience",
            pieces[1],
        )
        phases.append(
            LossStopPhase(
                loss_change_tol=loss_change_tol,
                loss_patience=loss_patience,
            )
        )
    return tuple(phases)


def _normalize_optional_loss_stop_schedule(value: str | None) -> str | None:
    if value is None:
        return None
    return ",".join(phase.spec() for phase in loss_stop_schedule_specs(value))
