"""Dataclasses for optimization logging and history."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FixedPointInfo:
    iterations_E: int
    iterations_Pi: int

@dataclass
class LinearSolveStats:
    method: str
    iters: int
    rel_residual: float
    fallback_used: bool

@dataclass
class StepRecord:
    iteration: int
    theta: torch.Tensor
    rates: torch.Tensor
    negative_log_likelihood: float
    log_likelihood: float
    theta_step_inf: float
    grad_infinity_norm: float
    fp_info: FixedPointInfo
    gradient: torch.Tensor
    solve_stats_F: LinearSolveStats
    solve_stats_G: LinearSolveStats
