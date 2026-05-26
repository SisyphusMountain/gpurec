from __future__ import annotations

from scripts import benchmark_hogenom_specieswise_multifidelity_adagrad as route


def test_fixed_schedule_replays_fixed8_route_by_default() -> None:
    args = route.build_arg_parser().parse_args([])

    assert route._fixed_phase_specs(args)[0] == ("fixed8_warmup", 8, 1.0, 60)
    assert route._initial_model_budget(args) == 8


def test_fixed_schedule_can_start_with_fixed4() -> None:
    args = route.build_arg_parser().parse_args(["--fixed-initial-budget", "4"])

    assert route._fixed_phase_specs(args)[:2] == [
        ("fixed4_initial", 4, 1.0, 40),
        ("fixed8_warmup", 8, 1.0, 60),
    ]
    assert route._initial_model_budget(args) == 4


def test_fixed_schedule_zero_initial_steps_keeps_fixed8_start() -> None:
    args = route.build_arg_parser().parse_args(
        ["--fixed-initial-budget", "4", "--fixed-initial-steps", "0"]
    )

    assert route._fixed_phase_specs(args)[0] == ("fixed8_warmup", 8, 1.0, 60)
    assert route._initial_model_budget(args) == 8


def test_adaptive_schedule_model_budget_uses_adaptive_initial_budget() -> None:
    args = route.build_arg_parser().parse_args(
        [
            "--schedule-mode",
            "adaptive",
            "--adaptive-initial-budget",
            "4",
            "--fixed-initial-budget",
            "0",
        ]
    )

    assert route._initial_model_budget(args) == 4
