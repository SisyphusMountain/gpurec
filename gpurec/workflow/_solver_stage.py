from __future__ import annotations

from dataclasses import dataclass

from ._phase import _uses_adagrad_restart_prefix
from .config import RunConfig


@dataclass
class SolverStageController:
    config: RunConfig

    def fixed_iters_E_for_pi_budget(self, pi_iters: int) -> int | None:
        pi_iters = int(pi_iters)
        if self.config.mode == "specieswise" and pi_iters > 16:
            if self.config.fixed_iters_e is None:
                return pi_iters
            return max(int(self.config.fixed_iters_e), pi_iters)
        return self.config.fixed_iters_e

    def final_check_fixed_iters_E(self, check_iters: int) -> int | None:
        if _uses_adagrad_restart_prefix(self.config.optimizer):
            return int(check_iters)
        return self.fixed_iters_E_for_pi_budget(check_iters)

    def final_iteration_check_iters(self) -> int:
        if _uses_adagrad_restart_prefix(self.config.optimizer):
            return int(self.config.adagrad_restart_final_check_iters)
        return int(self.config.final_check_iters)

    def configure_specieswise_final_eval_solver_stage(self, model) -> bool:
        if self.config.mode != "specieswise":
            return False
        if (
            not _uses_adagrad_restart_prefix(self.config.optimizer)
            and self.config.final_check_iters <= 0
        ):
            return False
        configure_solver = getattr(model, "configure_solver_iterations", None)
        if not callable(configure_solver):
            return False
        if _uses_adagrad_restart_prefix(self.config.optimizer):
            check_iters = int(self.config.adagrad_restart_final_check_iters)
        else:
            check_iters = self.final_iteration_check_iters()
        if check_iters <= 0:
            return False
        if _uses_adagrad_restart_prefix(self.config.optimizer):
            configure_solver(
                fixed_iters_E=check_iters,
                fixed_iters_Pi=check_iters,
                neumann_terms=check_iters,
            )
            return True
        configure_solver(
            fixed_iters_E=self.fixed_iters_E_for_pi_budget(
                self.config.fixed_iters_pi
            ),
            fixed_iters_Pi=self.config.fixed_iters_pi,
            neumann_terms=self.config.neumann_terms,
        )
        return True

    def configure_specieswise_adagrad_lbfgsb_tail_solver(self, model) -> None:
        configure_solver = getattr(model, "configure_solver_iterations", None)
        if not callable(configure_solver):
            raise RuntimeError(
                "adagrad-restarts-lbfgsb requires a model with "
                "configure_solver_iterations"
            )
        configure_solver(
            fixed_iters_E=self.fixed_iters_E_for_pi_budget(
                self.config.fixed_iters_pi
            ),
            fixed_iters_Pi=int(self.config.fixed_iters_pi),
            neumann_terms=int(self.config.neumann_terms),
        )

    def configure_specieswise_adagrad_restart_phase(self, model, phase) -> None:
        configure_solver = getattr(model, "configure_solver_iterations", None)
        if not callable(configure_solver):
            raise RuntimeError(
                "adagrad-restarts requires a model with configure_solver_iterations"
            )
        configure_solver(
            fixed_iters_E=int(phase.phase.fixed_iters_e),
            fixed_iters_Pi=int(phase.phase.fixed_iters_pi),
            neumann_terms=int(phase.phase.neumann_terms),
        )

    def uses_warmup(self) -> bool:
        config = self.config
        if config.solver_warmup_iters <= 0:
            return False
        if _uses_adagrad_restart_prefix(config.optimizer):
            return False
        if config.mode == "genewise" and config.optimizer in {
            "batched-lbfgs",
            "adam-fd-newton",
            "hessian-sgd",
        }:
            return True
        return config.mode == "specieswise" and config.fixed_iters_pi > config.solver_warmup_iters

    def configure_stage(self, model, stage: str) -> None:
        config = self.config
        if stage == "warmup":
            iters = self.hessian_sgd_warmup_iters(model)
            fixed_iters_E = (
                config.fixed_iters_e
                if config.optimizer == "hessian-sgd"
                else iters
            )
            model.configure_solver_iterations(
                fixed_iters_E=fixed_iters_E,
                fixed_iters_Pi=iters,
                neumann_terms=iters,
            )
            return
        if stage == "full":
            model.configure_solver_iterations(
                fixed_iters_E=self.fixed_iters_E_for_pi_budget(config.fixed_iters_pi),
                fixed_iters_Pi=config.fixed_iters_pi,
                neumann_terms=config.neumann_terms,
            )
            return
        raise ValueError(f"unknown solver stage {stage!r}")

    def hessian_sgd_warmup_iters(self, model) -> int:
        iters = int(self.config.solver_warmup_iters)
        if self.config.optimizer != "hessian-sgd":
            return iters
        active_clade_count = int(
            getattr(model.current_batch_metadata, "clade_count", 0) or 0
        )
        if active_clade_count >= 400000:
            return min(iters, 2)
        return iters

    def configure_active_stage(self, model, stage: str) -> None:
        config = self.config
        if config.optimizer == "hessian-sgd" and stage == "full":
            fixed_iters_pi = (
                config.hessian_sgd_normal_fixed_iters_pi
                if config.hessian_sgd_normal_fixed_iters_pi is not None
                else config.fixed_iters_pi
            )
            model.configure_solver_iterations(
                fixed_iters_E=self.fixed_iters_E_for_pi_budget(fixed_iters_pi),
                fixed_iters_Pi=fixed_iters_pi,
                neumann_terms=(
                    config.hessian_sgd_normal_neumann_terms
                    if config.hessian_sgd_normal_neumann_terms is not None
                    else config.neumann_terms
                ),
            )
            return
        self.configure_stage(model, stage)

    def hessian_sgd_validation_fixed_iters_pi(self) -> int:
        value = self.config.hessian_sgd_validation_fixed_iters_pi
        return int(value if value is not None else self.config.fixed_iters_pi)

    def hessian_sgd_validation_neumann_terms(self) -> int:
        value = self.config.hessian_sgd_validation_neumann_terms
        return int(value if value is not None else self.config.neumann_terms)

    def configure_hessian_sgd_validation_solver_stage(self, model) -> None:
        fixed_iters_pi = self.hessian_sgd_validation_fixed_iters_pi()
        model.configure_solver_iterations(
            fixed_iters_E=self.fixed_iters_E_for_pi_budget(fixed_iters_pi),
            fixed_iters_Pi=fixed_iters_pi,
            neumann_terms=self.hessian_sgd_validation_neumann_terms(),
        )

    def is_hessian_sgd_validation_step(
        self,
        *,
        phase: str,
        solver_stage: str,
        active_batch_local_step: int,
        line_search_active: bool,
    ) -> bool:
        interval = int(self.config.hessian_sgd_validation_interval)
        return (
            interval > 0
            and self.config.optimizer == "hessian-sgd"
            and phase == "hessian-sgd"
            and solver_stage == "full"
            and not line_search_active
            and active_batch_local_step % interval == 0
        )

    def hessian_sgd_validation_result_is_canonical_full_solver(self) -> bool:
        return (
            self.hessian_sgd_validation_fixed_iters_pi()
            == int(self.config.fixed_iters_pi)
            and self.hessian_sgd_validation_neumann_terms()
            == int(self.config.neumann_terms)
        )

    def active_batch_result_is_canonical_full_solver(
        self,
        *,
        phase: str,
        solver_stage: str,
    ) -> bool:
        config = self.config
        if solver_stage != "full":
            return False
        if phase != "hessian-sgd":
            return True
        normal_pi = (
            config.hessian_sgd_normal_fixed_iters_pi
            if config.hessian_sgd_normal_fixed_iters_pi is not None
            else config.fixed_iters_pi
        )
        normal_neumann = (
            config.hessian_sgd_normal_neumann_terms
            if config.hessian_sgd_normal_neumann_terms is not None
            else config.neumann_terms
        )
        return (
            normal_pi == config.fixed_iters_pi
            and normal_neumann == config.neumann_terms
        )

    def should_switch_solver_warmup(self, *, stable_loss_steps: int) -> bool:
        config = self.config
        return (
            config.solver_warmup_loss_patience > 0
            and stable_loss_steps >= config.solver_warmup_loss_patience
        )
