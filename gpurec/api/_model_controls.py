"""API model control and inspection helpers for ``GeneReconModel``.

This module is internal support for ``gpurec.api`` model methods, not a public
import surface.  It holds lightweight control and inspection methods only; it
does not own construction, resident batch lifecycle, likelihood, or autograd.
"""
from __future__ import annotations

from typing import Any

from gpurec._validation import disabled_adaptive_neumann_terms_value

from ._batch_specs import _normalize_pi_adjoint_cache_update_mode
from ._model_types import FamilyInput, _public_family_value
from ._validation import (
    bool_value,
    integer_value,
    nonnegative_float,
    positive_even_int,
    positive_float,
    positive_int,
)
from .autograd import ReconStaticState, _clear_pi_adjoint_runtime_cache

_UNSET = object()


class _GeneReconModelControlsMixin:
    """Private mixin for ``GeneReconModel`` control and inspection methods."""

    def _apply_pi_adjoint_warmstart_config(
        self,
        static: ReconStaticState,
        *,
        clear_cache: bool,
    ) -> None:
        static.pi_adjoint_warmstart = bool(self._pi_adjoint_warmstart)
        static.pi_adjoint_cache_update_mode = self._pi_adjoint_cache_update_mode
        static.pi_fixed_point_relaxation = self._pi_fixed_point_relaxation
        if clear_cache:
            _clear_pi_adjoint_runtime_cache(static)

    def configure_pi_adjoint_warmstart(
        self,
        *,
        enabled: bool,
        cache_update_mode: str = "immediate",
        pi_fixed_point_relaxation: float | None = None,
    ) -> None:
        """Update Pi-adjoint warm-start policy on defaults and built batches."""
        warmstart = bool_value("pi_adjoint_warmstart", enabled)
        cache_mode = _normalize_pi_adjoint_cache_update_mode(cache_update_mode)
        if pi_fixed_point_relaxation is not None:
            pi_fixed_point_relaxation = positive_float(
                "pi_fixed_point_relaxation",
                pi_fixed_point_relaxation,
            )
        self._pi_adjoint_warmstart = warmstart
        self._pi_adjoint_cache_update_mode = cache_mode
        if pi_fixed_point_relaxation is not None:
            self._pi_fixed_point_relaxation = pi_fixed_point_relaxation
        for static in self.cached_static_states:
            self._apply_pi_adjoint_warmstart_config(static, clear_cache=True)

    def configure_solver_iterations(
        self,
        *,
        fixed_iters_E: int | None | object = _UNSET,
        fixed_iters_Pi: int | None = None,
        neumann_terms: int | None = None,
        pi_max_diff_tol: float | None = None,
        gradient_change_tol: float | None = None,
        adaptive_neumann_terms: bool | None = None,
    ) -> None:
        """Update solver iteration controls on the model and built batches.

        The method updates model defaults and resident batch static states that
        are already built.  It does not cancel or rewrite pending background
        prefetch work; configure before scheduling lazy prefetch, or materialize
        resident batches and configure again when all batches should share the
        new controls.
        """
        if fixed_iters_E is not _UNSET:
            if fixed_iters_E is not None:
                fixed_iters_E = positive_int("fixed_iters_E", fixed_iters_E)
            self._fixed_iters_E = fixed_iters_E
        if fixed_iters_Pi is not None:
            fixed_iters_Pi = positive_even_int("fixed_iters_Pi", fixed_iters_Pi)
            self._fixed_iters_Pi = fixed_iters_Pi
        if neumann_terms is not None:
            neumann_terms = positive_int("neumann_terms", neumann_terms)
            self._neumann_terms = neumann_terms
        if pi_max_diff_tol is not None:
            pi_max_diff_tol = nonnegative_float("pi_max_diff_tol", pi_max_diff_tol)
            self._pi_max_diff_tol = pi_max_diff_tol
        if gradient_change_tol is not None:
            gradient_change_tol = nonnegative_float(
                "gradient_change_tol",
                gradient_change_tol,
            )
            self._gradient_change_tol = gradient_change_tol
        if adaptive_neumann_terms is not None:
            adaptive_neumann_terms = disabled_adaptive_neumann_terms_value(
                adaptive_neumann_terms
            )
            self._adaptive_neumann_terms = adaptive_neumann_terms

        for static in self.cached_static_states:
            if fixed_iters_E is not _UNSET:
                static.fixed_iters_E = fixed_iters_E
            if fixed_iters_Pi is not None:
                static.fixed_iters_Pi = fixed_iters_Pi
            if neumann_terms is not None:
                static.neumann_terms = neumann_terms
            if pi_max_diff_tol is not None:
                static.pi_max_diff_tol = pi_max_diff_tol
            if gradient_change_tol is not None:
                static.gradient_change_tol = gradient_change_tol
            if adaptive_neumann_terms is not None:
                static.adaptive_neumann_terms = adaptive_neumann_terms

    def solver_stat_records(self) -> list[dict[str, Any]]:
        """Copies of solver stats from already-built static states."""
        records: list[dict[str, Any]] = []
        for static in self.cached_static_states:
            stats = static.last_solver_stats
            if stats is not None:
                records.append(dict(stats))
        return records

    def family_input(self, family_index: int) -> FamilyInput:
        family_index = integer_value("family_index", family_index)
        if family_index < 0 or family_index >= self.n_families:
            raise IndexError(
                f"family_index {family_index} outside 0..{self.n_families}"
            )
        ensure_full = getattr(self._dataset, "_ensure_full_families", None)
        if callable(ensure_full):
            ensure_full()
        family = self._dataset.families[family_index]
        leaf_map = self._dataset.leaf_species_maps[family_index] or {}
        return FamilyInput(
            index=family_index,
            name=self._dataset.family_names[family_index],
            gene_tree_paths=list(self._dataset.gene_tree_paths[family_index]),
            leaf_species_map=dict(leaf_map),
            clade_count=int(family["C"]),
            split_count=int(family["N_splits"]),
            root_clade_id=int(family["root_clade_id"]),
            ccp_helpers=_public_family_value(family["ccp_helpers"]),
            leaf_row_index=_public_family_value(family["leaf_row_index"]),
            leaf_col_index=_public_family_value(family["leaf_col_index"]),
            clade_leaf_labels=list(family.get("clade_leaf_labels", [])),
        )
