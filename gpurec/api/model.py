import math
from dataclasses import replace as _replace_dataclass

import torch

from gpurec.api._autograd import _GeneReconFullLossFunction, _GeneReconFunction
from gpurec.api._batch_state import _BatchStatic, build_batch_static
from gpurec.api._execution import (
    stream_batches,
    stream_genewise_loss_vector_grad,
    theta_for_static,
)
from gpurec.api.solver_options import SolverOptions
from gpurec.config import GpurecConfig, PrecisionOptions, torch_dtype_name
from gpurec.config.rates import RateBounds
from gpurec.core.scheduling.batching import (
    plan_batch_wave_layouts,
    plan_batches_from_parsed,
    preprocess_dataset,
    preprocess_from_parsed,
)


_MODE_FLAGS = {
    "global": (False, False),
    "specieswise": (False, True),
    "genewise": (True, False),
}
_UNSET = object()


def _mode_flags(mode: str) -> tuple[bool, bool]:
    try:
        return _MODE_FLAGS[str(mode).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"mode must be one of {sorted(_MODE_FLAGS)}, got {mode!r}") from exc


class GeneReconModel(torch.nn.Module):
    def __init__(
        self,
        species_tree,
        gene_trees,
        *,
        mode: str = "global",
        device="cuda",
        family_chunk_size: int | None = 300,
        clade_budget: int | None = 315_000,
        batch_packing: str = "depth_first_fit",
        max_wave_size: int = 8192,
        solver_options: SolverOptions | dict | None = None,
        config: GpurecConfig | None = None,
        dtype: torch.dtype | str | None = None,
        per_family_origination: bool = False,
        fraction_missing=None,
        parsed_families=None,
        family_indices=None,
    ):
        super().__init__()
        device = torch.device(device)
        mode = str(mode).strip().lower()
        genewise, specieswise = _mode_flags(mode)
        if per_family_origination and not genewise:
            raise ValueError("per_family_origination requires mode='genewise'")
        # ``config`` supplies solver and precision defaults here. Explicit
        # ``solver_options`` and ``dtype`` arguments win over their matching
        # config values; the remaining config sections belong to fit/curvature
        # entry points rather than this constructor.
        if config is not None and solver_options is None:
            solver_options = config.solver
        precision_options = _replace_dataclass(
            config.precision if config is not None else PrecisionOptions()
        )
        if dtype is not None:
            precision_options.model_dtype = torch_dtype_name(dtype)
        precision_options.validate()
        dtype = precision_options.model_torch_dtype
        self.solver_options = solver_options
        self.precision_options = precision_options
        self.accumulator_dtype = precision_options.accumulator_torch_dtype
        self.dtype = dtype

        # ``parsed_families`` (a gpurec.core.scheduling.batching.parse_families handle) plus the
        # ``family_indices`` selecting a subset of it replaces re-reading and re-parsing the gene
        # tree files: same payload, no file I/O, no JSON. ``gene_trees`` is then only bookkeeping.
        if (parsed_families is None) != (family_indices is None):
            raise ValueError(
                "parsed_families and family_indices must be given together (or neither)"
            )
        if parsed_families is None:
            raw = preprocess_dataset(
                str(species_tree),
                gene_trees,
                accumulator_dtype=self.accumulator_dtype,
                family_chunk_size=family_chunk_size,
                clade_budget=clade_budget,
                batch_packing=batch_packing,
                max_wave_size=max_wave_size,
            )
        else:
            family_indices = [int(index) for index in family_indices]
            raw = preprocess_from_parsed(
                parsed_families,
                family_indices,
                family_chunk_size=family_chunk_size,
                clade_budget=clade_budget,
                batch_packing=batch_packing,
                max_wave_size=max_wave_size,
                family_group_assignments=None,
                accumulator_dtype=self.accumulator_dtype,
            )
        self.parsed_families = parsed_families
        self.parsed_family_indices = family_indices
        species_raw = raw["species"]
        families = raw["families"]
        species_helpers = {k: v.to(device=device) if torch.is_tensor(v) else v for k, v in species_raw.items()}
        self.families = families
        batches = raw.get("batches") or [list(range(len(families)))]
        if not batches:
            batches = [list(range(len(families)))]
        batch_wave_layouts = raw.get("batch_wave_layouts") or [None] * len(batches)
        theta_shape = (len(families), 3) if genewise else ((int(species_helpers["S"]), 3) if specieswise else (3,))
        bounds = RateBounds()
        self.theta = torch.nn.Parameter(
            torch.full(theta_shape, math.log2(bounds.init_rate), dtype=dtype, device=device)
        )
        self.receiver_weights = torch.nn.Parameter(
            torch.zeros((int(species_helpers["S"]),), dtype=dtype, device=device)
        )
        # Per-species origination logits (softmax over the S species nodes); when
        # per_family_origination=True this is instead per-family [G,S] (independent softmax per
        # family). Default all-zeros => the uniform origination prior the likelihood assumes; enters
        # ONLY the NLL aggregation.
        origination_shape = (len(families), int(species_helpers["S"])) if per_family_origination \
            else (int(species_helpers["S"]),)
        self.origination_weights = torch.nn.Parameter(
            torch.zeros(origination_shape, dtype=dtype, device=device)
        )
        self.per_family_origination = per_family_origination
        self.species_helpers = species_helpers
        from gpurec.core.parameters.fraction_missing import build_fraction_missing_tensors
        from gpurec.core.scheduling.batching import species_name_to_index as _species_name_to_index
        name_to_index = (
            _species_name_to_index(str(species_tree))
            if isinstance(fraction_missing, dict) else None
        )
        self.leaf_fm_log = build_fraction_missing_tensors(
            species_helpers,
            fraction_missing=fraction_missing,
            species_name_to_index=name_to_index,
            device=device,
            dtype=dtype,
        )
        self.mode = mode
        self.genewise = genewise
        self.specieswise = specieswise
        self.family_chunk_size = family_chunk_size
        self.clade_budget = clade_budget
        self.batch_packing = batch_packing
        self.max_wave_size = max_wave_size
        self.family_batches = [list(batch) for batch in batches]
        self.batch_wave_layouts = batch_wave_layouts
        self.batch_statics = [
            self._make_batch_static(batch, plan, device=device, max_wave_size=max_wave_size)
            for batch, plan in zip(self.family_batches, self.batch_wave_layouts)
        ]
        first = self.batch_statics[0]
        self.wave_layout = first.wave_layout
        self.rate_family_idx = first.rate_family_idx
        self.warm_E = first.warm_E
        self._resolve_warm_adjoint_gate()

    def _resolve_warm_adjoint_gate(self) -> None:
        """Memory-gate the GPUREC_WARM_ADJOINT cache by the clades x species product.

        The warm-adjoint cache (``static.warm_v``) is resident at ~``total_clades * S * dtype`` and on
        large family sets dwarfs both the static base (~0.25 GiB) and one batch's transient scratch
        (~1.8 GiB) -- full Hogenom ~19 GiB. Decide ONCE per (re)build whether it (plus the largest
        batch's backward scratch) fits the free-memory budget; if not, mark every batch static so
        ``_execution`` ignores the warm-adjoint env and runs cold. Small/medium sets (e.g. Hogenom-1055
        ~5.5 GiB) keep warm and its speed. This is the right gate (clades x species) vs a family count.
        """
        device = self.theta.device
        if device.type != "cuda":
            return
        from gpurec.core.memory_policy import warm_adjoint_fits

        S = int(self.species_helpers["S"])
        batch_clades = [
            sum(int(self.families[i]["C"]) for i in batch) for batch in self.family_batches
        ]
        total_clades = sum(batch_clades)
        ok, cache, scratch, budget = warm_adjoint_fits(
            total_clades, S, self.theta.dtype, device=device, max_batch_clades=max(batch_clades, default=0)
        )
        for static in self.batch_statics:
            static.warm_adjoint_ok = ok
            # When warm fits, the build gate verified ``cache + scratch <= budget`` with ``scratch``
            # covering the largest batch's per-wave scratch (W <= max_batch_clades). Record it so the
            # forward self-loop gate trusts this reservation instead of re-reading the depleted
            # post-cache free memory (the double-gate bug). Cleared to None when warm does not fit.
            static.warm_scratch_reserved_bytes = scratch if ok else None
        self.warm_adjoint_ok = ok
        if not ok:
            gib = 1024 ** 3
            print(
                f"[gpurec] warm-adjoint disabled (running cold): cache {cache / gib:.1f} GiB + scratch "
                f"{scratch / gib:.1f} GiB > budget {(budget or 0) / gib:.1f} GiB "
                f"[{total_clades:,} clades x {S} species, {len(self.family_batches)} batches]",
                flush=True,
            )

    def _make_batch_static(self, batch, plan=None, *, device: torch.device, max_wave_size: int) -> _BatchStatic:
        return build_batch_static(
            self.families,
            batch,
            plan,
            species_helpers=self.species_helpers,
            genewise=self.genewise,
            specieswise=self.specieswise,
            solver_options=self.solver_options,
            precision_options=self.precision_options,
            accumulator_dtype=self.accumulator_dtype,
            device=device,
            max_wave_size=max_wave_size,
            leaf_fm_log=self.leaf_fm_log,
        )

    @property
    def solver_options(self) -> SolverOptions:
        return self._solver_options

    @solver_options.setter
    def solver_options(self, options: SolverOptions | dict | None) -> None:
        if options is None:
            options = SolverOptions()
        elif isinstance(options, dict):
            options = SolverOptions(**options)
        elif not isinstance(options, SolverOptions):
            raise TypeError("solver_options must be a SolverOptions instance, dict, or None")

        options.validate()
        self._solver_options = options
        for static in getattr(self, "batch_statics", []):
            static.solver_options = options

    def configure_solver(self, **kwargs) -> SolverOptions:
        for name, value in kwargs.items():
            if not hasattr(self.solver_options, name):
                raise AttributeError(f"unknown solver option {name!r}")
            setattr(self.solver_options, name, value)
        self.solver_options.validate()
        return self.solver_options

    def clear_warm_starts(self) -> None:
        for static in self.batch_statics:
            static.warm_E = None
        self.warm_E = None

    def _theta_for_static(self, static: _BatchStatic, theta: torch.Tensor) -> torch.Tensor:
        return theta_for_static(static, theta, genewise=self.genewise)

    def _stream_batches(
        self,
        theta: torch.Tensor,
        receiver_weights: torch.Tensor,
        origination_weights: torch.Tensor,
        *,
        need_grad: bool,
        need_origination_grad: bool = False,
    ):
        return stream_batches(
            self.batch_statics,
            theta,
            receiver_weights,
            origination_weights,
            genewise=self.genewise,
            need_grad=need_grad,
            need_origination_grad=need_origination_grad,
        )

    def genewise_loss_vector_and_grad(
        self,
        *,
        theta: torch.Tensor | None = None,
        receiver_weights: torch.Tensor | None = None,
        origination_weights: torch.Tensor | None = None,
        need_grad: bool = True,
        update_warm_starts: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return per-family NLLs and optional independent genewise gradients.

        This is the closure helper for row-wise optimizers such as
        :class:`gpurec.optimization.BatchedLBFGS`. It is only valid in genewise
        mode because shared-theta modes do not have independent per-family
        parameter rows. Warm starts are not updated by default so line-search
        probes stay deterministic even when trial points are rejected.
        """
        if not self.genewise:
            raise ValueError("genewise_loss_vector_and_grad() requires genewise mode")
        theta = self.theta if theta is None else theta
        receiver_weights = self.receiver_weights if receiver_weights is None else receiver_weights
        origination_weights = self.origination_weights if origination_weights is None else origination_weights
        loss_vec, grad_theta, grad_receiver, _grad_origination = stream_genewise_loss_vector_grad(
            self.batch_statics,
            theta,
            receiver_weights,
            origination_weights,
            need_grad=need_grad,
            update_warm_starts=update_warm_starts,
        )
        return loss_vec, grad_theta, grad_receiver

    def genewise_loss_vector(
        self,
        *,
        theta: torch.Tensor | None = None,
        receiver_weights: torch.Tensor | None = None,
        update_warm_starts: bool = False,
    ) -> torch.Tensor:
        loss_vec, _grad_theta, _grad_receiver = self.genewise_loss_vector_and_grad(
            theta=theta,
            receiver_weights=receiver_weights,
            need_grad=False,
            update_warm_starts=update_warm_starts,
        )
        return loss_vec

    def replan_batches(
        self,
        *,
        family_chunk_size=_UNSET,
        clade_budget=_UNSET,
        batch_packing=_UNSET,
        max_wave_size=_UNSET,
        family_group_assignments=None,
    ):
        self.family_chunk_size = self.family_chunk_size if family_chunk_size is _UNSET else family_chunk_size
        self.clade_budget = self.clade_budget if clade_budget is _UNSET else clade_budget
        self.batch_packing = self.batch_packing if batch_packing is _UNSET else batch_packing
        self.max_wave_size = self.max_wave_size if max_wave_size is _UNSET else max_wave_size
        # Remember the active grouping so set_batch_solver_options() can recover each batch's label.
        self.family_group_assignments = (
            list(family_group_assignments) if family_group_assignments is not None else None
        )
        # The parsed path re-plans in Rust over the same resident families; the legacy path
        # round-trips the (JSON-serialisable) family payloads through plan_batch_layouts.
        if self.parsed_families is None:
            raw_plan = plan_batch_wave_layouts(
                self.families,
                family_chunk_size=self.family_chunk_size,
                clade_budget=self.clade_budget,
                batch_packing=self.batch_packing,
                max_wave_size=self.max_wave_size,
                family_group_assignments=self.family_group_assignments,
            )
        else:
            raw_plan = plan_batches_from_parsed(
                self.parsed_families,
                self.parsed_family_indices,
                family_chunk_size=self.family_chunk_size,
                clade_budget=self.clade_budget,
                batch_packing=self.batch_packing,
                max_wave_size=self.max_wave_size,
                family_group_assignments=self.family_group_assignments,
            )
        self.family_batches = raw_plan["batches"]
        self.batch_wave_layouts = raw_plan["batch_wave_layouts"]
        self.batch_statics = [
            self._make_batch_static(batch, plan, device=self.theta.device, max_wave_size=self.max_wave_size)
            for batch, plan in zip(self.family_batches, self.batch_wave_layouts)
        ]
        self.wave_layout = self.batch_statics[0].wave_layout
        self.rate_family_idx = self.batch_statics[0].rate_family_idx
        self.warm_E = self.batch_statics[0].warm_E
        self._resolve_warm_adjoint_gate()
        return self.family_batches

    def set_batch_solver_options(self, per_group_options):
        """Assign a distinct ``SolverOptions`` to each batch by its group label.

        ``per_group_options`` maps a group label (int) to a ``SolverOptions`` (or a
        ``dict`` of overrides). Requires a prior ``replan_batches(family_group_assignments=...)``
        so each batch's label is well defined (every family in a batch shares one label by
        construction). This sets ``static.solver_options`` per batch directly and does NOT go
        through the ``solver_options`` setter, which would clobber every batch with one object.
        """
        assignments = getattr(self, "family_group_assignments", None)
        if assignments is None:
            raise ValueError(
                "set_batch_solver_options() requires a prior "
                "replan_batches(family_group_assignments=...)"
            )
        for batch, static in zip(self.family_batches, self.batch_statics):
            label = int(assignments[batch[0]])
            if label not in per_group_options:
                raise KeyError(f"no SolverOptions provided for group label {label}")
            options = per_group_options[label]
            if isinstance(options, dict):
                options = SolverOptions(**options)
            # fresh copy per batch so batches never alias one mutable options object
            options = _replace_dataclass(options)
            options.validate()
            static.solver_options = options
        return self.batch_statics

    def convergence_report(self, *, pi_iters_high: int = 400, neumann_terms=None):
        """Per-family solver-convergence diagnostics over all batches.

        Runs a diagnostic forward (at a high ``pi_iters`` so the forward fixed point is
        converged) + backward self-loop pass per batch, mapping each batch's local
        per-family results to global family indices. Returns a dict of three
        ``[n_families]`` tensors: ``forward_resid`` (last Pi update size),
        ``backward_relres`` (last Neumann increment / partial sum), and ``vk_magnitude``
        (max adjoint norm — large => non-normal overshoot). ``neumann_terms=None`` measures
        at each batch's current setting.
        """
        from gpurec.api._execution import evaluate_static_convergence

        n = len(self.families)
        device = self.theta.device
        forward_resid = torch.zeros(n, device=device, dtype=torch.float32)
        backward_relres = torch.zeros(n, device=device, dtype=torch.float32)
        vk_magnitude = torch.zeros(n, device=device, dtype=torch.float32)
        with torch.no_grad():
            for static in self.batch_statics:
                theta_b = theta_for_static(static, self.theta.detach(), genewise=self.genewise)
                f, b, v = evaluate_static_convergence(
                    static,
                    theta_b,
                    self.receiver_weights.detach(),
                    pi_iters_high=int(pi_iters_high),
                    neumann_terms=neumann_terms,
                )
                idx = static.family_index_tensor.to(device)
                forward_resid[idx] = f.to(device)
                backward_relres[idx] = b.to(device)
                vk_magnitude[idx] = v.to(device)
        return {
            "forward_resid": forward_resid,
            "backward_relres": backward_relres,
            "vk_magnitude": vk_magnitude,
        }

    def forward(
        self,
        theta: torch.Tensor | None = None,
        receiver_weights: torch.Tensor | None = None,
        origination_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        theta = self.theta if theta is None else theta
        receiver_weights = self.receiver_weights if receiver_weights is None else receiver_weights
        origination_weights = self.origination_weights if origination_weights is None else origination_weights
        if len(self.batch_statics) == 1:
            return _GeneReconFunction.apply(theta, receiver_weights, origination_weights, self.batch_statics[0])
        if torch.is_grad_enabled() and (
            theta.requires_grad or receiver_weights.requires_grad or origination_weights.requires_grad
        ):
            return _GeneReconFullLossFunction.apply(theta, receiver_weights, origination_weights, self)
        loss, _, _, _ = self._stream_batches(theta, receiver_weights, origination_weights, need_grad=False)
        return loss
