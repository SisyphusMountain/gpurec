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
from gpurec.core.scheduling.batching import plan_batch_wave_layouts, preprocess_dataset


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
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        device = torch.device(device)
        mode = str(mode).strip().lower()
        genewise, specieswise = _mode_flags(mode)
        self.solver_options = solver_options
        self.dtype = dtype

        raw = preprocess_dataset(
            str(species_tree),
            gene_trees,
            family_chunk_size=family_chunk_size,
            clade_budget=clade_budget,
            batch_packing=batch_packing,
            max_wave_size=max_wave_size,
        )
        species_raw = raw["species"]
        families = raw["families"]
        species_helpers = {k: v.to(device=device) if torch.is_tensor(v) else v for k, v in species_raw.items()}
        self.families = families
        batches = raw.get("batches") or [list(range(len(families)))]
        if not batches:
            batches = [list(range(len(families)))]
        batch_wave_layouts = raw.get("batch_wave_layouts") or [None] * len(batches)
        theta_shape = (len(families), 3) if genewise else ((int(species_helpers["S"]), 3) if specieswise else (3,))
        self.theta = torch.nn.Parameter(
            torch.full(theta_shape, math.log2(1e-10), dtype=dtype, device=device)
        )
        self.receiver_weights = torch.nn.Parameter(
            torch.zeros((int(species_helpers["S"]),), dtype=dtype, device=device)
        )
        # Per-species origination logits (softmax over the S species nodes). Default all-zeros =>
        # the uniform origination prior the likelihood assumes; enters ONLY the NLL aggregation.
        self.origination_weights = torch.nn.Parameter(
            torch.zeros((int(species_helpers["S"]),), dtype=dtype, device=device)
        )
        self.species_helpers = species_helpers
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
            device=device,
            max_wave_size=max_wave_size,
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
        raw_plan = plan_batch_wave_layouts(
            self.families,
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

    def classify_families(self, report, *, fwd_tol: float = 1e-3, bwd_tol: float = 1e-3, overshoot_tol: float = 1e3):
        """Label each family 0/1/2 from a ``convergence_report``.

        0 = converged (forward & backward residuals below tol, no overshoot);
        1 = mild-stiff (a residual exceeds tol but the adjoint stays bounded -> more
        iterations help); 2 = severe non-normal (non-finite residual or ``vk_magnitude``
        overshoot -> more Neumann terms will not help, use GMRES).
        """
        fwd = report["forward_resid"]
        bwd = report["backward_relres"]
        vk = report["vk_magnitude"]
        labels = torch.zeros(fwd.numel(), dtype=torch.long, device=fwd.device)
        severe = (~torch.isfinite(bwd)) | (~torch.isfinite(vk)) | (vk > float(overshoot_tol))
        mild = (~severe) & ((fwd > float(fwd_tol)) | (bwd > float(bwd_tol)))
        labels = torch.where(mild, torch.ones_like(labels), labels)
        labels = torch.where(severe, torch.full_like(labels, 2), labels)
        return labels

    def _default_group_options(self):
        base = self.solver_options
        even = lambda v: int(v) + (int(v) % 2)
        return {
            0: _replace_dataclass(base),
            1: _replace_dataclass(
                base,
                pi_iters=even(max(int(base.pi_iters) * 2, 32)),
                neumann_terms=max(int(base.neumann_terms) * 2, 32),
            ),
            2: _replace_dataclass(
                base,
                self_loop_solver="gmres",
                pi_iters=even(max(int(base.pi_iters) * 2, 64)),
                neumann_terms=max(int(base.neumann_terms) * 2, 32),
            ),
        }

    def adapt_solver_to_convergence(
        self,
        *,
        pi_iters_high: int = 400,
        neumann_terms=None,
        thresholds=None,
        group_options=None,
    ):
        """Detect -> classify -> rebatch -> assign per-tier solver options.

        Measures convergence (``convergence_report``), labels families into 3 tiers
        (``classify_families``), rebatches by label, clears warm starts (batch membership
        changed), and gives each batch its tier's ``SolverOptions``. Re-run periodically
        during optimization, since stiffness drifts with ``theta``. Returns a dict with the
        labels, the new batch count, and per-tier family counts.
        """
        report = self.convergence_report(pi_iters_high=pi_iters_high, neumann_terms=neumann_terms)
        labels = self.classify_families(report, **(thresholds or {}))
        if group_options is None:
            group_options = self._default_group_options()
        present = sorted(set(int(x) for x in labels.tolist()))
        self.replan_batches(family_group_assignments=labels.tolist())
        self.clear_warm_starts()  # batch membership changed -> old warm_E is invalid
        self.set_batch_solver_options({g: group_options[g] for g in present})
        return {
            "labels": labels,
            "n_batches": len(self.family_batches),
            "counts": {g: int((labels == g).sum()) for g in present},
            "report": report,
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
