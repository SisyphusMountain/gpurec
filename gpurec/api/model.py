"""High-level ``nn.Module`` for phylogenetic reconciliation.

Wraps a :class:`gpurec.core.model.GeneDataset` (used purely for preprocessing)
and exposes ``theta`` as an ``nn.Parameter`` so notebook users can use any
``torch.optim`` optimizer with the standard pattern::

    model = GeneReconModel.from_trees(
        species_tree="sp.nwk", gene_trees=["g1.nwk"], mode="global", device="cuda",
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        loss = model()              # NLL
        loss.backward()
        opt.step()
        model.clamp_theta_()
"""
from __future__ import annotations

import math
import os
from typing import Any, Optional

import torch

from gpurec.core.batching import (
    build_wave_layout,
    collate_gene_families,
    collate_wave,
    split_phase_waves,
)
from gpurec.core.model import GeneDataset
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import (
    E_fixed_point,
    compute_log_likelihood_root_rows,
)
from gpurec.core._helpers import _nvtx_range

from .autograd import (
    ReconStaticState,
    _GeneReconFunction,
    _apply_to_static,
    _extract_parameters,
)

_MODE_MAP: dict[str, tuple[bool, bool]] = {
    "global": (False, False),
    "specieswise": (False, True),
    "genewise": (True, False),
}


def _mode_to_flags(mode: str) -> tuple[bool, bool]:
    if mode not in _MODE_MAP:
        raise ValueError(f"Unknown mode {mode!r}. Valid: {sorted(_MODE_MAP)}")
    return _MODE_MAP[mode]


def _default_theta_init(dataset: GeneDataset, mode: str) -> torch.Tensor:
    base = math.log2(1e-10)
    genewise, specieswise = _mode_to_flags(mode)
    if genewise:
        shape = (len(dataset.families), 3)
    elif specieswise:
        shape = (int(dataset.S), 3)
    else:
        shape = (3,)
    return torch.full(shape, base, dtype=dataset.dtype, device=dataset.device)


def _build_static_state(
    dataset: GeneDataset,
    *,
    fixed_iters_E: Optional[int],
    max_iters_E: int,
    tol_E: float,
    max_iters_Pi: int,
    tol_Pi: float,
    fixed_iters_Pi: Optional[int],
    neumann_terms: int,
    use_pruning: bool,
    pruning_threshold: float,
    max_wave_size: Optional[int] = 8192,
    max_root_wave_size: Optional[int] = None,
) -> ReconStaticState:
    """Absorb the wave-layout boilerplate that lives in
    ``experiments/validate_three_modes.py:100-149`` and
    ``GeneDataset.compute_likelihood_batch:393-428``.

    Builds a single cross-family wave layout for the entire dataset and
    moves species helpers (and ``ancestors_T`` for uniform mode) onto the
    target device. The result is cached on the model and reused across
    every ``forward()`` call.
    """
    device = dataset.device
    dtype = dataset.dtype

    # 1. Cross-family wave layout
    items = [
        {
            "ccp": fam["ccp_helpers"],
            "leaf_row_index": fam["leaf_row_index"],
            "leaf_col_index": fam["leaf_col_index"],
            "root_clade_id": int(fam["root_clade_id"]),
        }
        for fam in dataset.families
    ]
    batched = collate_gene_families(items, dtype=dtype, device=device)

    fams_waves: list = []
    fams_phases: list = []
    for fam in dataset.families:
        w, p = compute_clade_waves(fam["ccp_helpers"])
        fams_waves.append(w)
        fams_phases.append(p)

    offsets = [m["clade_offset"] for m in batched["family_meta"]]
    cross_waves = collate_wave(fams_waves, offsets)

    max_n_waves = max(len(p) for p in fams_phases)
    cross_phases: list[int] = []
    for k in range(max_n_waves):
        phase_k = 1
        for fp in fams_phases:
            if k < len(fp):
                phase_k = max(phase_k, fp[k])
        cross_phases.append(phase_k)

    cross_waves, cross_phases = split_phase_waves(
        cross_waves,
        cross_phases,
        phase=None,
        max_wave_size=max_wave_size,
    )
    cross_waves, cross_phases = split_phase_waves(
        cross_waves,
        cross_phases,
        phase=3,
        max_wave_size=max_root_wave_size,
    )

    family_clade_counts = [m["C"] for m in batched["family_meta"]]
    family_clade_offsets = [m["clade_offset"] for m in batched["family_meta"]]

    wave_layout = build_wave_layout(
        waves=cross_waves,
        phases=cross_phases,
        ccp_helpers=batched["ccp"],
        leaf_row_index=batched["leaf_row_index"],
        leaf_col_index=batched["leaf_col_index"],
        root_clade_ids=batched["root_clade_ids"],
        device=device,
        dtype=dtype,
        family_clade_counts=family_clade_counts,
        family_clade_offsets=family_clade_offsets,
    )

    # 2. Species helpers on device.
    species_helpers, ancestors_T = dataset._species_helpers_for_mode(
        device=device, dtype=dtype,
    )

    # 3. Other static tensors
    unnorm_row_max = dataset.unnorm_row_max.to(device=device, dtype=dtype)

    return ReconStaticState(
        device=device,
        dtype=dtype,
        wave_layout=wave_layout,
        species_helpers=species_helpers,
        unnorm_row_max=unnorm_row_max,
        ancestors_T=ancestors_T,
        genewise=bool(dataset.genewise),
        specieswise=bool(dataset.specieswise),
        fixed_iters_E=fixed_iters_E,
        max_iters_E=max_iters_E,
        tol_E=tol_E,
        max_iters_Pi=max_iters_Pi,
        tol_Pi=tol_Pi,
        fixed_iters_Pi=fixed_iters_Pi,
        neumann_terms=neumann_terms,
        use_pruning=use_pruning,
        pruning_threshold=pruning_threshold,
    )


class GeneReconModel(torch.nn.Module):
    """A ``nn.Module`` view over a :class:`GeneDataset`.

    ``forward()`` returns the negative log-likelihood as a differentiable
    scalar. ``theta`` is registered as an ``nn.Parameter`` so any
    ``torch.optim`` optimizer can be used directly.
    """

    def __init__(
        self,
        *,
        dataset: GeneDataset,
        mode: str,
        fixed_iters_E: Optional[int] = None,
        max_iters_E: int = 2000,
        tol_E: float = 1e-8,
        max_iters_Pi: int = 2000,
        tol_Pi: float = 1e-6,
        fixed_iters_Pi: Optional[int] = 6,
        neumann_terms: int = 3,
        use_pruning: bool = True,
        pruning_threshold: float = 1e-6,
        theta_init: Optional[torch.Tensor] = None,
        max_wave_size: Optional[int] = 8192,
        max_root_wave_size: Optional[int] = None,
    ):
        super().__init__()
        # Validate mode early
        _mode_to_flags(mode)
        if fixed_iters_E is not None:
            fixed_iters_E = int(fixed_iters_E)
            if fixed_iters_E < 1:
                raise ValueError("fixed_iters_E must be >= 1 when provided")

        # Sanity check: dataset flags must be consistent with mode
        ds_g, ds_sw = (dataset.genewise, dataset.specieswise)
        expected_g, expected_sw = _mode_to_flags(mode)
        if (ds_g, ds_sw) != (expected_g, expected_sw):
            raise ValueError(
                f"Dataset flags (genewise={ds_g}, specieswise={ds_sw}) do not "
                f"match requested mode {mode!r} "
                f"(expected genewise={expected_g}, specieswise={expected_sw}). "
                "Construct GeneDataset with matching flags or use "
                "GeneReconModel.from_trees()."
            )

        self._mode = mode
        self._dataset = dataset

        if theta_init is None:
            theta_init = _default_theta_init(dataset, mode)
        self.theta = torch.nn.Parameter(theta_init.clone())

        self._static = _build_static_state(
            dataset,
            fixed_iters_E=fixed_iters_E,
            max_iters_E=max_iters_E,
            tol_E=tol_E,
            max_iters_Pi=max_iters_Pi,
            tol_Pi=tol_Pi,
            fixed_iters_Pi=fixed_iters_Pi,
            neumann_terms=neumann_terms,
            use_pruning=use_pruning,
            pruning_threshold=pruning_threshold,
            max_wave_size=max_wave_size,
            max_root_wave_size=max_root_wave_size,
        )

    # ──────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────
    @classmethod
    def from_trees(
        cls,
        species_tree: str,
        gene_trees: list[str],
        *,
        mode: str = "global",
        device: Any = "cuda",
        dtype: torch.dtype = torch.float32,
        theta_init_rates: Optional[tuple[float, float, float]] = None,
        preprocess_cache_dir: str | os.PathLike | None = None,
        refresh_preprocess_cache: bool = False,
        **solver_kwargs,
    ) -> "GeneReconModel":
        """One-liner: Newick paths → ready-to-optimize model.

        Parameters
        ----------
        species_tree : str
            Path to the species tree (Newick).
        gene_trees : list[str]
            Paths to gene trees (Newick).
        mode : str
            "global" | "specieswise" | "genewise".
        device : str | torch.device
            Target device. Defaults to ``"cuda"``.
        dtype : torch.dtype
            Floating-point dtype. ``torch.float32`` is the default; switch to
            ``torch.float64`` if optimization stalls due to precision.
        theta_init_rates : (D, L, T) | None
            Optional natural-space initial rates. If ``None``, the dataset
            default of ``log2(1e-10)`` is used (matching GeneDataset).
        preprocess_cache_dir : str | os.PathLike | None
            Optional directory for CPU preprocessing cache files. Reusing the
            same cache avoids reparsing/rebuilding unchanged gene trees.
        refresh_preprocess_cache : bool
            Ignore existing preprocessing cache entries and overwrite them.
        """
        genewise, specieswise = _mode_to_flags(mode)
        if isinstance(device, str):
            device = torch.device(device)
        ds = GeneDataset(
            species_tree_path=species_tree,
            gene_tree_paths=gene_trees,
            genewise=genewise,
            specieswise=specieswise,
            dtype=dtype,
            device=device,
            preprocess_cache_dir=preprocess_cache_dir,
            refresh_preprocess_cache=refresh_preprocess_cache,
        )
        theta_init = None
        if theta_init_rates is not None:
            D, L, T = theta_init_rates
            base = torch.log2(
                torch.tensor([D, L, T], dtype=dtype, device=device)
            )
            if mode == "specieswise":
                theta_init = base.unsqueeze(0).expand(int(ds.S), -1).clone()
            elif mode == "genewise":
                theta_init = (
                    base.unsqueeze(0).expand(len(gene_trees), -1).clone()
                )
            else:
                theta_init = base
        return cls(
            dataset=ds,
            mode=mode,
            theta_init=theta_init,
            **solver_kwargs,
        )

    # ──────────────────────────────────────────────────────────────────
    # Likelihood / loss
    # ──────────────────────────────────────────────────────────────────
    def forward(self, reduce: str = "sum") -> torch.Tensor:
        """Returns negative log-likelihood (a loss).

        ``reduce="sum"`` (default) returns a scalar (sum over families).
        ``reduce="per_family"`` returns a ``[G]`` vector and is only valid in
        genewise mode.
        """
        if not torch.is_grad_enabled():
            return self._forward_inference(reduce=reduce)
        return _GeneReconFunction.apply(self.theta, self._static, reduce)

    @torch.no_grad()
    def _forward_inference(self, reduce: str = "sum") -> torch.Tensor:
        """No-grad NLL path that avoids backward-only forward outputs.

        The differentiable forward must save the full wave-ordered ``Pi`` and
        ``Pibar`` tensors for the implicit backward pass. In inference/no-grad
        contexts those tensors are dead: the likelihood only needs root rows.
        This mirrors the optimized uniform forward path used by
        ``GeneDataset.compute_likelihood_batch`` and the profiling harnesses.
        """
        if reduce not in ("sum", "per_family"):
            raise ValueError(f"reduce must be 'sum' or 'per_family', got {reduce!r}")
        if reduce == "per_family" and self._mode != "genewise":
            raise ValueError(
                "reduce='per_family' is only valid in genewise mode."
            )

        static = self._static
        device = static.device
        dtype = static.dtype

        with _nvtx_range("inference extract parameters"):
            log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = (
                _extract_parameters(self.theta.detach(), static)
            )

        with _nvtx_range("inference E fixed point"):
            e_max_iters = (
                static.fixed_iters_E
                if static.fixed_iters_E is not None
                else static.max_iters_E
            )
            e_tolerance = -1.0 if static.fixed_iters_E is not None else static.tol_E
            E_out = E_fixed_point(
                species_helpers=static.species_helpers,
                log_pS=log_pS,
                log_pD=log_pD,
                log_pL=log_pL,
                transfer_mat=transfer_mat,
                max_transfer_mat=max_transfer_vec,
                max_iters=e_max_iters,
                tolerance=e_tolerance,
                warm_start_E=(
                    None if static.fixed_iters_E is not None else static.warm_E
                ),
                dtype=dtype,
                device=device,
                ancestors_T=static.ancestors_T,
            )
            E = E_out["E"]

        with _nvtx_range("inference Pi waves root rows"):
            Pi_out = Pi_wave_forward(
                wave_layout=static.wave_layout,
                species_helpers=static.species_helpers,
                E=E,
                Ebar=E_out["E_bar"],
                E_s1=E_out["E_s1"],
                E_s2=E_out["E_s2"],
                log_pS=log_pS,
                log_pD=log_pD,
                log_pL=log_pL,
                transfer_mat=transfer_mat,
                max_transfer_mat=max_transfer_vec,
                device=device,
                dtype=dtype,
                local_iters=static.max_iters_Pi,
                local_tolerance=static.tol_Pi,
                fixed_iters=static.fixed_iters_Pi,
                return_original=False,
                need_pibar=False,
                return_root_rows=True,
                family_idx=(
                    static.wave_layout.get("family_idx") if static.genewise else None
                ),
            )

        with _nvtx_range("inference root likelihood"):
            nll_vec = compute_log_likelihood_root_rows(
                Pi_out["Pi_root_rows"],
                E,
            )
            static.warm_E = (
                None if static.fixed_iters_E is not None else E.detach()
            )
            return nll_vec.sum() if reduce == "sum" else nll_vec

    def nll(self) -> torch.Tensor:
        """Alias for ``self()``."""
        return self.forward(reduce="sum")

    def nll_per_family(self) -> torch.Tensor:
        """Per-family NLL ``[G]``. Only valid in genewise mode."""
        if self._mode != "genewise":
            raise ValueError(
                "nll_per_family() is only valid in genewise mode; in global / "
                "specieswise mode all families share theta, so independent "
                "per-family gradients are not defined."
            )
        return self.forward(reduce="per_family")

    @torch.no_grad()
    def pi_matrix(self, *, original_order: bool = True) -> torch.Tensor:
        """Return converged Pi rows for the retained uniform-transfer path.

        The model mode controls parameter sharing:
        ``global`` uses one theta vector, ``specieswise`` uses ``[S, 3]``
        theta, and ``genewise`` uses ``[G, 3]`` theta addressed by the cached
        family index.  The returned tensor is ``[C_total, S]``.
        """
        static = self._static
        log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = (
            _extract_parameters(self.theta.detach(), static)
        )
        e_max_iters = (
            static.fixed_iters_E
            if static.fixed_iters_E is not None
            else static.max_iters_E
        )
        e_tolerance = -1.0 if static.fixed_iters_E is not None else static.tol_E
        E_out = E_fixed_point(
            species_helpers=static.species_helpers,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            transfer_mat=transfer_mat,
            max_transfer_mat=max_transfer_vec,
            max_iters=e_max_iters,
            tolerance=e_tolerance,
            warm_start_E=None,
            dtype=static.dtype,
            device=static.device,
            ancestors_T=static.ancestors_T,
        )
        Pi_out = Pi_wave_forward(
            wave_layout=static.wave_layout,
            species_helpers=static.species_helpers,
            E=E_out["E"],
            Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"],
            E_s2=E_out["E_s2"],
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            transfer_mat=transfer_mat,
            max_transfer_mat=max_transfer_vec,
            device=static.device,
            dtype=static.dtype,
            local_iters=static.max_iters_Pi,
            local_tolerance=static.tol_Pi,
            fixed_iters=static.fixed_iters_Pi,
            return_original=original_order,
            need_pibar=False,
            return_root_rows=False,
            family_idx=(
                static.wave_layout.get("family_idx") if static.genewise else None
            ),
        )
        return Pi_out["Pi"] if original_order else Pi_out["Pi_wave_ordered"]

    @torch.no_grad()
    def log_likelihood(self) -> float:
        """Inference helper: returns ``+log_likelihood`` (Python float)."""
        return float(-self.forward(reduce="sum").item())

    # ──────────────────────────────────────────────────────────────────
    # Parameter management
    # ──────────────────────────────────────────────────────────────────
    def clamp_theta_(
        self,
        min_rate: float = 1e-10,
        max_rate: Optional[float] = None,
    ) -> None:
        """In-place safety floor on theta to prevent rate underflow.

        Useful after ordinary PyTorch optimizer steps to keep rates in a
        numerically valid range.
        """
        if min_rate <= 0:
            raise ValueError("min_rate must be strictly positive")
        if max_rate is not None and max_rate < min_rate:
            raise ValueError("max_rate must be greater than or equal to min_rate")
        with torch.no_grad():
            self.theta.clamp_(
                min=math.log2(min_rate),
                max=None if max_rate is None else math.log2(max_rate),
            )

    @property
    def rates(self) -> torch.Tensor:
        """Natural-space rates: ``2^theta``. Shape mirrors ``self.theta``."""
        return torch.exp2(self.theta.detach())

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def n_families(self) -> int:
        return len(self._dataset.families)

    @property
    def n_species(self) -> int:
        return int(self._dataset.S)

    @property
    def static(self) -> ReconStaticState:
        """Read-only access to the cached static state (for advanced use)."""
        return self._static

    # ──────────────────────────────────────────────────────────────────
    # Device / dtype handling
    # ──────────────────────────────────────────────────────────────────
    def _apply(self, fn):
        """Override so that ``.to(device)`` / ``.to(dtype)`` walks the
        non-Parameter tensors held inside ``self._static`` (wave_layout,
        species_helpers, etc.)."""
        super()._apply(fn)
        self._static = _apply_to_static(self._static, fn)
        # Reset warm-start cache when moving (its old device/dtype is stale)
        self._static.warm_E = None
        return self
