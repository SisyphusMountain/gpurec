"""Independent pure-torch CPU oracle for the fraction-missing DTL likelihood.

This is a TEST-ONLY reference. It recomputes the extinction ``E[S]`` and
reconciliation ``Pi[C, S]`` fixed points from scratch in plain float64 torch and
returns the reconciliation NLL. It does NOT import any Triton kernel, the wave
forward, or the production E/Pi solvers -- it only borrows this repo's Rust
``preprocess_dataset`` for the (shared, unavoidable) tree topology + CCP splits.
Cross-checking it against ``GeneReconModel`` therefore validates the production
wave path independently.

Math implemented (base-2 log space throughout, matching production):

* Event rates: ``theta = log2([D, L, T])``; a base-2 log-softmax over
  ``[0, log2 D, log2 L, log2 T]`` gives ``log_pS, log_pD, log_pL, log_pT``.
* Transfer normalizer (uniform receivers, the model default with all-zero
  ``receiver_weights``): ``max_transfer[s] = log_pT + unnorm_row_max[s]`` where
  ``unnorm_row_max`` = ``-log2(#valid receivers of donor s)`` comes from the
  preprocessor. Valid receivers of ``s`` = every species that is neither ``s``
  nor a strict ancestor of ``s``.
* E-step at species ``s`` (fixed point):
  ``E_s = lse2( log_pS + E_s1 + E_s2, log_pD + 2 E_s, E_s + Ebar_s, log_pL )``
  with ``Ebar_s = lse2_{r valid}(E_r) + max_transfer[s]``.
  **Fraction-missing E boundary**: at a species leaf ``l`` with
  ``fm_l = 1 - p_obs_l > 0`` the terminal speciation term is the single factor
  ``p^S_l * fm_l``: set ``E_s1 = log2(fm_l)``, ``E_s2 = 0``.
* Pi recurrence at clade ``gamma`` / species ``s`` (fixed point) uses the fixed
  converged ``E, Ebar, E_s1, E_s2``. Self-loop terms (all clades):
  DL ``= (1 + log_pD + E_s) + Pi``, TL ``= Pi + Ebar_s``,
  T ``= Pibar + E_s``, and the two speciation-loss terms
  ``SL1 = (log_pS + E_s2) + Pi[:, child1(s)]``,
  ``SL2 = (log_pS + E_s1) + Pi[:, child2(s)]``,
  with ``Pibar[gamma, s] = lse2_{r valid}(Pi[gamma, r]) + max_transfer[s]``.
  Internal clades additionally get the gene-split (DTS) term; leaf clades
  additionally get the leaf term.
  **E-only fraction-missing (AleRax v1.4.0)**: the Pi/CLV leaf boundary carries
  NO fraction-missing term. A leaf clade mapped to species column ``s`` keeps the
  plain speciation ``log_pS[s]`` and every other column is -inf -- exactly the
  standard no-missing boundary. Fraction-missing enters ONLY the E-step above.

No ``clamp`` anywhere: log2 is only taken on a strictly-positive branch via the
``torch.where`` + ``masked_fill(1.0)`` pattern.
"""
from __future__ import annotations

import math

import torch

from gpurec.core.scheduling.batching import preprocess_dataset

NEG_INF = float("-inf")
_DT = torch.float64


# ---------------------------------------------------------------------------
# clamp-free log-space primitives
# ---------------------------------------------------------------------------

def _safe_log2(x: torch.Tensor) -> torch.Tensor:
    positive = x > 0
    return torch.where(
        positive,
        torch.log2(x.masked_fill(~positive, 1.0)),
        torch.full_like(x, NEG_INF),
    )


def _lse2(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Base-2 logsumexp with an explicit -inf guard (no clamp)."""
    row_max = x.max(dim=dim, keepdim=True).values
    row_max_safe = torch.where(torch.isneginf(row_max), torch.zeros_like(row_max), row_max)
    total = torch.exp2(x - row_max_safe).sum(dim=dim, keepdim=True)
    result = _safe_log2(total) + row_max
    return result.squeeze(dim)


def _logaddexp2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _lse2(torch.stack([a, b], dim=0), dim=0)


def _gather_species(row: torch.Tensor, child: torch.Tensor, child_valid: torch.Tensor) -> torch.Tensor:
    """``row`` indexed at ``child[s]`` per species ``s`` (-inf where child is the sentinel).

    ``row`` may be 1-D ``[S]`` (gather over its own species axis) or 2-D
    ``[C, S]`` (gather over the last axis for every clade). Sentinel indices are
    masked to a safe 0 before the index_select (no clamp), then overwritten with
    -inf via ``torch.where``.
    """
    safe = torch.where(child_valid, child, torch.zeros_like(child))
    if row.ndim == 1:
        gathered = row.index_select(0, safe)
        return torch.where(child_valid, gathered, torch.full_like(gathered, NEG_INF))
    gathered = row.index_select(1, safe)  # [C, S]
    mask = child_valid.unsqueeze(0).expand_as(gathered)
    return torch.where(mask, gathered, torch.full_like(gathered, NEG_INF))


def oracle_nll(species_nwk: str, gene_nwk: str, D: float, L: float, T: float, fraction_missing: float) -> float:
    """Reconciliation NLL for one gene family with the fraction-missing boundary.

    Pure-torch CPU float64, independent of the Triton kernels / wave forward.
    """
    raw = preprocess_dataset(str(species_nwk), [str(gene_nwk)])
    sh = raw["species"]
    fam = raw["families"][0]

    S = int(sh["S"])
    C = int(fam["C"])

    sp_parent = sh["sp_parent"].to(torch.long)
    sp_child1 = sh["sp_child1"].to(torch.long)
    sp_child2 = sh["sp_child2"].to(torch.long)
    unnorm_row_max = sh["unnorm_row_max"].to(_DT)

    c1_valid = sp_child1 < S
    c2_valid = sp_child2 < S
    # A species is a leaf iff its first child is the Rust sentinel S.
    leaf_species_mask = ~c1_valid

    # ----- event log-probabilities: softmax over [1, D, L, T] in log2 space ---
    theta = torch.log2(torch.tensor([D, L, T], dtype=_DT))
    logits = torch.cat([torch.zeros(1, dtype=_DT), theta])  # [1, D, L, T] logits
    log_probs = (torch.log_softmax(logits * math.log(2.0), dim=-1) / math.log(2.0))
    log_pS = log_probs[0]
    log_pD = log_probs[1]
    log_pL = log_probs[2]
    log_pT = log_probs[3]
    max_transfer = log_pT + unnorm_row_max  # [S]

    # ----- valid-receiver mask: excluded[s, r] iff r == s or r ancestor of s ---
    excluded = torch.zeros((S, S), dtype=torch.bool)
    idx = torch.arange(S)
    excluded[idx, idx] = True
    for s in range(S):
        a = int(sp_parent[s].item())
        while a >= 0:
            excluded[s, a] = True
            a = int(sp_parent[a].item())
    valid = ~excluded  # [s, r]

    def transfer_complement_row(vals_over_r: torch.Tensor) -> torch.Tensor:
        """[S] receiver values -> [S] donor transfer complement (+ max_transfer)."""
        masked = torch.where(valid, vals_over_r.unsqueeze(0).expand(S, S), torch.full((S, S), NEG_INF, dtype=_DT))
        return _lse2(masked, dim=1) + max_transfer

    def transfer_complement_mat(pi: torch.Tensor) -> torch.Tensor:
        """[C, S] receiver values -> [C, S] donor transfer complement (+ max_transfer)."""
        broadcast = pi.unsqueeze(1).expand(C, S, S)  # [gamma, s, r] = Pi[gamma, r]
        masked = torch.where(valid.unsqueeze(0), broadcast, torch.full((1, S, S), NEG_INF, dtype=_DT))
        return _lse2(masked, dim=2) + max_transfer.unsqueeze(0)

    # ----- fraction-missing leaf boundary (clamp-free) -----------------------
    fm = torch.zeros(S, dtype=_DT)
    fm[leaf_species_mask] = float(fraction_missing)
    positive = fm > 0
    leaf_fm_log = torch.where(positive, torch.log2(fm.masked_fill(~positive, 1.0)), torch.full_like(fm, NEG_INF))
    missing_leaf = leaf_fm_log > NEG_INF  # [S] leaf species with fm > 0

    # ======================= E fixed point ===================================
    E = torch.full((S,), torch.finfo(_DT).min, dtype=_DT)
    for _ in range(20000):
        E_c1 = _gather_species(E, sp_child1, c1_valid)
        E_c2 = _gather_species(E, sp_child2, c2_valid)
        E_s1 = torch.where(missing_leaf, leaf_fm_log, E_c1)
        E_s2 = torch.where(missing_leaf, torch.zeros_like(E_c2), E_c2)
        Ebar = transfer_complement_row(E)
        terms = torch.stack([
            log_pS + E_s1 + E_s2,
            log_pD + 2.0 * E,
            E + Ebar,
            log_pL.expand(S),
        ], dim=0)
        E_new = _lse2(terms, dim=0)
        if torch.abs(E_new - E).max().item() < 1e-15:
            E = E_new
            break
        E = E_new

    # Final consistent boundary tensors from the converged E.
    E_c1 = _gather_species(E, sp_child1, c1_valid)
    E_c2 = _gather_species(E, sp_child2, c2_valid)
    E_s1 = torch.where(missing_leaf, leaf_fm_log, E_c1)
    E_s2 = torch.where(missing_leaf, torch.zeros_like(E_c2), E_c2)
    Ebar = transfer_complement_row(E)

    # Per-species Pi self-loop constants.
    dl_const = 1.0 + log_pD + E          # log2(2) + log_pD + E
    sl1_const = log_pS + E_s2            # child1 retained, child2 extinct
    sl2_const = log_pS + E_s1            # child2 retained, child1 extinct

    # ----- leaf-clade Pi boundary (leaf clades only) -------------------------
    # E-only model (AleRax v1.4.0): NO fraction-missing term in the Pi/CLV leaf
    # boundary. A leaf clade mapped to species column ``col`` keeps the plain
    # speciation ``log_pS[col]``; every other column is -inf (standard no-missing).
    leaf_row = [int(v) for v in fam["leaf_row_index"]]
    leaf_col = [int(v) for v in fam["leaf_col_index"]]
    clade_species_map = torch.full((C, S), NEG_INF, dtype=_DT)
    for gamma, col in zip(leaf_row, leaf_col):
        clade_species_map[gamma, col] = 0.0  # mapped clade keeps log_pS[col]
    leaf_term = log_pS + clade_species_map  # -inf where clade_species_map is -inf

    # ----- gene-tree CCP splits ---------------------------------------------
    N = int(fam["N_splits"])
    lr = [int(v) for v in fam["split_leftrights_sorted"]]
    lefts = lr[:N]
    rights = lr[N:]
    parents = [int(v) for v in fam["split_parents_sorted"]]
    split_logp = [float(v) for v in fam["log_split_probs_sorted"]]

    root_clade = int(fam["root_clade_id"])

    # ======================= Pi fixed point ==================================
    Pi = torch.full((C, S), NEG_INF, dtype=_DT)
    neg_row = torch.full((S,), NEG_INF, dtype=_DT)
    for _ in range(20000):
        Pibar = transfer_complement_mat(Pi)

        # Gene-split (DTS) term for internal clades.
        gene_split = torch.full((C, S), NEG_INF, dtype=_DT)
        for i in range(N):
            p, Lc, Rc, lp = parents[i], lefts[i], rights[i], split_logp[i]
            pi_L, pi_R = Pi[Lc], Pi[Rc]
            pibar_L, pibar_R = Pibar[Lc], Pibar[Rc]
            pi_L_c1 = _gather_species(pi_L, sp_child1, c1_valid)
            pi_R_c2 = _gather_species(pi_R, sp_child2, c2_valid)
            pi_R_c1 = _gather_species(pi_R, sp_child1, c1_valid)
            pi_L_c2 = _gather_species(pi_L, sp_child2, c2_valid)
            split_terms = torch.stack([
                lp + log_pD + pi_L + pi_R,             # duplication
                lp + pi_L + pibar_R,                   # transfer, left retained
                lp + pi_R + pibar_L,                   # transfer, right retained
                lp + log_pS + pi_L_c1 + pi_R_c2,       # speciation L->c1, R->c2
                lp + log_pS + pi_R_c1 + pi_L_c2,       # speciation R->c1, L->c2
            ], dim=0)
            contrib = _lse2(split_terms, dim=0)
            gene_split[p] = _logaddexp2(gene_split[p], contrib)

        # Self-loop terms (all clades).
        DL = dl_const.unsqueeze(0) + Pi
        TL = Pi + Ebar.unsqueeze(0)
        Tr = Pibar + E.unsqueeze(0)
        SL1 = sl1_const.unsqueeze(0) + _gather_species(Pi, sp_child1, c1_valid)
        SL2 = sl2_const.unsqueeze(0) + _gather_species(Pi, sp_child2, c2_valid)

        Pi_new = _lse2(torch.stack([DL, TL, Tr, SL1, SL2, gene_split, leaf_term], dim=0), dim=0)
        if torch.abs(Pi_new - Pi).max().item() < 1e-14:
            Pi = Pi_new
            break
        Pi = Pi_new

    # ======================= NLL head ========================================
    root_row = Pi[root_clade]  # [S]
    numerator = _lse2(root_row, dim=0) - math.log2(S)
    # survival = 1 - mean_s 2^{E_s} via compensated expm1 (no cancellation).
    survival = (-torch.expm1(E * math.log(2.0))).mean()
    denominator = _safe_log2(survival)
    nll = -(numerator - denominator)
    return float(nll.item())
