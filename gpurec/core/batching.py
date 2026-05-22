from typing import Any, Dict, List, Sequence

import torch


def collate_gene_families(
    batch: List[Dict[str, Any]],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """
    Collate multiple preprocessed gene-family CCP payloads into one batched layout.

    Each item in `batch` must be a dict with keys:
      - 'ccp': dict containing at least
          'C' (int), 'N_splits' (int),
          'split_leftrights_sorted' (Long[2*N_i]),
          'log_split_probs_sorted' (Float[N_i]),
          'split_parents_sorted' (Long[N_i]),
          'num_segs_eq1' (int),
          'end_rows_ge2' (int).
      - 'leaf_row_index': Long[K_i]
      - 'leaf_col_index': Long[K_i]
      - 'root_clade_id': int

    Returns a dict with:
      - 'ccp': merged CCP helpers consumed by build_wave_layout()
      - 'root_clade_ids': Long[F] (roots offset into concatenated clade axis)
      - 'family_meta': bookkeeping (clade offsets, per-family sizes)
    """
    if not batch:
        raise ValueError("batch must contain at least one family")

    # ---- running offsets and accumulators ----
    clade_offset = 0

    root_ids: List[int] = []
    leaf_row_parts, leaf_col_parts = [], []
    # (>=2) block
    # we will accumulate left and right halves separately to preserve [all_lefts ; all_rights] order
    ge2_left_parts: List[torch.Tensor] = []
    ge2_right_parts: List[torch.Tensor] = []
    ge2_logp_parts: List[torch.Tensor] = []

    # (=1) block
    eq1_left_parts: List[torch.Tensor] = []
    eq1_right_parts: List[torch.Tensor] = []
    eq1_logp_parts: List[torch.Tensor] = []

    # split_parents_sorted (one entry per split, needed for wave scheduling)
    ge2_split_parents_parts: List[torch.Tensor] = []
    eq1_split_parents_parts: List[torch.Tensor] = []

    # Totals & per-family meta
    total_C = 0
    total_N = 0

    family_meta: List[Dict[str, int]] = []

    for family_index, item in enumerate(batch):
        ccp = item["ccp"]
        root_i: int = int(item["root_clade_id"])

        # Family sizes
        C_i = int(ccp["C"])
        N_i = int(ccp["N_splits"])
        num_eq1_i = int(ccp["num_segs_eq1"])
        end_rows_ge2_i = int(ccp["end_rows_ge2"])
        _validate_split_block_lengths(
            family_index=family_index,
            n_splits=N_i,
            num_eq1=num_eq1_i,
            end_rows_ge2=end_rows_ge2_i,
        )

        # Pull split arrays
        leftrights_i = ccp["split_leftrights_sorted"].to(torch.long).cpu()  # [2*N_i]
        logp_i = ccp["log_split_probs_sorted"].to(dtype).cpu()              # [N_i]
        _require_numel(
            "split_leftrights_sorted",
            leftrights_i,
            2 * N_i,
            family_index=family_index,
        )
        _require_numel(
            "log_split_probs_sorted",
            logp_i,
            N_i,
            family_index=family_index,
        )

        # Split left/right halves so we can cut the ge2 vs eq1 ranges cleanly
        lefts_i = leftrights_i[:N_i]
        rights_i = leftrights_i[N_i:]

        # Offsets on clade indices for this family
        lefts_i = lefts_i + clade_offset
        rights_i = rights_i + clade_offset

        if "split_parents_sorted" not in ccp:
            raise RuntimeError("preprocessed CCP helpers must include split_parents_sorted")
        sp_sorted_i = ccp["split_parents_sorted"].to(torch.long).cpu()
        _require_numel(
            "split_parents_sorted",
            sp_sorted_i,
            N_i,
            family_index=family_index,
        )

        # (>=2) rows for this family
        if end_rows_ge2_i > 0:
            ge2_left = lefts_i[:end_rows_ge2_i]
            ge2_right = rights_i[:end_rows_ge2_i]
            ge2_left_parts.append(ge2_left)
            ge2_right_parts.append(ge2_right)
            ge2_logp_parts.append(logp_i[:end_rows_ge2_i])
            ge2_split_parents_parts.append(sp_sorted_i[:end_rows_ge2_i] + clade_offset)

        # (=1) rows for this family (exactly one split per clade)
        if num_eq1_i > 0:
            start = end_rows_ge2_i
            stop = end_rows_ge2_i + num_eq1_i
            eq1_left = lefts_i[start:stop]
            eq1_right = rights_i[start:stop]
            eq1_left_parts.append(eq1_left)
            eq1_right_parts.append(eq1_right)
            eq1_logp_parts.append(logp_i[start:stop])
            eq1_split_parents_parts.append(sp_sorted_i[start:stop] + clade_offset)

        # Species/clade leaf mappings
        lr = item["leaf_row_index"].to(torch.long).to(device) + clade_offset
        lc = item["leaf_col_index"].to(torch.long).to(device)
        leaf_row_parts.append(lr)
        leaf_col_parts.append(lc)
        root_ids.append(root_i + clade_offset)

        # Totals and offsets
        total_C += C_i
        total_N += N_i

        family_meta.append({
            "C": C_i,
            "clade_offset": clade_offset,
        })

        clade_offset += C_i  # advance clade offset for next family

    # ---- Build the batched arrays in the global order: [all GE2 rows] then [all EQ1 rows] ----
    # split_leftrights_sorted must be [all_lefts ; all_rights], each of length total_N
    if len(ge2_left_parts) > 0:
        ge2_left = torch.cat(ge2_left_parts, dim=0)
        ge2_right = torch.cat(ge2_right_parts, dim=0)
    else:
        ge2_left = torch.empty((0,), dtype=torch.long)
        ge2_right = torch.empty((0,), dtype=torch.long)
    if len(eq1_left_parts) > 0:
        eq1_left = torch.cat(eq1_left_parts, dim=0)
        eq1_right = torch.cat(eq1_right_parts, dim=0)
    else:
        eq1_left = torch.empty((0,), dtype=torch.long)
        eq1_right = torch.empty((0,), dtype=torch.long)
    left_all = torch.cat([ge2_left, eq1_left], dim=0)
    right_all = torch.cat([ge2_right, eq1_right], dim=0)
    split_leftrights_sorted_batch = torch.cat([left_all, right_all], dim=0)

    # log_split_probs_sorted (N_total)
    if len(ge2_logp_parts) > 0:
        ge2_logp = torch.cat(ge2_logp_parts, dim=0)
    else:
        ge2_logp = torch.empty((0,), dtype=dtype)
    if len(eq1_logp_parts) > 0:
        eq1_logp = torch.cat(eq1_logp_parts, dim=0)
    else:
        eq1_logp = torch.empty((0,), dtype=dtype)
    log_split_probs_sorted_batch = torch.cat([ge2_logp, eq1_logp], dim=0)

    # split_parents_sorted: [all ge2 split parents ; all eq1 split parents]
    if len(ge2_split_parents_parts) > 0:
        ge2_sp = torch.cat(ge2_split_parents_parts, dim=0)
    else:
        ge2_sp = torch.empty((0,), dtype=torch.long)
    if len(eq1_split_parents_parts) > 0:
        eq1_sp = torch.cat(eq1_split_parents_parts, dim=0)
    else:
        eq1_sp = torch.empty((0,), dtype=torch.long)
    split_parents_sorted_batch = torch.cat([ge2_sp, eq1_sp], dim=0)

    _require_numel("split_leftrights_sorted", split_leftrights_sorted_batch, 2 * total_N)
    _require_numel("log_split_probs_sorted", log_split_probs_sorted_batch, total_N)
    _require_numel("split_parents_sorted", split_parents_sorted_batch, total_N)

    leaf_row_index = torch.cat(leaf_row_parts, 0).to(device)
    leaf_col_index = torch.cat(leaf_col_parts, 0).to(device)
    out = {
        "ccp": {
            "C": total_C,
            "N_splits": total_N,
            "split_leftrights_sorted": split_leftrights_sorted_batch.to(device),
            "log_split_probs_sorted": log_split_probs_sorted_batch.to(device),
            "split_parents_sorted": split_parents_sorted_batch.to(device),
        },
        "leaf_row_index": leaf_row_index,
        "leaf_col_index": leaf_col_index,
        "root_clade_ids": torch.tensor(root_ids, dtype=torch.long, device=device),
        "family_meta": family_meta,
    }
    return out


def _numel(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel())
    try:
        return len(value)
    except TypeError as exc:
        raise ValueError("CCP helper values must be tensors or sized sequences") from exc


def _require_numel(
    name: str,
    value: Any,
    expected: int,
    *,
    family_index: int | None = None,
) -> None:
    actual = _numel(value)
    if actual == expected:
        return
    prefix = "" if family_index is None else f"family {family_index} "
    raise ValueError(f"{prefix}{name} has length {actual} but expected {expected}")


def _validate_split_block_lengths(
    *,
    family_index: int,
    n_splits: int,
    num_eq1: int,
    end_rows_ge2: int,
) -> None:
    if n_splits < 0:
        raise ValueError(f"family {family_index} N_splits must be non-negative")
    if num_eq1 < 0:
        raise ValueError(f"family {family_index} num_segs_eq1 must be non-negative")
    if end_rows_ge2 < 0:
        raise ValueError(f"family {family_index} end_rows_ge2 must be non-negative")
    if end_rows_ge2 + num_eq1 != n_splits:
        raise ValueError(
            f"family {family_index} split block lengths cover "
            f"{end_rows_ge2 + num_eq1} rows but N_splits={n_splits}"
        )


def _tensor_from_plan(
    values: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.tensor(list(values), dtype=dtype, device=device).contiguous()


def _build_wave_layout_rust(
    waves: List[List[int]],
    phases: List[int],
    ccp_helpers: Dict[str, Any],
    leaf_row_index: torch.Tensor,
    leaf_col_index: torch.Tensor,
    root_clade_ids: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
    family_clade_counts: List[int] | None,
    family_clade_offsets: List[int] | None,
) -> Dict[str, Any]:
    C = int(ccp_helpers["C"])
    N_splits = int(ccp_helpers["N_splits"])
    if C > torch.iinfo(torch.int32).max:
        raise ValueError(f"wave split metadata requires int32 clade ids, got C={C}")
    if "split_parents_sorted" not in ccp_helpers:
        raise RuntimeError("preprocessed CCP helpers must include split_parents_sorted")
    _require_numel(
        "log_split_probs_sorted",
        ccp_helpers["log_split_probs_sorted"],
        N_splits,
    )

    from gpurec.core.schedule_rust import build_wave_layout_plan

    plan = build_wave_layout_plan(
        waves=waves,
        phases=phases,
        c=C,
        n_splits=N_splits,
        split_leftrights_sorted=ccp_helpers["split_leftrights_sorted"],
        split_parents_sorted=ccp_helpers["split_parents_sorted"],
        leaf_row_index=leaf_row_index,
        leaf_col_index=leaf_col_index,
        root_clade_ids=root_clade_ids,
        family_clade_counts=family_clade_counts,
        family_clade_offsets=family_clade_offsets,
    )

    log_split_probs = torch.as_tensor(
        ccp_helpers["log_split_probs_sorted"],
        dtype=dtype,
        device=device,
    )
    index_dtype = torch.int32
    wave_metas: List[Dict[str, Any]] = []
    for meta_plan in plan["wave_metas"]:
        meta: Dict[str, Any] = {
            "start": int(meta_plan["start"]),
            "end": int(meta_plan["end"]),
            "W": int(meta_plan["W"]),
            "has_splits": bool(meta_plan["has_splits"]),
            "phase": int(meta_plan["phase"]),
        }
        if meta["has_splits"]:
            split_indices = _tensor_from_plan(
                meta_plan["split_indices"],
                dtype=torch.long,
                device=device,
            )
            meta["sl"] = _tensor_from_plan(
                meta_plan["sl"],
                dtype=index_dtype,
                device=device,
            )
            meta["sr"] = _tensor_from_plan(
                meta_plan["sr"],
                dtype=index_dtype,
                device=device,
            )
            meta["log_split_probs"] = log_split_probs[
                split_indices
            ].unsqueeze(1).contiguous()
            reduce_idx = _tensor_from_plan(
                meta_plan["reduce_idx"],
                dtype=index_dtype,
                device=device,
            )
            meta["reduce_idx"] = reduce_idx
            meta["n_eq1"] = int(meta_plan["n_eq1"])
            if "eq1_reduce_idx" in meta_plan:
                meta["eq1_reduce_idx"] = _tensor_from_plan(
                    meta_plan["eq1_reduce_idx"],
                    dtype=index_dtype,
                    device=device,
                )
            if "ge2_ptr" in meta_plan:
                meta["ge2_ptr"] = _tensor_from_plan(
                    meta_plan["ge2_ptr"],
                    dtype=torch.long,
                    device=device,
                )
                meta["ge2_parent_ids"] = _tensor_from_plan(
                    meta_plan["ge2_parent_ids"],
                    dtype=index_dtype,
                    device=device,
                )
                meta["ge2_max_fanout"] = int(meta_plan["ge2_max_fanout"])
        wave_metas.append(meta)

    result = {
        "perm": _tensor_from_plan(plan["perm"], dtype=torch.long, device=device),
        "C": int(plan["c"]),
        "leaf_row_index": _tensor_from_plan(
            plan["leaf_row_index"],
            dtype=torch.long,
            device=device,
        ),
        "leaf_species_index": _tensor_from_plan(
            plan["leaf_species_index"],
            dtype=torch.long,
            device=device,
        ),
        "root_clade_ids": _tensor_from_plan(
            plan["root_clade_ids"],
            dtype=torch.long,
            device=device,
        ),
        "root_clade_ids_cpu": [int(value) for value in plan["root_clade_ids_cpu"]],
        "wave_metas": wave_metas,
    }
    if "family_idx" in plan:
        result["family_idx"] = _tensor_from_plan(
            plan["family_idx"],
            dtype=torch.long,
            device=device,
        )
    return result


def build_wave_layout(
    waves: List[List[int]],
    phases: List[int],
    ccp_helpers: Dict[str, Any],
    leaf_row_index: torch.Tensor,
    leaf_col_index: torch.Tensor,
    root_clade_ids: torch.Tensor,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    family_clade_counts: List[int] | None = None,
    family_clade_offsets: List[int] | None = None,
) -> Dict[str, Any]:
    """Build wave-ordered layout: permute clade indices so each wave is contiguous.

    After this, wave k's clades occupy Pi[wave_starts[k] : wave_starts[k+1]]
    as a contiguous block, eliminating gather/scatter in the self-loop.

    Args:
        waves: list of lists of global clade IDs per wave
        phases: phase label per wave (1=leaf, 2=internal, 3=root)
        ccp_helpers: merged CCP dict (from collate_gene_families)
        leaf_row_index: Long[K] clade indices for leaf mapping
        leaf_col_index: Long[K] species indices for leaf mapping
        root_clade_ids: Long[F] root clade indices
        device: target device
        dtype: float dtype
        family_clade_counts: per-family clade counts [G] (for family_idx)
        family_clade_offsets: per-family clade offsets [G] (for family_idx)

    Returns:
        Dict with:
          'perm': Long[C] — original-to-new mapping
          'C': total clade count
          'leaf_row_index': remapped leaf row indices
          'leaf_species_index': species id per leaf clade, -1 otherwise
          'root_clade_ids': remapped root clade IDs
          'wave_metas': list of per-wave metadata dicts
          'family_idx': Long[C] clade→family (only if family_clade_counts provided)
    """
    return _build_wave_layout_rust(
        waves,
        phases,
        ccp_helpers,
        leaf_row_index,
        leaf_col_index,
        root_clade_ids,
        device,
        dtype,
        family_clade_counts,
        family_clade_offsets,
    )
