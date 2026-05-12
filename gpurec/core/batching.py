from typing import Any, Dict, List, Tuple

import torch


def collate_gene_families(
    batch: List[Dict[str, Any]],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """
    Collate multiple gene-family samples (each from `preprocess_gene_with_species`)
    into a single batched CCP for likelihood_2.py.

    Each item in `batch` must be a dict with keys:
      - 'ccp': dict containing at least
          'C' (int), 'N_splits' (int),
          'split_leftrights_sorted' (Long[2*N_i]),
          'log_split_probs_sorted' (Float[N_i]),
          'seg_parent_ids' (Long[num_ge2_i + num_eq1_i + num_eq0_i]),
          'ptr_ge2' (Long[num_ge2_i + 1]),
          'num_segs_ge2' (int), 'num_segs_eq1' (int), 'num_segs_eq0' (int),
          'end_rows_ge2' (int).
      - 'leaf_row_index': Long[K_i]
      - 'leaf_col_index': Long[K_i]
      - 'root_clade_id': int

    Returns a dict with:
      - 'ccp': merged CCP helpers compatible with Pi_step()
      - 'root_clade_ids': Long[F] (roots offset into concatenated clade axis)
      - 'family_meta': bookkeeping (clade offsets, per-family sizes)
    """
    # ---- running offsets and accumulators ----
    clade_offset = 0
    row_offset_ge2 = 0  # counts DTS rows only in the >=2 block

    root_ids: List[int] = []
    leaf_row_parts, leaf_col_parts = [], []
    # (>=2) block
    # we will accumulate left and right halves separately to preserve [all_lefts ; all_rights] order
    ge2_left_parts: List[torch.Tensor] = []
    ge2_right_parts: List[torch.Tensor] = []
    ge2_logp_parts: List[torch.Tensor] = []
    ge2_parent_ids_parts: List[torch.Tensor] = []
    ge2_ptr_pieces: List[torch.Tensor] = []  # we will build [0] + cat of (ptr[1:]+offset)

    # (=1) block
    eq1_left_parts: List[torch.Tensor] = []
    eq1_right_parts: List[torch.Tensor] = []
    eq1_logp_parts: List[torch.Tensor] = []
    eq1_parent_ids_parts: List[torch.Tensor] = []

    # split_parents_sorted (one entry per split, needed for wave scheduling)
    ge2_split_parents_parts: List[torch.Tensor] = []
    eq1_split_parents_parts: List[torch.Tensor] = []

    # Totals & per-family meta
    total_C = 0
    total_N = 0
    total_num_ge2 = 0
    total_num_eq1 = 0
    total_num_eq0 = 0
    total_end_rows_ge2 = 0

    family_meta: List[Dict[str, int]] = []

    for item in batch:
        ccp = item["ccp"]
        root_i: int = int(item["root_clade_id"])

        # Family sizes
        C_i = int(ccp["C"])
        N_i = int(ccp["N_splits"])
        num_ge2_i = int(ccp["num_segs_ge2"])
        num_eq1_i = int(ccp["num_segs_eq1"])
        num_eq0_i = int(ccp.get("num_segs_eq0", 0))
        end_rows_ge2_i = int(ccp["end_rows_ge2"])

        # Pull split arrays
        leftrights_i = ccp["split_leftrights_sorted"].to(torch.long).cpu()  # [2*N_i]
        logp_i = ccp["log_split_probs_sorted"].to(dtype).cpu()              # [N_i]
        seg_parent_ids_i = ccp["seg_parent_ids"].to(torch.long).cpu()       # [num_ge2_i + num_eq1_i + num_eq0_i]
        ptr_ge2_i = ccp["ptr_ge2"].to(torch.long).cpu()                     # [num_ge2_i + 1]

        # Split left/right halves so we can cut the ge2 vs eq1 ranges cleanly
        assert leftrights_i.numel() == 2 * N_i
        lefts_i = leftrights_i[:N_i]
        rights_i = leftrights_i[N_i:]

        # Offsets on clade indices for this family
        lefts_i = lefts_i + clade_offset
        rights_i = rights_i + clade_offset

        if "split_parents_sorted" not in ccp:
            raise RuntimeError("preprocessed CCP helpers must include split_parents_sorted")
        sp_sorted_i = ccp["split_parents_sorted"].to(torch.long).cpu()

        # (>=2) rows for this family
        if end_rows_ge2_i > 0:
            ge2_left = lefts_i[:end_rows_ge2_i]
            ge2_right = rights_i[:end_rows_ge2_i]
            ge2_left_parts.append(ge2_left)
            ge2_right_parts.append(ge2_right)
            ge2_logp_parts.append(logp_i[:end_rows_ge2_i])
            ge2_parent_ids_parts.append(seg_parent_ids_i[:num_ge2_i] + clade_offset)
            ge2_split_parents_parts.append(sp_sorted_i[:end_rows_ge2_i] + clade_offset)
            # stitch the ptrs: skip the leading 0 and add the current global row offset
            if num_ge2_i > 0:
                ge2_ptr_pieces.append(ptr_ge2_i[1:] + row_offset_ge2)
            row_offset_ge2 += end_rows_ge2_i

        # (=1) rows for this family (exactly one split per clade)
        if num_eq1_i > 0:
            start = end_rows_ge2_i
            stop = end_rows_ge2_i + num_eq1_i
            eq1_left = lefts_i[start:stop]
            eq1_right = rights_i[start:stop]
            eq1_left_parts.append(eq1_left)
            eq1_right_parts.append(eq1_right)
            eq1_logp_parts.append(logp_i[start:stop])
            eq1_parent_ids_parts.append(seg_parent_ids_i[num_ge2_i:num_ge2_i+num_eq1_i] + clade_offset)
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
        total_num_ge2 += num_ge2_i
        total_num_eq1 += num_eq1_i
        total_num_eq0 += num_eq0_i
        total_end_rows_ge2 += end_rows_ge2_i

        family_meta.append({
            "C": C_i,
            "N_splits": N_i,
            "num_segs_ge2": num_ge2_i,
            "num_segs_eq1": num_eq1_i,
            "num_segs_eq0": num_eq0_i,
            "end_rows_ge2": end_rows_ge2_i,
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

    # seg_parent_ids: [all GE2 parents] ⧺ [all EQ1 parents] (EQ0 not used in Pi_step)
    if len(ge2_parent_ids_parts) > 0:
        seg_parent_ids_ge2 = torch.cat(ge2_parent_ids_parts, dim=0)
    else:
        seg_parent_ids_ge2 = torch.empty((0,), dtype=torch.long)
    if len(eq1_parent_ids_parts) > 0:
        seg_parent_ids_eq1 = torch.cat(eq1_parent_ids_parts, dim=0)
    else:
        seg_parent_ids_eq1 = torch.empty((0,), dtype=torch.long)
    seg_parent_ids_batch = torch.cat([seg_parent_ids_ge2, seg_parent_ids_eq1], dim=0)

    # ptr_ge2: stitch families’ >=2 pointers into one global pointer for the
    #          first total_end_rows_ge2 rows of DTS_term
    if total_num_ge2 > 0:
        ptr_start = torch.tensor([0], dtype=torch.long)
        if len(ge2_ptr_pieces) > 0:
            ptr_rest = torch.cat(ge2_ptr_pieces, dim=0)
            ptr_ge2_batch = torch.cat([ptr_start, ptr_rest], dim=0)
        else:
            # No family has >=2 segments, but total_num_ge2>0 implies logic error
            ptr_ge2_batch = ptr_start
    else:
        # Degenerate case: no clade has >=2 splits in the entire batch
        ptr_ge2_batch = torch.tensor([0], dtype=torch.long)

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

    # Sanity checks
    assert split_leftrights_sorted_batch.numel() == 2 * total_N
    assert log_split_probs_sorted_batch.numel() == total_N
    assert ptr_ge2_batch.numel() == (total_num_ge2 + 1)
    assert seg_parent_ids_batch.numel() == (total_num_ge2 + total_num_eq1)
    assert split_parents_sorted_batch.numel() == total_N

    leaf_row_index = torch.cat(leaf_row_parts, 0).to(device)
    leaf_col_index = torch.cat(leaf_col_parts, 0).to(device)
    out = {
        "ccp": {
            "C": total_C,
            "N_splits": total_N,
            "split_leftrights_sorted": split_leftrights_sorted_batch.to(device),
            "log_split_probs_sorted": log_split_probs_sorted_batch.to(device),
            "split_parents_sorted": split_parents_sorted_batch.to(device),
            "seg_parent_ids": seg_parent_ids_batch.to(device),
            "ptr_ge2": ptr_ge2_batch.to(device),
            "num_segs_ge2": total_num_ge2,
            "num_segs_eq1": total_num_eq1,
            "num_segs_eq0": total_num_eq0,
            "end_rows_ge2": total_end_rows_ge2,
            "stop_reduce_ptr_idx": total_num_ge2,
        },
        "leaf_row_index": leaf_row_index,
        "leaf_col_index": leaf_col_index,
        "root_clade_ids": torch.tensor(root_ids, dtype=torch.long, device=device),
        "family_meta": family_meta,  # optional, but useful bookkeeping
    }
    return out


def collate_wave(
    families_waves: List[List[List[int]]],
    families_clade_offsets: List[int],
) -> List[List[int]]:
    """Merge per-family wave assignments into cross-family waves.

    For each wave index k, collects all families' wave-k clade IDs (with
    global clade offsets applied) into a single list. This enables batching
    all families' wave-k clades into one matmul.

    Args:
        families_waves: list of per-family wave lists (from compute_clade_waves)
        families_clade_offsets: global clade offset for each family

    Returns:
        cross_waves: list of lists of globally-offset clade IDs per wave
    """
    max_waves = max(len(w) for w in families_waves) if families_waves else 0
    cross_waves: List[List[int]] = [[] for _ in range(max_waves)]

    for fam_idx, (fam_waves, offset) in enumerate(zip(families_waves, families_clade_offsets)):
        for k, wave_clades in enumerate(fam_waves):
            cross_waves[k].extend(c + offset for c in wave_clades)

    return cross_waves


def split_phase_waves(
    waves: List[List[int]],
    phases: List[int],
    *,
    phase: int | None,
    max_wave_size: int | None,
) -> Tuple[List[List[int]], List[int]]:
    """Split selected waves, preserving the original order.

    If ``phase`` is ``None``, split waves from all phases.
    """
    if max_wave_size is None:
        return waves, phases
    if max_wave_size <= 0:
        raise ValueError("max_wave_size must be positive")

    out_waves: List[List[int]] = []
    out_phases: List[int] = []
    for wave, ph in zip(waves, phases):
        if (phase is None or ph == phase) and len(wave) > max_wave_size:
            for start in range(0, len(wave), max_wave_size):
                out_waves.append(wave[start:start + max_wave_size])
                out_phases.append(ph)
        else:
            out_waves.append(wave)
            out_phases.append(ph)
    return out_waves, out_phases


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
          'ccp_helpers': remapped CCP dict (all clade IDs in new space)
          'leaf_row_index': remapped leaf row indices
          'leaf_col_index': unchanged leaf col indices
          'root_clade_ids': remapped root clade IDs
          'wave_metas': list of per-wave metadata dicts
          'family_idx': Long[C] clade→family (only if family_clade_counts provided)
    """
    C = int(ccp_helpers['C'])
    N_splits = int(ccp_helpers['N_splits'])
    if C > torch.iinfo(torch.int32).max:
        raise ValueError(f"wave split metadata requires int32 clade ids, got C={C}")

    # --- 2a. Build permutation ---
    all_clades: List[int] = []
    wave_starts_list: List[int] = [0]
    for wave_ids in waves:
        all_clades.extend(wave_ids)
        wave_starts_list.append(len(all_clades))

    assert len(all_clades) == C, (
        f"Wave layout covers {len(all_clades)} clades but C={C}"
    )

    inv_perm = torch.tensor(all_clades, dtype=torch.long, device=device)
    perm = torch.empty(C, dtype=torch.long, device=device)
    perm[inv_perm] = torch.arange(C, dtype=torch.long, device=device)

    # --- 2b. Remap all clade-index tensors (fully vectorized) ---
    split_lr = ccp_helpers['split_leftrights_sorted'].to(device=device, dtype=torch.long)
    lefts_orig = split_lr[:N_splits]
    rights_orig = split_lr[N_splits:]
    lefts_new = perm[lefts_orig]
    rights_new = perm[rights_orig]

    if 'split_parents_sorted' not in ccp_helpers:
        raise RuntimeError("preprocessed CCP helpers must include split_parents_sorted")
    split_parents = ccp_helpers['split_parents_sorted']
    sp_new = perm[split_parents.to(device=device, dtype=torch.long)]

    leaf_row_new = perm[leaf_row_index.to(device=device, dtype=torch.long)]
    root_ids_new = perm[root_clade_ids.to(device=device, dtype=torch.long)]
    root_ids_new_cpu = [int(x) for x in root_ids_new.detach().cpu().tolist()]
    leaf_col_new = leaf_col_index.to(device=device, dtype=torch.long)
    leaf_species_index = torch.full((C,), -1, dtype=torch.long, device=device)
    if leaf_row_new.numel() > 0:
        leaf_species_index[leaf_row_new] = leaf_col_new

    log_split_probs = ccp_helpers['log_split_probs_sorted']
    if torch.is_tensor(log_split_probs):
        log_split_probs = log_split_probs.to(device=device, dtype=dtype)

    # --- 2c. Vectorized per-wave metadata ---
    # For each split, find which wave its parent belongs to via searchsorted
    # sp_new[i] is the new-space parent clade of split i, which is in [0, C)
    # wave_starts is sorted → searchsorted gives the wave index
    wave_starts_cpu = torch.tensor(wave_starts_list, dtype=torch.long)
    sp_new_cpu = sp_new.cpu()

    # searchsorted: find wave index for each split's parent
    # wave_starts_list = [0, w0_end, w1_end, ...]. searchsorted(right) - 1 gives wave idx.
    split_wave_idx = torch.searchsorted(wave_starts_cpu[1:], sp_new_cpu, right=True)
    # split_wave_idx[i] = wave index of split i's parent

    # Sort splits by wave index for efficient slicing
    sort_order = split_wave_idx.argsort()
    split_wave_sorted = split_wave_idx[sort_order]

    # Find boundaries: where does each wave's splits start/end in the sorted order
    n_waves = len(waves)
    # Use searchsorted on the sorted wave indices
    wave_split_starts = torch.searchsorted(split_wave_sorted, torch.arange(n_waves, dtype=torch.long))
    wave_split_ends = torch.searchsorted(split_wave_sorted, torch.arange(n_waves, dtype=torch.long), right=True)

    # Move sort_order to device for indexing
    sort_order_dev = sort_order.to(device)

    wave_metas: List[Dict[str, Any]] = []
    for wi in range(n_waves):
        ws = wave_starts_list[wi]
        we = wave_starts_list[wi + 1]
        W = we - ws

        ss = int(wave_split_starts[wi].item())
        se = int(wave_split_ends[wi].item())
        n_ws = se - ss

        meta: Dict[str, Any] = {
            'start': ws,
            'end': we,
            'W': W,
            'phase': phases[wi],
            'has_splits': n_ws > 0,
        }

        if n_ws > 0:
            wst = sort_order_dev[ss:se]  # split indices for this wave
            reduce_idx = sp_new[wst] - ws  # [n_ws] wave-local clade index

            # Sort splits: single-split clades first, then multi-split clades.
            clade_split_counts = torch.zeros(W, dtype=torch.long, device=device)
            clade_split_counts.scatter_add_(0, reduce_idx,
                                            torch.ones(n_ws, dtype=torch.long, device=device))
            # Per-split: count for the parent clade of that split
            per_split_count = clade_split_counts[reduce_idx]  # [n_ws]
            # Composite sort key: eq1 first (is_ge2=0), ge2 after (is_ge2=1),
            # within ge2 sorted by parent clade (ascending) for CSR contiguity.
            sort_key = (per_split_count > 1).long() * (W + 1) + reduce_idx
            inner_order = sort_key.argsort(stable=True)
            wst = wst[inner_order]
            reduce_idx = reduce_idx[inner_order]

            n_eq1 = int((per_split_count == 1).sum().item())
            n_ge2_clades = int((clade_split_counts >= 2).sum().item())

            index_dtype = torch.int32
            sl_index = lefts_new[wst].to(index_dtype).contiguous()
            sr_index = rights_new[wst].to(index_dtype).contiguous()
            reduce_idx_index = reduce_idx.to(index_dtype).contiguous()

            meta['sl'] = sl_index
            meta['sr'] = sr_index
            meta['log_split_probs'] = log_split_probs[wst].unsqueeze(1).contiguous()
            meta['reduce_idx'] = reduce_idx_index
            meta['n_ws'] = n_ws
            meta['n_eq1'] = n_eq1
            meta['n_ge2_clades'] = n_ge2_clades

            if n_eq1 > 0:
                meta['eq1_reduce_idx'] = reduce_idx_index[:n_eq1]

            if n_ge2_clades > 0:
                # Build CSR pointers for the ge2 portion (splits n_eq1:).
                # Splits are sorted by parent clade, so same-parent splits are contiguous.
                ge2_reduce = reduce_idx[n_eq1:]  # [n_ge2_splits]
                # Unique parent clades in order of first appearance (= ascending,
                # since we sorted by clade index)
                ge2_parent_ids, ge2_counts = ge2_reduce.unique_consecutive(return_counts=True)
                ge2_ptr = torch.zeros(len(ge2_parent_ids) + 1, dtype=torch.long, device=device)
                torch.cumsum(ge2_counts, dim=0, out=ge2_ptr[1:])

                meta['ge2_ptr'] = ge2_ptr
                meta['ge2_parent_ids'] = ge2_parent_ids.to(index_dtype).contiguous()  # wave-local clade indices
                meta['ge2_max_fanout'] = int(ge2_counts.max().item())
                meta['ge2_mean_fanout'] = float(ge2_counts.float().mean().item())

        wave_metas.append(meta)

    result = {
        'perm': perm,
        'ccp_helpers': {
            'C': C,
            'N_splits': N_splits,
        },
        'leaf_row_index': leaf_row_new,
        'leaf_col_index': leaf_col_new,
        'leaf_species_index': leaf_species_index,
        'root_clade_ids': root_ids_new,
        'root_clade_ids_cpu': root_ids_new_cpu,
        'wave_metas': wave_metas,
    }

    # Build clade→family mapping in wave-ordered space
    if family_clade_counts is not None and family_clade_offsets is not None:
        family_idx_orig = torch.empty(C, dtype=torch.long, device=device)
        for g, (offset, c_g) in enumerate(zip(family_clade_offsets, family_clade_counts)):
            family_idx_orig[offset:offset + c_g] = g
        # Permute to wave-ordered space: result[new_idx] = family of original clade inv_perm[new_idx]
        result['family_idx'] = family_idx_orig[inv_perm]

    return result
