import heapq
from typing import Any, Dict, List, Sequence, Tuple

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

    for item in batch:
        ccp = item["ccp"]
        root_i: int = int(item["root_clade_id"])

        # Family sizes
        C_i = int(ccp["C"])
        N_i = int(ccp["N_splits"])
        num_eq1_i = int(ccp["num_segs_eq1"])
        end_rows_ge2_i = int(ccp["end_rows_ge2"])

        # Pull split arrays
        leftrights_i = ccp["split_leftrights_sorted"].to(torch.long).cpu()  # [2*N_i]
        logp_i = ccp["log_split_probs_sorted"].to(dtype).cpu()              # [N_i]

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

    # Sanity checks
    assert split_leftrights_sorted_batch.numel() == 2 * total_N
    assert log_split_probs_sorted_batch.numel() == total_N
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
        },
        "leaf_row_index": leaf_row_index,
        "leaf_col_index": leaf_col_index,
        "root_clade_ids": torch.tensor(root_ids, dtype=torch.long, device=device),
        "family_meta": family_meta,
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

    for fam_waves, offset in zip(families_waves, families_clade_offsets):
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


def _cpu_long_list(value: Any) -> List[int]:
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().tolist()]
    return [int(x) for x in value]


def _ccp_split_counts(ccp: Dict[str, Any], C: int, parents: Sequence[int]) -> List[int]:
    if "split_counts" in ccp:
        counts = _cpu_long_list(ccp["split_counts"])
        if len(counts) != C:
            raise ValueError(f"split_counts has length {len(counts)} but C={C}")
        return counts
    counts = [0] * C
    for p in parents:
        counts[int(p)] += 1
    return counts


def _family_schedule_data(ccp: Dict[str, Any]) -> Dict[str, Any]:
    """Build the dependency data needed for cross-family wave scheduling."""
    C = int(ccp["C"])
    N = int(ccp["N_splits"])
    parents = _cpu_long_list(ccp["split_parents_sorted"])
    leftrights = _cpu_long_list(ccp["split_leftrights_sorted"])
    if len(parents) != N:
        raise ValueError(f"split_parents_sorted has length {len(parents)} but N_splits={N}")
    if len(leftrights) != 2 * N:
        raise ValueError(
            f"split_leftrights_sorted has length {len(leftrights)} but 2*N_splits={2 * N}"
        )
    lefts = leftrights[:N]
    rights = leftrights[N:]
    split_counts = _ccp_split_counts(ccp, C, parents)

    children: List[List[int]] = [[] for _ in range(C)]
    parents_of: List[List[int]] = [[] for _ in range(C)]
    remaining = [0] * C
    child_sets: List[set[int]] = [set() for _ in range(C)]
    for p, l, r in zip(parents, lefts, rights):
        p = int(p)
        for child in (int(l), int(r)):
            if child not in child_sets[p]:
                child_sets[p].add(child)
                children[p].append(child)
                parents_of[child].append(p)
                remaining[p] += 1

    # Bottom-up levels identify when a clade can first become ready; lambda is
    # a root-distance priority used to keep long chains moving when a ready set
    # is larger than the wave cap.
    bfs_level = [0] * C
    remaining_bfs = list(remaining)
    queue = [c for c in range(C) if remaining_bfs[c] == 0]
    head = 0
    max_level = 0
    while head < len(queue):
        c = queue[head]
        head += 1
        for p in parents_of[c]:
            if bfs_level[p] <= bfs_level[c]:
                bfs_level[p] = bfs_level[c] + 1
                max_level = max(max_level, bfs_level[p])
            remaining_bfs[p] -= 1
            if remaining_bfs[p] == 0:
                queue.append(p)

    levels: List[List[int]] = [[] for _ in range(max_level + 1)]
    for c, level in enumerate(bfs_level):
        levels[level].append(c)
    priority = [0] * C
    for level in range(max_level, -1, -1):
        for c in levels[level]:
            for child in children[c]:
                priority[child] = max(priority[child], priority[c] + 1)

    return {
        "C": C,
        "split_counts": split_counts,
        "children": children,
        "parents_of": parents_of,
        "remaining": remaining,
        "priority": priority,
        "root_id": int(ccp.get("root_clade_id", -1)),
    }


def schedule_global_phased_waves(
    items: Sequence[Dict[str, Any]],
    family_clade_offsets: Sequence[int],
    *,
    max_wave_size: int | None,
    max_root_wave_size: int | None = None,
) -> Tuple[List[List[int]], List[int]]:
    """Schedule one resident batch with globally packed ready waves.

    The retained kernels still need leaf clades handled before non-leaf clades,
    but after the leaf phase this schedules all ready clades from all families
    into waves capped by ``max_wave_size``.  This replaces the older lockstep
    ``family wave k -> resident wave k`` collation, which left many waves far
    below the GPU-friendly cap.
    """
    if len(items) != len(family_clade_offsets):
        raise ValueError("items and family_clade_offsets must have matching lengths")
    total_clades = sum(int(item["ccp"]["C"]) for item in items)
    if total_clades == 0:
        return [], []
    wave_cap = total_clades if max_wave_size is None else int(max_wave_size)
    if wave_cap <= 0:
        raise ValueError("max_wave_size must be positive")
    root_cap = (
        None
        if max_root_wave_size is None
        else int(max_root_wave_size)
    )
    if root_cap is not None and root_cap <= 0:
        raise ValueError("max_root_wave_size must be positive")

    families = [_family_schedule_data(item["ccp"]) for item in items]
    scheduled = [[False] * fam["C"] for fam in families]
    queued = [[False] * fam["C"] for fam in families]
    remaining = [list(fam["remaining"]) for fam in families]

    waves: List[List[int]] = []
    phases: List[int] = []

    all_leaves: List[Tuple[int, int]] = []
    for fi, fam in enumerate(families):
        for c, count in enumerate(fam["split_counts"]):
            if int(count) == 0:
                all_leaves.append((fi, c))

    all_leaves.sort(key=lambda item: (item[0], item[1]))
    for start in range(0, len(all_leaves), wave_cap):
        chunk = all_leaves[start:start + wave_cap]
        wave = [
            int(family_clade_offsets[fi]) + c
            for fi, c in chunk
        ]
        waves.append(wave)
        phases.append(1)
        for fi, c in chunk:
            scheduled[fi][c] = True
            for parent in families[fi]["parents_of"][c]:
                remaining[fi][parent] -= 1

    ready: List[Tuple[int, int, int]] = []

    def push_ready(fi: int, c: int) -> None:
        if scheduled[fi][c] or queued[fi][c] or remaining[fi][c] != 0:
            return
        queued[fi][c] = True
        priority = int(families[fi]["priority"][c])
        # heapq is a min-heap.  Use negative priority so long root-distance
        # chains are drained first, then stable family/clade tie breakers.
        heapq.heappush(ready, (-priority, fi, c))

    for fi, fam in enumerate(families):
        for c in range(fam["C"]):
            push_ready(fi, c)

    while ready:
        batch: List[Tuple[int, int]] = []
        while ready and len(batch) < wave_cap:
            _neg_priority, fi, c = heapq.heappop(ready)
            if scheduled[fi][c]:
                continue
            queued[fi][c] = False
            if remaining[fi][c] != 0:
                continue
            batch.append((fi, c))
        if not batch:
            continue

        all_roots = all(c == families[fi]["root_id"] for fi, c in batch)
        phase = 3 if all_roots else 2
        if phase == 3 and root_cap is not None and len(batch) > root_cap:
            chunks = [
                batch[start:start + root_cap]
                for start in range(0, len(batch), root_cap)
            ]
        else:
            chunks = [batch]

        for chunk in chunks:
            waves.append([
                int(family_clade_offsets[fi]) + c
                for fi, c in chunk
            ])
            phases.append(phase)
            for fi, c in chunk:
                scheduled[fi][c] = True
            for fi, c in chunk:
                for parent in families[fi]["parents_of"][c]:
                    remaining[fi][parent] -= 1
                    push_ready(fi, parent)

    scheduled_count = sum(sum(1 for done in family if done) for family in scheduled)
    if scheduled_count != total_clades:
        raise RuntimeError(
            "global wave scheduler did not cover all clades: "
            f"scheduled={scheduled_count}, total={total_clades}"
        )
    return waves, phases


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
            meta['n_eq1'] = n_eq1

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

        wave_metas.append(meta)

    result = {
        'perm': perm,
        'C': C,
        'leaf_row_index': leaf_row_new,
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
