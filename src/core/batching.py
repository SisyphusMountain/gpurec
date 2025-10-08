from typing import Any, Dict, List
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

        # (>=2) rows for this family
        if end_rows_ge2_i > 0:
            ge2_left = lefts_i[:end_rows_ge2_i]
            ge2_right = rights_i[:end_rows_ge2_i]
            ge2_left_parts.append(ge2_left)
            ge2_right_parts.append(ge2_right)
            ge2_logp_parts.append(logp_i[:end_rows_ge2_i])
            ge2_parent_ids_parts.append(seg_parent_ids_i[:num_ge2_i] + clade_offset)
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

    # Sanity checks
    assert split_leftrights_sorted_batch.numel() == 2 * total_N
    assert log_split_probs_sorted_batch.numel() == total_N
    assert ptr_ge2_batch.numel() == (total_num_ge2 + 1)
    assert seg_parent_ids_batch.numel() == (total_num_ge2 + total_num_eq1)


    leaf_row_index = torch.cat(leaf_row_parts, 0).to(device)
    leaf_col_index = torch.cat(leaf_col_parts, 0).to(device)
    out = {
        "ccp": {
            "C": total_C,
            "N_splits": total_N,
            "split_leftrights_sorted": split_leftrights_sorted_batch.to(device),
            "log_split_probs_sorted": log_split_probs_sorted_batch.to(device),
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
