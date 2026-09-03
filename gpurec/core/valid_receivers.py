"""Where a donor's valid receiver mass comes from, for both the forward and the backward.

A donor species ``s`` may transfer to every species that is neither ``s`` itself nor one of its
ancestors. The obvious way to get that mass is ``total row mass - mass on s's own lineage``. That
subtraction is what this module exists to avoid: once the transfer rate is high enough that the
row's mass sits on the donor's own lineage, the two numbers agree to more than float32's 24 bits
and their difference is noise. :func:`valid_receiver_index_tables` prepares the species orders that
let both the forward self-loop and the backward self-loop VJP build the same mass as two running
sums of non-negative terms instead, which cannot cancel.

One definition, two users: ``gpurec/core/kernels/pi_forward.py`` (through
``gpurec/core/inference/forward.py``) and ``gpurec/core/kernels/wave_backward.py``.
"""

import torch


def valid_receiver_index_tables(species_start, species_end, species_count):
    """Index tables that let a kernel build a donor's valid receiver mass without subtracting.

    With the depth-first interval numbering (``sp_subtree_start`` is a permutation of ``0..S-1``;
    each subtree owns ``[start, end)``), ``a`` is an ancestor-or-self of ``s`` exactly when
    ``start[a] <= start[s] < end[a]``. So the ALLOWED recipients are exactly the union of two
    disjoint groups -- those whose subtree has not opened yet (``start[a] > start[s]``) and those
    whose subtree already closed (``end[a] <= start[s]``) -- and each group's mass is a running sum
    of non-negative terms.

    Returns the four tensors the kernel needs: the species order each running sum scans (shifted by
    one position, so an inclusive scan already yields the exclusive prefix), and, per species, the
    position in each scan where that donor's prefix ends.
    """
    count = int(species_count)
    device = species_start.device
    start = species_start.to(dtype=torch.long)
    end = species_end.to(dtype=torch.long)
    positions = torch.arange(count, device=device, dtype=torch.long)
    subtree_opening_at = torch.empty(count, dtype=torch.long, device=device)
    subtree_opening_at[start] = positions            # the species whose subtree opens at each position
    reverse_opening_order = subtree_opening_at.flip(0)
    closing_order = torch.argsort(end, stable=True)
    closed_by = torch.searchsorted(
        end.index_select(0, closing_order).contiguous(), positions, right=True
    )
    # ``count`` is the shift sentinel: the kernel reads it as "contributes nothing".
    sentinel = torch.full((1,), count, dtype=torch.long, device=device)
    not_open_source = torch.cat((sentinel, reverse_opening_order[:-1]))
    closed_source = torch.cat((sentinel, closing_order[:-1]))
    not_open_index = (count - 1) - start
    closed_index = closed_by.index_select(0, start)
    return (
        not_open_source.to(torch.int32).contiguous(),
        closed_source.to(torch.int32).contiguous(),
        not_open_index.to(torch.int32).contiguous(),
        closed_index.to(torch.int32).contiguous(),
    )
