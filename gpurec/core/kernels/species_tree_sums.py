"""Species-tree sums that a kernel holding one whole row in registers builds by ADDITION.

Two sums keep coming up in this model, both over "every species EXCEPT a few near one lane":

* the **valid receiver mass** of a donor ``s`` -- a donor may transfer into every species that is
  neither ``s`` itself nor one of ``s``'s ancestors, so the sum runs over everything off ``s``'s
  ancestor chain (:func:`valid_receiver_sum`);
* the **off-subtree donor adjoint** of a receiver ``s`` -- ``s`` takes transfer mass from every
  donor outside ``s``'s own subtree, so the sum runs over everything off that subtree
  (:func:`off_subtree_sum`).

Both used to be written as "the whole row's total minus the few excluded entries". That is the
defect this module exists to prevent. In a high-loss or high-transfer regime one lane holds
essentially all of a row's mass, and the species hanging under that lane sit 50 to 300 binary
orders below it; their true share of the row is far below the unit roundoff of the total (2^-24 in
float32, 2^-53 in float64), so the subtraction returns rounding noise instead of a small number.
Those lanes then multiply into products upstream and the noise grows wave by wave: on a
1007-species Coleman family at log2 rates D=-19.9, L=1, T=-19.9 the likelihood came out 8.7 bits
wrong and the gradient 1e8 times too large.

Both helpers are therefore built out of ADDITIONS of non-negative-or-signed terms and never a
difference of two nearly equal numbers:

1. bottom-up, each node's subtree sum from its two children's (a node of height ``h`` reads
   children of height below ``h``, already final);
2. top-down, what lies outside a node, from its parent's outside sum plus the sibling's whole
   subtree (plus, for the subtree flavour, the parent's own term).

Both take the whole species row as one register vector (``BLOCK_S >= S``) and walk the tree with
``tl.gather``, so they suit the per-row kernels -- the wave second-order contraction and the
E-step -- rather than the block-tiled ones, which walk the compact level tables in global memory
instead (see ``_reconciliation_self_loop_transpose_term`` in wave_backward_kernels.py).
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def species_neighbourhood(
    species_child1_ptr, species_child2_ptr, species_parent_ptr, species_height_ptr,
    s_offs, mask, S: tl.constexpr,
):
    """Children, parent, sibling and heights of every lane, as whole-row gathers.

    Returns ``(species_height, c1_valid, c1_safe, c2_valid, c2_safe, has_parent, parent_safe,
    parent_height, has_sibling, sibling_safe)``; ``*_safe`` indices are 0 where invalid so they
    can be gathered unconditionally and masked afterwards. A missing child is written either as
    ``S`` (the preprocessor's sentinel) or as ``-1`` (hand-built test topologies), so both ends
    are checked.
    """
    c1 = tl.load(species_child1_ptr + s_offs, mask=mask, other=S)
    c2 = tl.load(species_child2_ptr + s_offs, mask=mask, other=S)
    c1_valid = mask & (c1 >= 0) & (c1 < S)
    c2_valid = mask & (c2 >= 0) & (c2 < S)
    c1_safe = tl.where(c1_valid, c1, 0)
    c2_safe = tl.where(c2_valid, c2, 0)
    species_height = tl.load(species_height_ptr + s_offs, mask=mask, other=0)
    parent_species = tl.load(species_parent_ptr + s_offs, mask=mask, other=-1)
    has_parent = mask & (parent_species >= 0) & (parent_species < S)
    parent_safe = tl.where(has_parent, parent_species, 0)
    parent_height = tl.where(has_parent, tl.gather(species_height, parent_safe, axis=0), 0)
    parent_child1 = tl.gather(c1, parent_safe, axis=0)
    parent_child2 = tl.gather(c2, parent_safe, axis=0)
    sibling = tl.where(parent_child1 == s_offs, parent_child2, parent_child1)
    has_sibling = has_parent & (sibling >= 0) & (sibling < S)
    sibling_safe = tl.where(has_sibling, sibling, 0)
    return (
        species_height, c1_valid, c1_safe, c2_valid, c2_safe,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe,
    )


@triton.jit
def valid_receiver_sum(
    value, mask, zero, species_height,
    c1_valid, c1_safe, c2_valid, c2_safe,
    has_parent, parent_safe, parent_height, has_sibling, sibling_safe,
    N_LEVELS: tl.constexpr,
):
    """Per lane s, the sum of ``value`` over every species that is neither s nor an ancestor of s.

    Built by ADDITION only: subtree sums bottom-up (a lane of height ``level`` reads its children,
    already final), then off-chain sums top-down (what hangs off a lane's ancestor chain is what
    hangs off its parent's chain plus the sibling's whole subtree; a lane is settled in the pass
    of its PARENT's height), and finally ``off_chain + subtree(child1) + subtree(child2)``.

    Never ``row total - ancestor chain``: for a species under the lane holding the row's mass the
    true remainder is below the unit roundoff of the total (2^-24 float32, 2^-53 float64) and that
    difference is noise -- see this module's docstring. ``value`` may be signed (a tangent
    numerator); the walk is the same.
    """
    subtree = value
    for level in range(1, N_LEVELS + 1):
        at_level = mask & (species_height == level)
        subtree = tl.where(
            at_level,
            value
            + tl.where(c1_valid, tl.gather(subtree, c1_safe, axis=0), zero)
            + tl.where(c2_valid, tl.gather(subtree, c2_safe, axis=0), zero),
            subtree,
        )
    off_chain = zero
    for level_index in range(0, N_LEVELS):
        level = N_LEVELS - level_index
        at_level = has_parent & (parent_height == level)
        off_chain = tl.where(
            at_level,
            tl.gather(off_chain, parent_safe, axis=0)
            + tl.where(has_sibling, tl.gather(subtree, sibling_safe, axis=0), zero),
            off_chain,
        )
    return (
        off_chain
        + tl.where(c1_valid, tl.gather(subtree, c1_safe, axis=0), zero)
        + tl.where(c2_valid, tl.gather(subtree, c2_safe, axis=0), zero)
    )


@triton.jit
def off_subtree_sum(
    value, mask, zero, species_height,
    c1_valid, c1_safe, c2_valid, c2_safe,
    has_parent, parent_safe, parent_height, has_sibling, sibling_safe,
    N_LEVELS: tl.constexpr,
):
    """Per lane s, the sum of ``value`` over every species OUTSIDE s's own subtree.

    The companion of :func:`valid_receiver_sum` and its transpose: that one drops the lane's
    ancestor chain, this one drops the lane's descendants. Receiver s takes transfer mass from
    every donor that is neither s nor a descendant of s, so a transfer VJP needs exactly this sum
    of the donor adjoints.

    Same two additive passes: subtree sums bottom-up, then top-down
    ``off_subtree(child) = off_subtree(parent) + parent's own term + sibling's whole subtree``,
    zero at the root (whose subtree is the whole tree, so nothing lies outside it). Each lane is
    settled in the pass of its PARENT's height, and its parent was settled in an earlier pass
    because a parent's parent is taller still.

    Never ``row total - subtree sum``: each donor adjoint divides by that donor's own valid
    receiver mass, so for a species hanging under the lane holding the row's mass it is
    astronomically large; the total is dominated by those terms and, for the dominant lane -- whose
    subtree holds all of them -- the difference cancels to rounding noise of that same size.
    """
    subtree = value
    for level in range(1, N_LEVELS + 1):
        at_level = mask & (species_height == level)
        subtree = tl.where(
            at_level,
            value
            + tl.where(c1_valid, tl.gather(subtree, c1_safe, axis=0), zero)
            + tl.where(c2_valid, tl.gather(subtree, c2_safe, axis=0), zero),
            subtree,
        )
    off_subtree = zero
    for level_index in range(0, N_LEVELS):
        level = N_LEVELS - level_index
        at_level = has_parent & (parent_height == level)
        off_subtree = tl.where(
            at_level,
            tl.gather(off_subtree, parent_safe, axis=0)
            + tl.gather(value, parent_safe, axis=0)
            + tl.where(has_sibling, tl.gather(subtree, sibling_safe, axis=0), zero),
            off_subtree,
        )
    return off_subtree
