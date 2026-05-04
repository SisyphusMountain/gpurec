import torch

from gpurec.core.species_euler_layout import (
    euler_subtree_sums,
    reference_subtree_sums,
    species_child_arrays_from_helpers,
    species_euler_layout_report,
)


def test_postorder_species_ids_are_single_fragment_intervals():
    # Current preprocessing enumerates species in DFS postorder:
    # leaves 0, 1, parent 2, leaf 3, leaf 4, parent 5, root 6.
    S = 7
    helpers = {
        "S": S,
        "s_P_indexes": torch.tensor([2, 5, 6, 2 + S, 5 + S, 6 + S]),
        "s_C12_indexes": torch.tensor([0, 3, 2, 1, 4, 5]),
    }

    child1, child2 = species_child_arrays_from_helpers(helpers)
    report = species_euler_layout_report(helpers)

    assert child1 == [S, S, 0, S, S, 3, 2]
    assert child2 == [S, S, 1, S, S, 4, 5]
    assert report.dfs_order == tuple(range(S))
    assert report.fragment_count == (1,) * S
    assert report.excess_span == (0,) * S
    assert report.all_subtrees_contiguous
    assert report.current_interval_start[6] == 0
    assert report.current_interval_end[6] == S


def test_non_euler_current_order_reports_fragmentation():
    S = 7
    child1 = [S, S, 0, S, S, 3, 2]
    child2 = [S, S, 1, S, S, 4, 5]
    # Swap one leaf from each side.  Topology is unchanged but current ids no
    # longer put the left internal subtree in one interval.
    permutation = [0, 3, 2, 1, 4, 5, 6]
    inverse = [0] * S
    for new_id, old_id in enumerate(permutation):
        inverse[old_id] = new_id
    perm_child1 = [S] * S
    perm_child2 = [S] * S
    for old_parent in range(S):
        new_parent = inverse[old_parent]
        c1 = child1[old_parent]
        c2 = child2[old_parent]
        perm_child1[new_parent] = inverse[c1] if c1 < S else S
        perm_child2[new_parent] = inverse[c2] if c2 < S else S

    report = species_euler_layout_report(sp_child1=perm_child1, sp_child2=perm_child2)

    left_internal = inverse[2]
    assert report.fragment_count[left_internal] == 2
    assert report.excess_span[left_internal] == 1
    assert not report.all_subtrees_contiguous


def test_euler_prefix_subtree_sums_match_tree_walk_for_batched_values():
    S = 7
    child1 = torch.tensor([S, S, 0, S, S, 3, 2])
    child2 = torch.tensor([S, S, 1, S, S, 4, 5])
    report = species_euler_layout_report(sp_child1=child1, sp_child2=child2)
    values = torch.arange(3 * S, dtype=torch.float64).reshape(3, S)

    actual = euler_subtree_sums(values, report)
    expected = reference_subtree_sums(values, child1, child2)

    torch.testing.assert_close(actual, expected)
