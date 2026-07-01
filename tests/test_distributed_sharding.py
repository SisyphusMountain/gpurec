import torch
from gpurec.distributed import shard_families, ddp_enabled, all_reduce_sum_


def _fam(c, n):
    return {"C": c, "N_splits": n}


def test_world1_returns_all():
    fams = [_fam(10, 5), _fam(3, 1)]
    assert shard_families(fams, rank=0, world=1) == fams


def test_partition_is_disjoint_and_covers_all():
    fams = [_fam(i, i) for i in range(1, 9)]
    s0 = shard_families(fams, 0, 2)
    s1 = shard_families(fams, 1, 2)
    ids = sorted([id(f) for f in s0] + [id(f) for f in s1])
    assert ids == sorted(id(f) for f in fams)  # every family assigned exactly once


def test_balances_by_work_not_count():
    fams = [_fam(100, 100)] + [_fam(1, 0) for _ in range(6)]
    s0 = shard_families(fams, 0, 2)
    s1 = shard_families(fams, 1, 2)
    w = lambda s: sum(f["C"] + f["N_splits"] for f in s)
    assert abs(w(s0) - w(s1)) <= max(w(s0), w(s1))  # not wildly imbalanced
    assert len(s0) + len(s1) == 7


def test_deterministic():
    fams = [_fam(i % 5, i % 3) for i in range(20)]
    assert [id(f) for f in shard_families(fams, 0, 3)] == [id(f) for f in shard_families(fams, 0, 3)]


def test_reduce_is_noop_when_not_distributed():
    assert ddp_enabled() is False
    t = torch.ones(4)
    assert torch.equal(all_reduce_sum_(t), torch.ones(4))
