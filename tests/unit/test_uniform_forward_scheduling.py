from profiling.bench_uniform_forward_chunking import _split_waves_by_split_budget


def test_split_budget_preserves_order_and_caps_split_rows():
    waves = [[0, 1, 2, 3], [4, 5]]
    phases = [2, 3]
    split_counts = [5, 20, 6, 1, 50, 2]

    out_waves, out_phases, n_split = _split_waves_by_split_budget(
        waves,
        phases,
        split_counts,
        max_split_rows=25,
        max_split_fanout=None,
    )

    assert out_waves == [[0, 1], [2, 3], [4], [5]]
    assert out_phases == [2, 2, 3, 3]
    assert n_split == 2
    assert [c for wave in out_waves for c in wave] == [0, 1, 2, 3, 4, 5]
    assert all(sum(split_counts[c] for c in wave) <= 25 for wave in out_waves if len(wave) > 1)


def test_split_budget_caps_average_fanout_when_possible():
    waves = [[0, 1, 2, 3]]
    phases = [3]
    split_counts = [4, 8, 2, 2]

    out_waves, out_phases, n_split = _split_waves_by_split_budget(
        waves,
        phases,
        split_counts,
        max_split_rows=None,
        max_split_fanout=5.0,
    )

    assert out_waves == [[0], [1, 2, 3]]
    assert out_phases == [3, 3]
    assert n_split == 1
