"""Unit tests for clade wave scheduling."""

import pathlib
import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "tests" / "data"


def _load_helpers(case_name: str):
    """Return (ccp_helpers, root_clade_id) for the given test case."""
    from src.core.preprocess_cpp import _load_extension
    d = _DATA_DIR / case_name
    ext = _load_extension()
    raw = ext.preprocess(str(d / "sp.nwk"), [str(d / "g.nwk")])
    ccp_raw = raw["ccp"]
    device = torch.device("cpu")
    ccp_helpers = {
        "C": int(ccp_raw["C"]),
        "N_splits": int(ccp_raw["N_splits"]),
        "split_counts": ccp_raw["split_counts"].to(device=device),
        "split_parents_sorted": ccp_raw["split_parents_sorted"].to(device=device),
        "split_leftrights_sorted": ccp_raw["split_leftrights_sorted"].to(device=device),
    }
    root_clade_id = int(ccp_raw["root_clade_id"])
    return ccp_helpers, root_clade_id


class TestComputeCladeWaves:
    def test_coverage_small(self):
        """Every clade appears exactly once across all waves."""
        ccp_helpers, _ = _load_helpers("test_trees_1")
        from src.core.scheduling import compute_clade_waves
        waves, level = compute_clade_waves(ccp_helpers)
        C = ccp_helpers["C"]
        seen = []
        for w in waves:
            seen.extend(w)
        assert sorted(seen) == list(range(C)), "Each clade must appear exactly once"

    def test_topological_order_small(self):
        """For every split (p, l, r), level[p] > level[l] and level[p] > level[r]."""
        ccp_helpers, _ = _load_helpers("test_trees_1")
        from src.core.scheduling import compute_clade_waves
        waves, level = compute_clade_waves(ccp_helpers)
        N = ccp_helpers["N_splits"]
        parents = ccp_helpers["split_parents_sorted"].tolist()
        lr = ccp_helpers["split_leftrights_sorted"].tolist()
        lefts = lr[:N]
        rights = lr[N:]
        for idx in range(N):
            p, l, r = parents[idx], lefts[idx], rights[idx]
            assert level[p] > level[l], (
                f"split {idx}: level[{p}]={level[p]} not > level[{l}]={level[l]}"
            )
            assert level[p] > level[r], (
                f"split {idx}: level[{p}]={level[p]} not > level[{r}]={level[r]}"
            )

    def test_root_in_last_wave_small(self):
        """The root clade should be in the last wave."""
        ccp_helpers, root_clade_id = _load_helpers("test_trees_1")
        from src.core.scheduling import compute_clade_waves
        waves, level = compute_clade_waves(ccp_helpers)
        assert level[root_clade_id] == len(waves) - 1, (
            f"root level={level[root_clade_id]}, last wave={len(waves)-1}"
        )

    def test_balance_no_worse(self):
        """Balanced scheduling should not increase max split load."""
        ccp_helpers, _ = _load_helpers("test_trees_1")
        from src.core.scheduling import compute_clade_waves
        split_counts = ccp_helpers["split_counts"].tolist()

        waves_unbal, _ = compute_clade_waves(ccp_helpers, balance=False)
        waves_bal, _ = compute_clade_waves(ccp_helpers, balance=True)

        def max_load(waves):
            return max(sum(split_counts[c] for c in w) for w in waves if w)

        assert max_load(waves_bal) <= max_load(waves_unbal), (
            "balance=True should not increase max wave split load"
        )

    def test_wave_count_unchanged_by_balance(self):
        """Balancing must not increase the number of waves."""
        ccp_helpers, _ = _load_helpers("test_trees_1")
        from src.core.scheduling import compute_clade_waves
        waves_unbal, _ = compute_clade_waves(ccp_helpers, balance=False)
        waves_bal, _ = compute_clade_waves(ccp_helpers, balance=True)
        assert len(waves_bal) == len(waves_unbal), "balance=True must not add extra waves"


class TestWaveStats:
    def test_stats_keys(self):
        ccp_helpers, _ = _load_helpers("test_trees_1")
        from src.core.scheduling import compute_clade_waves, wave_stats
        waves, _ = compute_clade_waves(ccp_helpers)
        stats = wave_stats(waves, ccp_helpers)
        assert len(stats) == len(waves)
        for s in stats:
            for key in ("wave", "n_clades", "n_splits", "max_splits_per_clade"):
                assert key in s

    def test_total_splits_matches(self):
        """Sum of n_splits across waves equals N_splits."""
        ccp_helpers, _ = _load_helpers("test_trees_1")
        from src.core.scheduling import compute_clade_waves, wave_stats
        waves, _ = compute_clade_waves(ccp_helpers)
        stats = wave_stats(waves, ccp_helpers)
        assert sum(s["n_splits"] for s in stats) == ccp_helpers["N_splits"]
