"""Unit tests for clade wave scheduling."""

import pathlib
import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "tests" / "data"


def _load_helpers(case_name: str):
    """Return (ccp_helpers, root_clade_id) for the given test case."""
    from gpurec.core.preprocess_cpp import _load_extension
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
        from gpurec.core.scheduling import compute_clade_waves
        waves, level = compute_clade_waves(ccp_helpers)
        C = ccp_helpers["C"]
        seen = []
        for w in waves:
            seen.extend(w)
        assert sorted(seen) == list(range(C)), "Each clade must appear exactly once"

    def test_topological_order_small(self):
        """For every split (p, l, r), wave(p) > wave(l) and wave(p) > wave(r)."""
        ccp_helpers, _ = _load_helpers("test_trees_1")
        from gpurec.core.scheduling import compute_clade_waves
        waves, phases = compute_clade_waves(ccp_helpers)
        # Build clade → wave-index mapping
        level = {}
        for wi, w in enumerate(waves):
            for c in w:
                level[c] = wi
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
        from gpurec.core.scheduling import compute_clade_waves
        waves, phases = compute_clade_waves(ccp_helpers)
        assert root_clade_id in waves[-1], (
            f"root clade {root_clade_id} not found in last wave {waves[-1]}"
        )



class TestWaveStats:
    def test_stats_keys(self):
        ccp_helpers, _ = _load_helpers("test_trees_1")
        from gpurec.core.scheduling import compute_clade_waves, wave_stats
        waves, _ = compute_clade_waves(ccp_helpers)
        stats = wave_stats(waves, ccp_helpers)
        assert len(stats) == len(waves)
        for s in stats:
            for key in ("wave", "n_clades", "n_splits", "max_splits_per_clade"):
                assert key in s

    def test_total_splits_matches(self):
        """Sum of n_splits across waves equals N_splits."""
        ccp_helpers, _ = _load_helpers("test_trees_1")
        from gpurec.core.scheduling import compute_clade_waves, wave_stats
        waves, _ = compute_clade_waves(ccp_helpers)
        stats = wave_stats(waves, ccp_helpers)
        assert sum(s["n_splits"] for s in stats) == ccp_helpers["N_splits"]
