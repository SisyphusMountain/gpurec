import time
from collections import namedtuple

from gpurax.recon.adapter import ReconBatch

# seq_loglk: sequence log-likelihood (nats) already evaluated on this candidate's
# own topology (the topology produced by the SPR move) — must be captured at
# move-application time, not recomputed here, since a candidate's topology
# differs from the seqfam's current tree.
Candidate = namedtuple("Candidate", "seq_loglk newick_path rates")


class JointScorer:
    def __init__(self, species_path, rec_weight=1.0, device="cuda"):
        self._rec = ReconBatch(species_path, device=device)
        self._rw = rec_weight
        # Phase-2 trigger (task I3): wall-clock time spent on each term, so we
        # can measure whether porting the sequence term (currently CPU/libpll,
        # timed at SPR-candidate materialization in gpurax/search/spr.py) to
        # GPU is worthwhile relative to the batched GPU reconciliation term
        # (timed here). Accumulated across calls for the life of the scorer.
        self.timings = {"seq_s": 0.0, "rec_s": 0.0}

    def score_candidates(self, candidates):
        t0 = time.perf_counter()
        recs = self._rec.score(
            [c.newick_path for c in candidates],
            [c.rates for c in candidates],
        )
        self.timings["rec_s"] += time.perf_counter() - t0
        return [c.seq_loglk + self._rw * r for c, r in zip(candidates, recs)]
