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

    def score_candidates(self, candidates):
        recs = self._rec.score(
            [c.newick_path for c in candidates],
            [c.rates for c in candidates],
        )
        return [c.seq_loglk + self._rw * r for c, r in zip(candidates, recs)]
