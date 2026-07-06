import pathlib

from gpurax.search.scorer import Candidate


def _materialize(fam, p, r, tmpdir, tag):
    """Apply an SPR move, capture its seq loglk and newick on the moved
    topology, then roll back. Returns (seq_loglk, newick_path)."""
    fam.apply_spr(p, r)
    seq = fam.seq_loglk(False)
    path = str(pathlib.Path(tmpdir) / f"cand_{tag}.nwk")
    pathlib.Path(path).write_text(fam.newick() + "\n")
    fam.rollback()
    return seq, path


def spr_round(fam, rates, scorer, radius, tmpdir, current_joint):
    moves = fam.spr_neighbors(radius)
    if not moves:
        return False, current_joint

    cands, keys = [], []
    for i, (p, r, _path) in enumerate(moves):
        seq, nwk = _materialize(fam, p, r, tmpdir, i)
        cands.append(Candidate(seq_loglk=seq, newick_path=nwk, rates=rates))
        keys.append((p, r))

    scores = scorer.score_candidates(cands)
    best = max(range(len(scores)), key=lambda i: scores[i])

    if scores[best] > current_joint:
        p, r = keys[best]
        fam.apply_spr(p, r)  # apply permanently, no rollback
        return True, scores[best]

    return False, current_joint
