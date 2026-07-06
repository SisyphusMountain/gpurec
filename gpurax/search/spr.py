import pathlib
from collections import namedtuple

from gpurax.search.scorer import Candidate

FamilyState = namedtuple("FamilyState", "name seqfam rates")


def _materialize(fam, p, r, tmpdir, tag, warmup=False):
    """Apply an SPR move, capture its seq loglk and newick on the moved
    topology, then roll back. Returns (seq_loglk, newick_path).

    warmup=True forces the seq term to 0.0 (reconciliation-only scoring,
    mirrors GeneRax --rec-radius) without touching the seq_loglk machinery."""
    fam.apply_spr(p, r)
    seq = 0.0 if warmup else fam.seq_loglk(False)
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


def _initial_joints(families, scorer, tmpdir, warmup):
    """Score every family's starting (unmoved) topology in one batched
    call, seeding the current_joint each family's search trajectory
    advances from."""
    cands = []
    for fs in families:
        seq = 0.0 if warmup else fs.seqfam.seq_loglk(False)
        path = str(pathlib.Path(tmpdir) / f"init_{fs.name}.nwk")
        pathlib.Path(path).write_text(fs.seqfam.newick() + "\n")
        cands.append(Candidate(seq, path, fs.rates))
    scores = scorer.score_candidates(cands)
    return {fs.name: s for fs, s in zip(families, scores)}


def spr_search(families, scorer, max_radius, tmpdir, *, warmup=False):
    """Batch-native ragged multi-family SPR loop.

    Advances all families one accepted move at a time so their candidate
    evaluations co-batch into a single `scorer.score_candidates` call per
    round; retires families once no improving move remains at a radius;
    escalates radius 1..max_radius. Mutates each FamilyState.seqfam in
    place to the best tree found; returns None.

    warmup=True forces the seq term to 0.0 (reconciliation-only SPR,
    mirrors GeneRax --rec-radius / enableLibpll=false).

    Per-family trajectory is bit-for-bit the same greedy walk spr_round
    performs alone (apply single best move, re-enumerate, repeat, escalate
    radius on convergence); batching only changes when families
    synchronize their scoring calls, not what each family accepts.
    """
    current_joint = _initial_joints(families, scorer, tmpdir, warmup)

    for radius in range(1, max_radius + 1):
        active = list(families)
        while active:
            batch, owner = [], []
            for fs in active:
                moves = fs.seqfam.spr_neighbors(radius)
                for i, (p, r, _path) in enumerate(moves):
                    seq, nwk = _materialize(
                        fs.seqfam, p, r, tmpdir, f"r{radius}_{fs.name}_{i}",
                        warmup=warmup,
                    )
                    batch.append(Candidate(seq, nwk, fs.rates))
                    owner.append((fs.name, p, r))

            if not batch:
                break

            scores = scorer.score_candidates(batch)

            best_by_name = {}
            for (name, p, r), s in zip(owner, scores):
                if name not in best_by_name or s > best_by_name[name][0]:
                    best_by_name[name] = (s, p, r)

            still_active = []
            for fs in active:
                best = best_by_name.get(fs.name)
                if best is None:
                    continue  # no candidates this round for this family
                s, p, r = best
                if s > current_joint[fs.name]:
                    fs.seqfam.apply_spr(p, r)  # apply permanently, no rollback
                    current_joint[fs.name] = s
                    still_active.append(fs)
            active = still_active
