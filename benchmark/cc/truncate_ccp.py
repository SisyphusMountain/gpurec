#!/usr/bin/env python3
"""Write a coarse (truncated) copy of every `.ale` gene-family file.

WHY
---
The genewise rate fit costs roughly one gradient per clade, so a family file
with 16 000 clades costs about four times as much as one with 4 000.  A coarse
copy that keeps only the most probable splits of each clade is therefore a
cheap warm-start problem: fit the rates on the coarse files first, then hand
those rates to the full files as a starting point.

WHAT THE PARSER DOES WITH THE COUNTS
------------------------------------
Read `crates/gpurec-preprocess/src/lib.rs`, functions `build_ale_clade_data`,
`add_ale_synthetic_root_splits`, `ale_bip_denominator` and `build_ccp_arrays`.
Three facts matter, and each one was checked on a real file before this script
was written:

1. A non-root clade's splits come from `#Dip_counts`.  The parser first forms
   a raw weight `dip_count / Bip_count(parent)` (for a parent whose leaf set
   has size 1 or size n-1 it divides by `#observations` instead and never looks
   at `#Bip_counts`).  Then, in `build_ccp_arrays`, it divides every split
   weight by the sum of that parent's split weights:

       conditional probability of a split = weight / (sum of the parent's weights)

   The `Bip_count` denominator is the same for every split of one parent, so it
   cancels completely in that second division.  In other words the conditional
   split probabilities the solver sees are exactly

       dip_count / (sum of the parent's dip counts that are present in the file)

   and they therefore sum to 1 for every clade, whatever subset of splits we
   leave in the file.  The only thing `#Bip_counts` has to satisfy for a clade
   with splits is: the entry must exist and be strictly positive.

2. The ROOT clade (the set of all leaves) has no `#Dip_counts` line at all.
   The parser invents its splits: for every `#set-id` clade X that is neither
   empty nor the whole leaf set it adds a root split {X, complement of X} with
   raw weight `count(X) / (observations * 2 * (2n-3))`, where `count(X)` is
   `#observations` when X has 1 or n-1 leaves and `Bip_counts[X]` otherwise, and
   where a missing `#Bip_counts` entry counts as 0 and the split is skipped.
   The same unordered pair is produced twice (once from X, once from its
   complement) and the two weights are added together.  Two consequences:

   - The root's split set is decided by which clades have a `#set-id` line and
     a positive count, NOT by `#Dip_counts`.  Truncating the root therefore
     means deleting `#set-id` lines, and that is the only lever that actually
     removes clades from the file.
   - The complement of every clade that keeps a positive count MUST also have a
     `#set-id` line, otherwise the parser stops with "missing complement clade".
     So the set of clades we keep has to be closed under complement.

   A clade that has splits and whose size is not 1 or n-1 needs a positive
   `#Bip_counts` entry (fact 1), hence it always produces a root split, hence
   its complement is always required.  There is no way around the complement
   closure.

3. `#Dip_counts` may only reference clades that have a `#set-id` line
   (`ale_to_clade_id` is built from `#set-id` only), splits with count 0 are
   skipped, `#last_leafset_id` only has to be at least as large as the largest
   clade id still mentioned, `#Bip_bls` is ignored outright, and the leaf ids in
   `#leaf-id` must stay contiguous from 1 to n.

Measured on
data/.../coleman/ALEs_extract/ALEs/COG0001_0...ufboot.ale (1092 leaves,
10000 sampled trees, 16814 clades): every non-leaf clade's `#Bip_counts` value
equals the sum of its `#Dip_counts` values exactly; the clade set is closed
under complement; a clade and its complement carry the same count; and the
counts of all 8407 complement pairs add up to exactly
`observations * 2 * (2n-3)`, i.e. the root distribution is exactly "pick one of
the 2n-3 branches of one of the sampled trees to root on".

THE REWRITE
-----------
* `#Bip_counts` of a kept clade with splits is rewritten as the sum of that
  clade's KEPT `#Dip_counts` values.  Because the parser renormalises (fact 1)
  this cannot break the "probabilities sum to 1" property; the reason to do it
  is that it preserves the invariant the original files satisfy (Bip = sum of
  Dips), and it makes the root weight of a clade proportional to the split mass
  that clade actually kept (fact 2).  A kept leaf keeps whatever `#Bip_counts`
  line it had in the input (in this dataset leaves have none).
* At tau = 1.0 nothing is dropped, so the rewritten Bip value equals the
  original one and the output is equivalent to the input.
* One honest caveat: for a clade with 1 or n-1 leaves the parser uses
  `#observations` as the root count no matter what we write, so at tau < 1
  those "root on a terminal branch" splits keep their full weight while the
  internal ones are down-weighted by their retained mass.  This is an
  approximation, which is the whole point of a coarse warm-start file.

TWO WAYS OF MAKING THE FILE SMALLER (`--mode`)
----------------------------------------------
`--mode top` is the mass-truncation rule described below: keep each clade's
most probable splits.  It is BIASED on purpose -- the splits it throws away are
the rare, discordant ones, which are exactly the ones a transfer or a
duplication shows up in -- and measured on 200 Coleman families the optimum of
the truncated problem sits a median 0.8 to 1.3 log2 rate units away from the
real one.

`--mode thin` is the unbiased alternative: it makes the file look as if it had
been built from FEWER sampled trees.  Every conditional split frequency is
unchanged in expectation, so the optimum of the thinned problem should sit at
the real optimum plus sampling noise rather than at a systematically different
place.

THE TRUNCATION RULE, `--mode top` (one parameter, tau in (0, 1])
----------------------------------------------------------------
1. For every clade with splits: sort its splits by count descending and keep
   them in that order until the kept counts reach at least tau times the
   clade's total split count; always keep at least the first one.
2. For the root: build the list of complement pairs {X, complement of X} with
   the count the parser would use, sort descending, keep until the kept counts
   reach tau times the total, always at least the first pair.
3. Keep only the clades reachable from the root: start from the members of the
   kept root pairs, then repeatedly add the two children of every kept split of
   an already-kept clade, and the complement of every kept clade (required by
   fact 2).  Everything not reached is deleted, which is what makes the clade
   count actually drop.
4. `#constructor_string`, `#observations`, `#last_leafset_id` and `#leaf-id`
   are copied unchanged; `#Bip_bls` is dropped because the parser ignores it.

THE THINNING RULE, `--mode thin` (two parameters, keep-fraction f and a seed)
-----------------------------------------------------------------------------
The quantity that decides how much cheaper the thinned file is to solve is how
many clades disappear, and a clade disappears only when NONE of the sampled
trees it appears in is kept.  A clade that appears in B of the sampled trees
therefore survives with probability 1 - (1-f)^B, and that is the target this
rule aims at, per clade, exactly.

1. Every complement pair {X, complement of X} is one bipartition of the leaf
   set, and a sampled tree either contains that bipartition or does not, so X
   and its complement are in exactly the same trees.  ONE binomial draw per
   pair, with the pair's original count B as trials and f as success
   probability, therefore decides how many trees the pair still has:
   `trees_kept`.  Both members get the same number.
2. Given that a clade still has `trees_kept` of its B trees, the split each of
   those trees resolves the clade by is a draw WITHOUT replacement from that
   clade's original `#Dip_counts` (they add up to B, checked on the Coleman
   files), which is a multivariate hypergeometric draw.  Splits whose draw is 0
   were not seen in the smaller sample and are dropped.  `#Bip_counts` is
   rewritten as the sum of what is left, as `top` does.
3. `#observations` becomes round(f * observations).  This matters: the parser
   uses `#observations` (not `#Bip_counts`) as the root weight and the split
   denominator of a clade with 1 or n-1 leaves, so leaving it at the original
   value would make those "root on a terminal branch" splits f times too heavy
   relative to the internal ones.  Those clades are the two sides of a terminal
   branch and so are in every sampled tree; they are always kept.
4. A clade is kept when its pair still has at least one tree.  That set is
   closed under complement by construction (rule 1 gives both members the same
   number), and every kept clade gets a root split from the parser, so there is
   no separate reachability walk to do.
5. A split whose child clade was NOT kept cannot be shown by any kept tree, so
   it is dropped as well.  A kept clade all of whose splits go that way could
   not be expanded by the solver, so it is given back its largest original split
   with a count of 1 -- the smallest count that says "one sampled tree resolves
   this clade this way" -- and the two children of that split, and their
   complements, join the kept set.  That is why rule 5 runs to a fixed point.

Two honest caveats about the thinning:

* Rules 1 and 2 make each clade's own survival exact, and make the counts of one
  clade jointly right, but different bipartitions are still drawn independently,
  whereas really sub-sampling trees would couple them (one tree contributes a
  clade to every level of itself, so dropping that tree removes them all at
  once).  Rule 5 repairs the inconsistencies that leak through -- a split with
  trees left whose child has none -- in the only direction that keeps the clade
  count honest, by dropping the split.  It is an approximation of sub-sampling,
  not sub-sampling.
* In expectation nothing is biased: the expected value of every kept count is
  f times the original, so the expected conditional split probability
  (count divided by the sum of the parent's counts) is unchanged.  What changes
  is the sampling noise, which grows as 1 / sqrt(f * count).

Counts are rounded to the nearest whole number before the draw (the binomial and
the hypergeometric need whole numbers); every count in the Coleman files is
already a whole number.
"""

from __future__ import annotations

import argparse
import os
import time
import zlib
from multiprocessing import Pool

import numpy

# Number of worker processes.  Not a setting: it is fixed by the shape of the
# job (a few thousand independent files, each a short burst of text parsing) and
# by the machine this benchmark is run on.
WORKER_PROCESSES = 16

# 64-bit mask for the leaf-set fingerprint arithmetic below.  Not a setting.
MASK64 = (1 << 64) - 1


def leaf_fingerprint(leaf_id):
    """Return a fixed 64-bit number for one leaf id (SplitMix64).

    Leaf sets are compared by exclusive-or of their leaves' fingerprints: the
    fingerprint of the complement of a clade is then the fingerprint of the
    whole leaf set exclusive-or'd with the clade's own fingerprint, which is how
    complement pairs are found without ever materialising a 1000-bit leaf mask.
    The function is a fixed permutation, so runs are reproducible.
    """
    z = (leaf_id + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def format_count(value):
    """Write a count back the way ALE writes it: as an integer when it is one."""
    if value == int(value):
        return str(int(value))
    return repr(value)


def scan_ale(path):
    """First pass: read everything except the bulky `#set-id` leaf lists.

    Returns a dictionary with the header text, the counts, and for every
    `#set-id` clade only its leaf-set size and its 64-bit fingerprint, so the
    memory used stays proportional to the number of clades rather than to the
    number of leaf ids in the file (the `#set-id` section is most of the bytes).
    """
    constructor = None
    observations_text = None
    observations = None
    last_leafset_text = None
    leaf_lines = []
    leaf_ids = []
    bip_counts = {}
    bip_text = {}
    dip_counts = {}
    clade_size = {}
    clade_fingerprint = {}
    fingerprint_cache = {}
    section = ""

    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                section = line
                if section == "#END":
                    break
                continue
            if section == "#constructor_string":
                constructor = line
            elif section == "#observations":
                observations_text = line
                observations = float(line)
            elif section == "#Bip_counts":
                clade_token, count_token = line.split()
                clade_id = int(clade_token)
                bip_counts[clade_id] = float(count_token)
                bip_text[clade_id] = count_token
            elif section == "#Dip_counts":
                parent_token, left_token, right_token, count_token = line.split()
                parent = int(parent_token)
                entry = (int(left_token), int(right_token), float(count_token))
                if parent in dip_counts:
                    dip_counts[parent].append(entry)
                else:
                    dip_counts[parent] = [entry]
            elif section == "#last_leafset_id":
                last_leafset_text = line
            elif section == "#leaf-id":
                leaf_lines.append(line)
                leaf_ids.append(int(line.split()[1]))
            elif section == "#set-id":
                clade_token, leaf_list = line.split(":", 1)
                clade_id = int(clade_token.strip())
                fingerprint = 0
                size = 0
                for token in leaf_list.split():
                    value = fingerprint_cache.get(token)
                    if value is None:
                        value = leaf_fingerprint(int(token))
                        fingerprint_cache[token] = value
                    fingerprint ^= value
                    size += 1
                clade_fingerprint[clade_id] = fingerprint
                clade_size[clade_id] = size

    if constructor is None or observations is None or not leaf_lines or not clade_size:
        raise ValueError(f"{path}: not a complete ALE file")

    return {
        "constructor": constructor,
        "observations_text": observations_text,
        "observations": observations,
        "last_leafset_text": last_leafset_text,
        "leaf_lines": leaf_lines,
        "leaf_ids": leaf_ids,
        "bip_counts": bip_counts,
        "bip_text": bip_text,
        "dip_counts": dip_counts,
        "clade_size": clade_size,
        "clade_fingerprint": clade_fingerprint,
    }


def build_complement_map(clade_size, clade_fingerprint, leaf_ids):
    """Map every clade id to the clade id holding the complementary leaf set."""
    whole_set_fingerprint = 0
    for leaf_id in leaf_ids:
        whole_set_fingerprint ^= leaf_fingerprint(leaf_id)
    leaf_count = len(leaf_ids)

    by_leaf_set = {}
    for clade_id, fingerprint in clade_fingerprint.items():
        by_leaf_set[(fingerprint, clade_size[clade_id])] = clade_id

    complement = {}
    for clade_id, fingerprint in clade_fingerprint.items():
        key = (whole_set_fingerprint ^ fingerprint, leaf_count - clade_size[clade_id])
        complement[clade_id] = by_leaf_set.get(key)
    return complement, leaf_count


def surviving_splits(clade_size, dip_counts):
    """Drop zero-count splits and splits whose children cannot themselves be built.

    A clade is "usable" when it is a single leaf, or when at least one of its
    splits has a positive count and two usable children.  Children are always
    strictly smaller than their parent, so walking the clades in order of
    increasing leaf-set size settles this in one pass.  On the Coleman files
    nothing is ever dropped here (every non-leaf clade has splits with positive
    counts), but the pass guarantees that the file we write never contains a
    clade the solver cannot expand.
    """
    usable_splits = {}
    usable = set()
    for clade_id in sorted(clade_size, key=lambda cid: clade_size[cid]):
        if clade_size[clade_id] == 1:
            usable.add(clade_id)
            continue
        kept = [
            (left, right, count)
            for (left, right, count) in dip_counts.get(clade_id, ())
            if count > 0.0 and left in usable and right in usable
        ]
        if kept:
            usable_splits[clade_id] = kept
            usable.add(clade_id)
    return usable_splits, usable


def keep_top_counts(entries, count_of, tau):
    """Sort `entries` by count descending and keep a prefix reaching tau of the total.

    Always returns at least one entry.  `count_of` reads the count out of one
    entry; ties are broken by the entry itself so the result does not depend on
    the input order.
    """
    ordered = sorted(entries, key=lambda entry: (-count_of(entry), entry))
    total = sum(count_of(entry) for entry in ordered)
    target = tau * total
    kept = []
    running = 0.0
    for entry in ordered:
        kept.append(entry)
        running += count_of(entry)
        if running >= target:
            break
    return kept


def truncate_one_family(in_path, out_path, tau):
    """Write the coarse copy of one family.  Returns (original clades, kept clades)."""
    scanned = scan_ale(in_path)
    clade_size = scanned["clade_size"]
    dip_counts = scanned["dip_counts"]
    bip_counts = scanned["bip_counts"]
    observations = scanned["observations"]

    complement, leaf_count = build_complement_map(
        clade_size, scanned["clade_fingerprint"], scanned["leaf_ids"]
    )
    usable_splits, usable = surviving_splits(clade_size, dip_counts)

    # Rule 1: keep the most probable splits of every clade.
    kept_splits = {
        clade_id: keep_top_counts(splits, lambda entry: entry[2], tau)
        for clade_id, splits in usable_splits.items()
    }

    # Rule 2: keep the most probable root splits.  A root split is a pair
    # {clade, complement of clade}; its count is the sum of the two clades'
    # counts, exactly as the parser accumulates them.
    def root_count(clade_id):
        size = clade_size[clade_id]
        if size == 1 or size + 1 == leaf_count:
            return observations
        return bip_counts.get(clade_id, 0.0)

    pair_count = {}
    for clade_id in clade_size:
        other = complement.get(clade_id)
        if other is None or clade_id not in usable or other not in usable:
            continue
        pair = (clade_id, other) if clade_id < other else (other, clade_id)
        pair_count[pair] = root_count(pair[0]) + root_count(pair[1])
    if not pair_count:
        raise ValueError(f"{in_path}: no usable root split")
    kept_pairs = keep_top_counts(
        list(pair_count.items()), lambda entry: entry[1], tau
    )

    # Rule 3: keep only what the root can reach, closed under complement.
    kept_clades = set()
    frontier = []
    for (first, second), _count in kept_pairs:
        for clade_id in (first, second):
            if clade_id not in kept_clades:
                kept_clades.add(clade_id)
                frontier.append(clade_id)
    while frontier:
        clade_id = frontier.pop()
        other = complement.get(clade_id)
        if other is None:
            raise ValueError(
                f"{in_path}: clade {clade_id} has no complement in #set-id; "
                "the parser would reject this file"
            )
        if other not in kept_clades:
            kept_clades.add(other)
            frontier.append(other)
        for left, right, _count in kept_splits.get(clade_id, ()):
            for child in (left, right):
                if child not in kept_clades:
                    kept_clades.add(child)
                    frontier.append(child)

    return write_family(
        in_path=in_path, out_path=out_path, scanned=scanned,
        kept_clades=kept_clades, kept_splits=kept_splits,
        observations_text=scanned["observations_text"],
    )


def write_family(in_path, out_path, scanned, kept_clades, kept_splits, observations_text):
    """Write one coarse `.ale` file and return (original clades, kept clades,
    original `#Dip_counts` lines, written `#Dip_counts` lines).

    `kept_clades` is the set of clade ids that keep a `#set-id` line and
    `kept_splits` maps a kept clade id to the splits it keeps.  Both truncation
    modes end here, so the two produce byte-identical files whenever they decide
    on the same clades and splits.
    """
    clade_size = scanned["clade_size"]

    # A clade dragged in only as a complement may itself be unusable (no split
    # with two buildable children).  It then gets a `#set-id` line with no
    # `#Bip_counts` and no `#Dip_counts`, which is exactly the shape the parser
    # treats as "registered, but no root split and no splits of its own".  On
    # the Coleman files this never happens: every non-leaf clade is usable.
    written_splits = 0
    with open(out_path, "w") as out:
        out.write("#constructor_string\n")
        out.write(scanned["constructor"] + "\n")
        out.write("#observations\n")
        out.write(observations_text + "\n")

        out.write("#Bip_counts\n")
        for clade_id in sorted(kept_clades):
            splits = kept_splits.get(clade_id)
            if splits is None:
                # A leaf, or a clade kept only so that a complement lookup
                # succeeds: copy whatever the input had, or write nothing.
                original = scanned["bip_text"].get(clade_id)
                if original is not None and clade_size[clade_id] == 1:
                    out.write(f"{clade_id}\t{original}\n")
                continue
            out.write(f"{clade_id}\t{format_count(sum(c for _l, _r, c in splits))}\n")

        out.write("#Dip_counts\n")
        for clade_id in sorted(kept_clades):
            for left, right, count in kept_splits.get(clade_id, ()):
                out.write(f"{clade_id}\t{left}\t{right}\t{format_count(count)}\n")
                written_splits += 1

        if scanned["last_leafset_text"] is not None:
            out.write("#last_leafset_id\n")
            out.write(scanned["last_leafset_text"] + "\n")

        out.write("#leaf-id\n")
        for line in scanned["leaf_lines"]:
            out.write(line + "\n")

        # Second pass over the input, only to copy the kept `#set-id` lines.
        out.write("#set-id\n")
        section = ""
        with open(in_path, "r") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    section = line
                    if section == "#END":
                        break
                    continue
                if section != "#set-id":
                    continue
                clade_token, _rest = line.split(":", 1)
                if int(clade_token.strip()) in kept_clades:
                    out.write(line + "\n")
        out.write("#END\n")

    original_splits = sum(len(v) for v in scanned["dip_counts"].values())
    return len(clade_size), len(kept_clades), original_splits, written_splits


def thin_one_family(in_path, out_path, keep_fraction, seed):
    """Write the thinned copy of one family (`--mode thin`).

    Returns (original clades, kept clades, original splits, kept splits).
    """
    scanned = scan_ale(in_path)
    clade_size = scanned["clade_size"]
    dip_counts = scanned["dip_counts"]
    observations = scanned["observations"]

    complement, leaf_count = build_complement_map(
        clade_size, scanned["clade_fingerprint"], scanned["leaf_ids"]
    )
    usable_splits, usable = surviving_splits(clade_size, dip_counts)

    # One generator per file, seeded from the run's seed and the file's name, so
    # that a family's draw does not depend on how many workers ran or in which
    # order, and so that two families never share a stream.
    rng = numpy.random.default_rng(
        [seed, zlib.crc32(os.path.basename(in_path).encode("utf-8"))]
    )

    # Step 3 first, because steps 1 and 2 need the new tree count: the smaller
    # sample has fewer trees in it.
    new_observations = float(round(keep_fraction * observations))
    if new_observations < 1.0:
        raise ValueError(
            f"{in_path}: keep-fraction {keep_fraction} leaves "
            f"round({keep_fraction} * {observations}) = 0 sampled trees"
        )

    def on_a_terminal_branch(clade_id):
        """True for the two sides of a terminal branch (1 leaf and n-1 leaves).

        Every sampled tree contains them, and the parser weights them by
        `#observations` instead of by `#Bip_counts`, so they are always kept.
        """
        size = clade_size[clade_id]
        return size == 1 or size + 1 == leaf_count

    # Step 1: how many of a bipartition's sampled trees are still in the smaller
    # sample.  One draw for the whole complement pair, because the two sides of a
    # bipartition are in exactly the same trees.  The number of trees a clade is
    # in is the sum of its split counts (equal to its `#Bip_counts` entry on the
    # Coleman files, checked); a leaf has no splits and reads its complement's.
    trees_of_clade = {
        clade_id: int(round(sum(count for _l, _r, count in splits)))
        for clade_id, splits in usable_splits.items()
    }
    trees_kept = {}
    for clade_id in sorted(clade_size):
        if clade_id in trees_kept:
            continue
        other = complement.get(clade_id)
        if other is None:
            continue
        trials = max(trees_of_clade.get(clade_id, 0), trees_of_clade.get(other, 0))
        drawn = int(rng.binomial(trials, keep_fraction)) if trials > 0 else 0
        trees_kept[clade_id] = drawn
        trees_kept[other] = drawn

    # Step 2: which split each of the kept trees resolves the clade by -- a draw
    # without replacement from the clade's original split counts.
    thinned_splits = {}
    for clade_id in sorted(usable_splits):
        splits = usable_splits[clade_id]
        urn = numpy.rint([count for _l, _r, count in splits]).astype(numpy.int64)
        sample = min(trees_kept.get(clade_id, 0), int(urn.sum()))
        if sample <= 0:
            thinned_splits[clade_id] = []
            continue
        draws = rng.multivariate_hypergeometric(urn, sample)
        thinned_splits[clade_id] = [
            (splits[i][0], splits[i][1], float(draws[i]))
            for i in range(len(splits))
            if draws[i] > 0
        ]

    # Step 4: the clades the smaller sample still contains.  Both members of a
    # pair carry the same `trees_kept`, so this set is closed under complement.
    kept_clades = set()
    for clade_id in clade_size:
        if clade_id not in usable:
            continue
        if on_a_terminal_branch(clade_id) or trees_kept.get(clade_id, 0) > 0:
            kept_clades.add(clade_id)
    if not kept_clades:
        raise ValueError(f"{in_path}: no clade survives thinning at {keep_fraction}")

    # Step 5: a split pointing at a clade the smaller sample does not contain
    # cannot be shown by any kept tree, so it goes too; a kept clade left with no
    # split at all gets its largest original split back with a count of 1, and
    # that split's children and their complements join the kept set, which is
    # why this runs to a fixed point.
    kept_splits = {}
    growing = True
    while growing:
        growing = False
        for clade_id in sorted(kept_clades):
            if clade_id in kept_splits or clade_id not in usable_splits:
                continue
            splits = [
                (left, right, count)
                for (left, right, count) in thinned_splits[clade_id]
                if left in kept_clades and right in kept_clades
            ]
            if not splits:
                largest = max(usable_splits[clade_id], key=lambda entry: (entry[2], entry))
                splits = [(largest[0], largest[1], 1.0)]
            kept_splits[clade_id] = splits
            for left, right, _count in splits:
                for child in (left, right):
                    if child in kept_clades:
                        continue
                    other = complement.get(child)
                    if other is None:
                        raise ValueError(
                            f"{in_path}: clade {child} has no complement in #set-id; "
                            "the parser would reject this file"
                        )
                    kept_clades.add(child)
                    kept_clades.add(other)
                    growing = True

    return write_family(
        in_path=in_path, out_path=out_path, scanned=scanned,
        kept_clades=kept_clades, kept_splits=kept_splits,
        observations_text=format_count(new_observations),
    )


def one_family_job(job):
    """Pool worker entry point: unpack the one tuple `Pool.map` can hand over."""
    mode, in_path, out_path, tau, keep_fraction, seed = job
    if mode == "top":
        counts = truncate_one_family(in_path, out_path, tau)
    else:
        counts = thin_one_family(in_path, out_path, keep_fraction, seed)
    return (out_path,) + counts


def read_family_list(families_path, limit):
    """Read the input list, one absolute family path per line."""
    with open(families_path, "r") as handle:
        paths = [line.strip() for line in handle if line.strip()]
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"{families_path}: no family paths to process")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Write coarse (truncated) copies of ALE gene-family files."
    )
    parser.add_argument("--families", required=True, help="file listing one .ale path per line")
    parser.add_argument("--limit", type=int, required=True, help="0 = all, else first N families")
    parser.add_argument("--mode", required=True, choices=("top", "thin"),
                        help="top = keep each clade's most probable splits (needs --tau); "
                             "thin = binomial sub-sample of the counts (needs --keep-fraction and --seed)")
    parser.add_argument("--tau", type=float, help="mode top: kept split-count fraction in (0, 1]")
    parser.add_argument("--keep-fraction", type=float,
                        help="mode thin: probability a sampled tree survives, in (0, 1]")
    parser.add_argument("--seed", type=int, help="mode thin: seed of the binomial draws")
    parser.add_argument("--out-dir", required=True, help="directory for the truncated files")
    args = parser.parse_args()

    # Every parameter of the mode that runs must be stated, and a parameter that
    # belongs to the other mode must not: a run's command line then says exactly
    # what produced the files.
    if args.mode == "top":
        if args.tau is None:
            raise SystemExit("--mode top needs --tau")
        if args.keep_fraction is not None or args.seed is not None:
            raise SystemExit("--keep-fraction and --seed belong to --mode thin")
        if not 0.0 < args.tau <= 1.0:
            raise SystemExit(f"--tau must be in (0, 1], got {args.tau}")
    else:
        if args.keep_fraction is None or args.seed is None:
            raise SystemExit("--mode thin needs --keep-fraction and --seed")
        if args.tau is not None:
            raise SystemExit("--tau belongs to --mode top")
        if not 0.0 < args.keep_fraction <= 1.0:
            raise SystemExit(f"--keep-fraction must be in (0, 1], got {args.keep_fraction}")

    in_paths = read_family_list(args.families, args.limit)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    jobs = [
        (args.mode, path, os.path.join(out_dir, os.path.basename(path)),
         args.tau, args.keep_fraction, args.seed)
        for path in in_paths
    ]

    started = time.perf_counter()
    with Pool(processes=WORKER_PROCESSES) as pool:
        results = pool.map(one_family_job, jobs, chunksize=1)
    elapsed = time.perf_counter() - started

    with open(os.path.join(out_dir, "families.txt"), "w") as handle:
        for row in results:
            handle.write(row[0] + "\n")

    original_clades = sum(row[1] for row in results)
    kept_clades = sum(row[2] for row in results)
    original_splits = sum(row[3] for row in results)
    kept_splits = sum(row[4] for row in results)
    print(f"families         {len(results)}")
    print(f"mode             {args.mode}")
    if args.mode == "top":
        print(f"tau              {args.tau}")
    else:
        print(f"keep fraction    {args.keep_fraction}")
        print(f"seed             {args.seed}")
    print(f"original clades  {original_clades}")
    print(f"kept clades      {kept_clades}")
    print(f"clade ratio      {kept_clades / original_clades:.4f}")
    print(f"original splits  {original_splits}")
    print(f"kept splits      {kept_splits}")
    print(f"split ratio      {kept_splits / original_splits:.4f}")
    print(f"seconds          {elapsed:.1f}")
    print(f"out dir          {out_dir}")


if __name__ == "__main__":
    main()
