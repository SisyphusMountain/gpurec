# Clade Wave Scheduling Problem

## Context

We are computing a probabilistic reconciliation of gene trees against a species
tree.  The core computation is evaluating **Pi[c, s]** — the probability that
clade `c` reconciles at species branch `s` — for all clades `c` in a gene
family.  This is done on a GPU with a fixed kernel width of **W** (currently
256): each GPU kernel invocation computes Pi for up to W clades simultaneously.

Pi[c] depends only on Pi[l] and Pi[r] for every split `(c, l, r)` in the
Conditional Clade Probability (CCP) distribution of the gene family.  This
induces a **DAG** of dependencies: clade `c` can only be computed once all its
children have been computed.

The problem is to partition the clades into an ordered sequence of **batches**
(waves), each of size ≤ W, respecting the dependency order.

---

## Formal Problem Statement

**Input:**
- A directed acyclic graph G = (V, E) where
  - V = {0, …, C−1} are clades (|V| = C ≈ 4,000–10,000)
  - (c, p) ∈ E means "clade p has child clade c", i.e. p must be computed
    after c.  (Edges point from children to parents.)
  - Each clade c has a weight w(c) = split_count(c) ≥ 0 (number of splits,
    i.e. the compute load of the Pi kernel for c).
- A batch capacity W (e.g. 256).

**Output:** An assignment of each clade to a batch index level(c) ∈ {0,…,K−1}
such that:
1. **Validity:** if (c, p) ∈ E then level(p) > level(c).
2. **Capacity:** |{c : level(c) = k}| ≤ W for all k.

**Primary objective:** Minimise K (number of batches = GPU latency).

**Secondary objective:** For each batch k, minimise

    required_clades(k) = |{ c : level(c) < k  AND  ∃ p with level(p)=k, (c,p)∈E }|

i.e. the number of distinct "input" clades that must be resident in memory
when computing batch k.  Summing over k gives total memory traffic; the
maximum over k bounds peak memory.

---

## Structure of the DAG

The DAG is the CCP structure derived from one (or more) gene trees.  Key
properties:

- **Source nodes** (in-degree 0): leaf clades, i.e. split_count(c) = 0.
  Typically ~1,500 leaf clades for a 1,000-species gene tree.
- **Sink node** (out-degree 0): the root / ubiquitous clade (one per family).
- **Depth** D = length of the longest path from any leaf to the root.
  For our 1,000-species families D ≈ 40–50.
- The DAG is NOT a tree in general: a clade can be a child of many parents
  through different splits (especially with horizontal gene transfers).
- The weight distribution w(c) is very skewed: the ubiquitous clade has
  w(c) ≈ C/2 (thousands of splits), while most leaf clades have w(c) = 0.

**Scale (one typical family):** C ≈ 6,000, N_splits ≈ 7,500, D ≈ 42.

---

## Lower Bounds on K

The minimum number of batches satisfies:

    K ≥ ⌈C / W⌉        (packing bound: 6000/256 ≈ 24)
    K ≥ D + 1           (critical-path bound: ≈ 43)

So K* ≥ 43 for our typical instances.

---

## Algorithms Tried and Results

### 1. BFS levels + split at W (baseline)

Assign clades to BFS levels (Kahn topological sort), then split each level
into chunks of W.  Simple, O(C + N), but does not mix levels.

### 2. Greedy LIFO (newly-ready parents pushed to front of deque)

Newly-ready parents are pushed to the front, producing DFS-like ordering.
Some interleaving of levels.

### 3. Hu's critical-path priority (longest path to sink)

Max-heap keyed by bl(c) = longest path from c to root.  This is optimal for
in-trees (Hu 1961).  However, in our CCP DAG **all leaves have the same bl**
(= D), so the priority queue degenerates to BFS for the leaf phase.

### 4. Eager-parent + sibling grouping (current best)

- **Sibling grouping:** For each leaf, find its "closest binary parent" (the
  parent with the fewest children, usually 2).  Use this parent ID as a
  secondary priority, so sibling leaves are adjacent in the queue and land in
  the same batch, unlocking their parent immediately.
- **Eager parents:** Internal clades get higher priority than remaining
  leaves: `priority = (is_internal, bl, sibling_key, clade_id)`.  Parents are
  processed in the very next batch after they become ready, creating a
  pipeline.

### Comparison (W = 256, 1,000 gene families, ~6,000 clades each)

| Algorithm | K mean | K min | K max | Clades/batch |
|---|---|---|---|---|
| BFS + split | 57.6 | 49 | 77 | 111.4 |
| Greedy LIFO | 52.9 | 48 | 60 | 125.8 |
| Hu (critical path) | 52.8 | 49 | 58 | 126.0 |
| Eager-parent + sibling | **51.9** | **46** | 71 | **123.6** |
| **Lower bound** | ≥43 | | | |

### Structural analysis (family_0000: C = 6071, D = 42)

BFS level sizes: [1519, 514, 308, 185, 133, 97, 78, 51, 41, 29, 20, 14, 12,
9, 8, …, 68, 36, 22, 4, 1].

Children per clade: **4551 binary** (2 children) + **1 root** (6070 children).
So the DAG is essentially a binary DAG with one mega-sink.

Best algorithm achieves K = 49 for this family:

    BFS levels:    6 + 3 + 2 + 1 + 1 + ... + 1  =  50  (split-only)
    With mixing:   first 10 batches are full (256), then the tree narrows
    Pipeline gain: ~1 batch saved from interleaving
    Achieved:      49 batches  (vs lower bound 42, gap = 7)

The gap of 7 is exactly the surplus batches needed to split the three
bottom-heavy levels (1519 → 6, 514 → 3, 308 → 2: extra 5 + 2 + 1 = 8,
minus 1 from pipeline).

### Root cause of remaining gap

The DAG has a **diamond shape**: wide at the leaves (1519), narrow in the
middle (as few as 8 clades at a level), then slightly wider near the root.
The narrow waist creates 25+ batches with <100 clades.  These cannot be
merged because each depends on the previous level.

The lower bound K* = D+1 = 43 is achievable only if every batch is full
(≥ W clades).  But levels 3–35 each have < W clades, and no interleaving
can help because all their predecessors must be completed first.

---

## Questions for the Mathematician

1. **Optimality gap:** Is there a polynomial-time algorithm that achieves
   K = K* = D + 1 for DAGs, or is finding an optimal schedule NP-hard for
   general DAGs?  What about for CCP DAGs specifically (which are derived from
   trees and have bounded tree-width)?

2. **Better heuristic:** What ordering of the initial leaves (and of
   newly-ready internal nodes) minimises the number of batches K?  Specifically,
   what property of the initial deque ordering guarantees that a large fraction
   of batches are at full capacity W?

3. **required_clades minimisation:** Given a fixed batch assignment (fixed K,
   fixed level(c) for each c), what reordering of clades *within* their slack
   window [earliest(c), latest(c)] minimises max_k required_clades(k)?  Is
   this a tractable problem?

4. **Joint optimisation:** Is there a scheduling algorithm that simultaneously
   (or near-simultaneously) minimises K and max_k required_clades(k)?

---

## Relevant Literature Pointers

- Precedence-constrained scheduling / Hu's algorithm (unit-time jobs on
  trees, P|tree,p_j=1|C_max is polynomial).
- List scheduling (Graham, 1969): approximation ratio 2 - 1/m for general
  precedence constraints.
- Memory-constrained scheduling / I/O-complexity of tree computations.
- Pebbling games on DAGs (Pebbling number = minimum memory).
